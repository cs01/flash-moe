#!/usr/bin/env python3
"""
OpenAI-compatible API server for Flash-MoE inference engine.

Keeps a persistent ./chat subprocess warm — model loads once at startup.
Subsequent requests get immediate token generation with no cold start.

Usage:
    cd metal_infer && uv run server.py [--port 8000] [--2bit]

Then: export OPENAI_BASE_URL=http://localhost:8000/v1
"""

import argparse
import json
import os
import re
import select
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from http.server import HTTPServer, BaseHTTPRequestHandler
from threading import Lock

CHAT_BIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chat")
ENCODE_BIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "encode_prompt.py")
MODEL_NAME = "qwen3.5-397b-a17b"


class InferenceEngine:
    """Persistent chat subprocess. Model stays warm between requests."""

    def __init__(self, use_2bit=False, cache_entries=0):
        self.lock = Lock()
        self.proc = None
        self.use_2bit = use_2bit
        self.cache_entries = cache_entries
        self._start()

    def _start(self):
        cmd = [CHAT_BIN, "--cache-entries", str(self.cache_entries)]
        if self.use_2bit:
            cmd.append("--2bit")
        print(f"[engine] Starting: {' '.join(cmd)}")
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        # Wait for "Ready to chat" or "> " prompt
        self._wait_for_ready()
        print("[engine] Model loaded. Ready for requests.")

    def _wait_for_ready(self):
        """Read stderr until we see the ready prompt."""
        while True:
            line = self.proc.stderr.readline()
            if not line:
                break
            text = line.decode("utf-8", errors="replace")
            sys.stderr.write(f"  {text}")
            if "Ready to chat" in text or "ready" in text.lower():
                # Consume the "> " prompt from stdout
                time.sleep(0.1)
                # Drain any pending stdout
                while select.select([self.proc.stdout], [], [], 0.1)[0]:
                    self.proc.stdout.read(1)
                break

    def _wait_for_prompt(self):
        """Read stdout until we see '> ' prompt, return everything before it."""
        buf = b""
        while True:
            byte = self.proc.stdout.read(1)
            if not byte:
                break
            buf += byte
            # Check for prompt pattern: newline + "> "
            if buf.endswith(b"\n> ") or buf.endswith(b"\n\n> "):
                return buf[:-3] if buf.endswith(b"\n> ") else buf[:-4]
            # Also check at start (first prompt)
            if buf == b"> ":
                return b""
        return buf

    def generate(self, user_text, max_tokens=1024, stream_callback=None):
        """Send a message and stream back the response."""
        with self.lock:
            if self.proc.poll() is not None:
                print("[engine] Process died, restarting...")
                self._start()

            # Send the user message (chat binary reads a line from stdin)
            message = user_text.strip() + "\n"
            self.proc.stdin.write(message.encode("utf-8"))
            self.proc.stdin.flush()

            # Read response tokens until we see the stats line and next prompt
            full_response = ""
            stats_line = ""

            while True:
                byte = self.proc.stdout.read(1)
                if not byte:
                    break

                char = byte.decode("utf-8", errors="replace")
                full_response += char

                # Stream each character
                if stream_callback and char:
                    stream_callback(char)

                # Check for end: stats line followed by prompt
                # Pattern: "\n\n[N tokens, X.XX tok/s, Y.Ys]\n\n> "
                if full_response.endswith("\n> "):
                    # Strip the trailing prompt
                    full_response = full_response[:-3].rstrip()
                    break

            # Extract actual text (strip stats line at end)
            lines = full_response.rsplit("\n", 2)
            text = full_response
            for i in range(len(lines)-1, -1, -1):
                if re.match(r'\[\d+ tokens,', lines[i].strip()):
                    text = "\n".join(lines[:i]).rstrip()
                    stats_line = lines[i].strip()
                    break
                # Also strip [prefill] lines
                if "[prefill]" in lines[i] or "[cache]" in lines[i]:
                    text = "\n".join(lines[:i]).rstrip()
                    break

            # Clean up thinking tokens for API output
            # Keep them in a separate field if present
            thinking = ""
            clean_text = text
            think_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
            if think_match:
                thinking = think_match.group(1).strip()
                clean_text = text[:think_match.start()] + text[think_match.end():]
                clean_text = clean_text.strip()

            return {
                "text": clean_text,
                "thinking": thinking,
                "stats": stats_line,
                "raw": text,
            }

    def clear(self):
        """Reset conversation state."""
        with self.lock:
            self.proc.stdin.write(b"/clear\n")
            self.proc.stdin.flush()
            # Wait for the cleared message and prompt
            time.sleep(0.2)
            while select.select([self.proc.stdout], [], [], 0.1)[0]:
                self.proc.stdout.read(4096)


# Global engine instance
engine = None


def format_messages_to_turns(messages, tools=None):
    """Convert OpenAI messages to sequential chat turns for the persistent process.

    Returns list of (role, text) pairs. The chat binary handles one user turn at a time,
    so for multi-turn conversations we need to replay history.

    For simplicity with the persistent subprocess: we clear and replay all messages
    as a single concatenated prompt for the final user turn.
    """
    # Build the full prompt text that encode_prompt.py would create
    parts = []

    tool_text = ""
    if tools:
        tool_defs = []
        for tool in tools:
            if tool.get("type") == "function":
                fn = tool["function"]
                tool_defs.append({
                    "type": "function",
                    "function": {
                        "name": fn["name"],
                        "description": fn.get("description", ""),
                        "parameters": fn.get("parameters", {}),
                    }
                })
        if tool_defs:
            tool_text = (
                "\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
                "You are provided with function signatures within <tools></tools> XML tags:\n<tools>\n"
                + "\n".join(json.dumps(t) for t in tool_defs)
                + "\n</tools>\n\nFor each function call, return a json object with function name and "
                "arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n"
                '{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call>'
            )

    # Find the last user message — that's what we send to the chat process
    last_user_msg = ""
    for msg in messages:
        if msg["role"] == "user":
            last_user_msg = msg.get("content", "")

    return last_user_msg


def parse_tool_calls(text):
    """Parse <tool_call>...</tool_call> blocks from model output."""
    tool_calls = []
    pattern = r'<tool_call>\s*(\{.*?\})\s*</tool_call>'
    matches = re.findall(pattern, text, re.DOTALL)
    for match in matches:
        try:
            data = json.loads(match)
            tool_calls.append({
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "type": "function",
                "function": {
                    "name": data.get("name", ""),
                    "arguments": json.dumps(data.get("arguments", {})),
                }
            })
        except json.JSONDecodeError:
            pass
    return tool_calls


class APIHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        sys.stderr.write(f"[api] {args[0]}\n")

    def do_GET(self):
        if self.path == "/v1/models":
            self.send_json({
                "object": "list",
                "data": [{"id": MODEL_NAME, "object": "model", "owned_by": "local"}]
            })
        elif self.path == "/health":
            self.send_json({"status": "ok", "model": MODEL_NAME})
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self.send_error(404)
            return

        content_length = int(self.headers.get('Content-Length', 0))
        body = json.loads(self.rfile.read(content_length))

        messages = body.get("messages", [])
        tools = body.get("tools", None)
        max_tokens = body.get("max_tokens", 1024)
        stream = body.get("stream", False)

        user_text = format_messages_to_turns(messages, tools)
        if not user_text:
            self.send_json({"error": "No user message found"}, status=400)
            return

        request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())

        # Clear conversation for stateless API behavior
        # (each request is independent — use the chat TUI for multi-turn)
        engine.clear()

        if stream:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            buffer = []
            in_think = False

            def on_char(ch):
                nonlocal in_think
                buffer.append(ch)
                text_so_far = "".join(buffer)

                # Don't stream thinking tokens
                if "<think>" in text_so_far and not in_think:
                    in_think = True
                    return
                if "</think>" in text_so_far and in_think:
                    in_think = False
                    buffer.clear()
                    return
                if in_think:
                    return

                # Stream non-thinking content
                if len(buffer) > 0 and not in_think:
                    chunk = {
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": MODEL_NAME,
                        "choices": [{"index": 0, "delta": {"content": ch}, "finish_reason": None}]
                    }
                    try:
                        self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                        self.wfile.flush()
                    except BrokenPipeError:
                        pass

            result = engine.generate(user_text, max_tokens, on_char)

            tool_calls = parse_tool_calls(result["raw"]) if tools else []
            final = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": MODEL_NAME,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls" if tool_calls else "stop"}]
            }
            try:
                self.wfile.write(f"data: {json.dumps(final)}\n\n".encode())
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except BrokenPipeError:
                pass
        else:
            result = engine.generate(user_text, max_tokens)
            tool_calls = parse_tool_calls(result["raw"]) if tools else []

            response = {
                "id": request_id,
                "object": "chat.completion",
                "created": created,
                "model": MODEL_NAME,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": result["text"] if not tool_calls else None,
                    },
                    "finish_reason": "tool_calls" if tool_calls else "stop",
                }],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            }
            if tool_calls:
                response["choices"][0]["message"]["tool_calls"] = tool_calls

            self.send_json(response)

    def send_json(self, data, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()


def main():
    global engine

    parser = argparse.ArgumentParser(description="OpenAI-compatible API server for Flash-MoE")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--2bit", dest="use_2bit", action="store_true")
    parser.add_argument("--cache-entries", type=int, default=0)
    args = parser.parse_args()

    if not os.path.exists(CHAT_BIN):
        print(f"Error: {CHAT_BIN} not found. Run 'make chat' first.", file=sys.stderr)
        sys.exit(1)

    print(f"Flash-MoE API Server (persistent engine)")
    print(f"  Model:    {MODEL_NAME}")
    print(f"  Quant:    {'2-bit' if args.use_2bit else '4-bit (auto-detect)'}")
    print()

    # Start persistent inference engine
    engine = InferenceEngine(use_2bit=args.use_2bit, cache_entries=args.cache_entries)

    print()
    print(f"  Endpoint: http://{args.host}:{args.port}/v1/chat/completions")
    print(f"  Health:   http://{args.host}:{args.port}/health")
    print()
    print(f"  export OPENAI_BASE_URL=http://localhost:{args.port}/v1")
    print(f"  export OPENAI_API_KEY=local")
    print()

    def cleanup(sig, frame):
        print("\nShutting down...")
        if engine and engine.proc:
            engine.proc.terminate()
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    server = HTTPServer((args.host, args.port), APIHandler)
    server.serve_forever()


if __name__ == "__main__":
    main()
