#!/usr/bin/env python3
"""
Flash-MoE Setup — One command to go from zero to running a 397B parameter LLM locally.

Usage:
    ./setup.py          # interactive setup
    ./setup.py --yes    # skip confirmations
    ./setup.py --status # show what's done and what's left

This script is idempotent. Re-run it at any point and it picks up where it left off.
"""

import json
import os
import platform
import shutil
import struct
import subprocess
import sys
import time

REPO_URL = "https://github.com/cs01/flash-moe.git"
PINNED_COMMIT = "b2c1784"


def bootstrap():
    """If we're not inside the repo, shallow-clone to /tmp and re-exec from there."""
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        if os.path.exists(os.path.join(script_dir, "expert_index.json")):
            return script_dir
    except NameError:
        pass

    marker = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), "expert_index.json")
    if os.path.exists(marker):
        return os.path.dirname(os.path.abspath(sys.argv[0]))

    import tempfile
    clone_dir = os.path.join(tempfile.gettempdir(), "flash-moe-setup")

    print(f"\033[94m==>\033[0m \033[1mcloning flash-moe to {clone_dir}...\033[0m")
    if os.path.isdir(os.path.join(clone_dir, ".git")):
        subprocess.check_call(["git", "-C", clone_dir, "fetch", "origin"],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        subprocess.check_call(["git", "clone", "--depth", "1", REPO_URL, clone_dir])

    if PINNED_COMMIT != "WILL_BE_UPDATED":
        subprocess.call(["git", "-C", clone_dir, "fetch", "--depth", "1", "origin", PINNED_COMMIT],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.call(["git", "-C", clone_dir, "checkout", PINNED_COMMIT],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    setup_script = os.path.join(clone_dir, "setup.py")
    os.execv(sys.executable, [sys.executable, setup_script] + sys.argv[1:])


REPO_DIR = bootstrap()
METAL_DIR = os.path.join(REPO_DIR, "metal_infer")
VENV_DIR = os.path.join(REPO_DIR, ".venv")
VENV_PYTHON = os.path.join(VENV_DIR, "bin", "python3")

HF_REPO_ID = "mlx-community/Qwen3.5-397B-A17B-4bit"
HF_SNAPSHOT = "39159bd8aa74f5c8446d2b2dc584f62bb51cb0d3"

MODEL_WEIGHTS_BIN = os.path.join(METAL_DIR, "model_weights.bin")
MODEL_WEIGHTS_JSON = os.path.join(METAL_DIR, "model_weights.json")
TOKENIZER_BIN = os.path.join(METAL_DIR, "tokenizer.bin")
VOCAB_BIN = os.path.join(METAL_DIR, "vocab.bin")
EXPERT_INDEX = os.path.join(REPO_DIR, "expert_index.json")
INFER_BIN = os.path.join(METAL_DIR, "infer")
CHAT_BIN = os.path.join(METAL_DIR, "chat")

EXPERT_SIZE = 7077888
NUM_EXPERTS = 512
NUM_LAYERS = 60
LAYER_SIZE = NUM_EXPERTS * EXPERT_SIZE

REQUIRED_SPACE_GB = 220
TIGHT_SPACE_GB = 280

RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
BOLD = "\033[1m"
RESET = "\033[0m"


def status(msg):
    print(f"{BLUE}==>{RESET} {BOLD}{msg}{RESET}")


def success(msg):
    print(f"  {GREEN}✓{RESET} {msg}")


def warn(msg):
    print(f"  {YELLOW}!{RESET} {msg}")


def error(msg):
    print(f"  {RED}✗{RESET} {msg}")


def ask(prompt, default_yes=True, auto_yes=False):
    if auto_yes:
        return True
    suffix = "[Y/n]" if default_yes else "[y/N]"
    try:
        resp = input(f"  {prompt} {suffix} ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        sys.exit(1)
    if not resp:
        return default_yes
    return resp in ("y", "yes")


def free_space_gb(path):
    st = os.statvfs(path)
    return (st.f_bavail * st.f_frsize) / (1024 ** 3)


def hf_cache_dir():
    return os.path.expanduser("~/.cache/huggingface/hub")


def find_model_path():
    base = os.path.join(hf_cache_dir(), "models--mlx-community--Qwen3.5-397B-A17B-4bit", "snapshots")
    if not os.path.isdir(base):
        return None
    snapshots = os.listdir(base)
    if not snapshots:
        return None
    for s in snapshots:
        candidate = os.path.join(base, s)
        if os.path.exists(os.path.join(candidate, "model.safetensors.index.json")):
            return candidate
    return None


def packed_experts_dir(model_path=None):
    if model_path:
        return os.path.join(model_path, "packed_experts")
    return None


def count_packed_layers(model_path):
    d = packed_experts_dir(model_path)
    if not d or not os.path.isdir(d):
        return 0
    return sum(1 for f in os.listdir(d) if f.startswith("layer_") and f.endswith(".bin"))


def step_status():
    model_path = find_model_path()
    checks = {
        "venv": os.path.exists(VENV_PYTHON),
        "model_downloaded": model_path is not None,
        "weights_extracted": os.path.exists(MODEL_WEIGHTS_BIN) and os.path.exists(MODEL_WEIGHTS_JSON),
        "experts_repacked": count_packed_layers(model_path) == NUM_LAYERS if model_path else False,
        "built": os.path.exists(INFER_BIN),
    }
    return checks, model_path


def print_status():
    checks, model_path = step_status()
    status("Flash-MoE Setup Status")
    labels = {
        "venv": "Python venv created",
        "model_downloaded": f"Model downloaded ({HF_REPO_ID})",
        "weights_extracted": "Non-expert weights extracted",
        "experts_repacked": f"Expert weights repacked ({NUM_LAYERS} layers)",
        "built": "C/Metal engine compiled",
    }
    for key, label in labels.items():
        if checks[key]:
            success(label)
        else:
            if key == "experts_repacked" and model_path:
                n = count_packed_layers(model_path)
                warn(f"{label} — {n}/{NUM_LAYERS} done")
            else:
                error(label)

    free = free_space_gb(REPO_DIR)
    print(f"\n  Disk free: {free:.0f} GB")
    if model_path:
        print(f"  Model path: {model_path}")


def check_platform():
    if platform.system() != "Darwin":
        error("flash-moe requires macOS with Apple Silicon (Metal GPU)")
        sys.exit(1)
    machine = platform.machine()
    if machine not in ("arm64", "aarch64"):
        error(f"flash-moe requires Apple Silicon, got {machine}")
        sys.exit(1)
    success("macOS + Apple Silicon detected")


def check_disk_space(auto_yes):
    free = free_space_gb(REPO_DIR)
    model_path = find_model_path()
    already_downloaded = model_path is not None
    packed = count_packed_layers(model_path) if model_path else 0
    remaining_gb = (NUM_LAYERS - packed) * LAYER_SIZE / (1024 ** 3)

    if already_downloaded and packed > 0:
        needed = remaining_gb + 10
        if free < needed:
            warn(f"{free:.0f} GB free, need ~{needed:.0f} GB for {NUM_LAYERS - packed} remaining layers")
            if not ask("continue? (cleanup mode can help)", auto_yes=auto_yes):
                sys.exit(1)
        else:
            success(f"{free:.0f} GB free — enough for remaining work")
    elif already_downloaded:
        success(f"model already downloaded, {free:.0f} GB free")
    elif free < REQUIRED_SPACE_GB:
        error(f"need ~{REQUIRED_SPACE_GB} GB free, only {free:.0f} GB available")
        error("the model download alone is ~209 GB")
        if not ask("continue anyway?", default_yes=False, auto_yes=auto_yes):
            sys.exit(1)
    elif free < TIGHT_SPACE_GB:
        warn(f"{free:.0f} GB free — tight but workable")
        warn("the script can delete HF cache shards as it repacks to stay in budget")
    else:
        success(f"{free:.0f} GB free — plenty of room")


def setup_venv():
    if os.path.exists(VENV_PYTHON):
        success("venv already exists")
        return

    status("creating python venv")
    subprocess.check_call([sys.executable, "-m", "venv", VENV_DIR])
    subprocess.check_call([VENV_PYTHON, "-m", "pip", "install", "--quiet",
                           "huggingface_hub", "numpy"])
    success("venv created and dependencies installed")


def download_model(auto_yes):
    model_path = find_model_path()
    if model_path:
        success(f"model already downloaded at {model_path}")
        return model_path

    status(f"downloading {HF_REPO_ID} (~209 GB)")
    print("  this will take a while depending on your internet connection")
    if not ask("start download?", auto_yes=auto_yes):
        sys.exit(0)

    subprocess.check_call([VENV_PYTHON, "-c",
        f"from huggingface_hub import snapshot_download; "
        f"snapshot_download('{HF_REPO_ID}')"])

    model_path = find_model_path()
    if not model_path:
        error("download completed but model path not found — check ~/.cache/huggingface/hub")
        sys.exit(1)

    success(f"model downloaded to {model_path}")
    return model_path


def update_expert_index(model_path):
    with open(EXPERT_INDEX) as f:
        idx = json.load(f)

    if idx["model_path"] == model_path:
        success("expert_index.json already points to correct path")
        return

    status("updating expert_index.json with local model path")
    idx["model_path"] = model_path
    with open(EXPERT_INDEX, "w") as f:
        json.dump(idx, f, indent=2)
    success("expert_index.json updated")


def extract_weights(model_path):
    if os.path.exists(MODEL_WEIGHTS_BIN) and os.path.exists(MODEL_WEIGHTS_JSON):
        size_gb = os.path.getsize(MODEL_WEIGHTS_BIN) / (1024 ** 3)
        success(f"non-expert weights already extracted ({size_gb:.1f} GB)")
        return

    status("extracting non-expert weights (~5.5 GB)")
    subprocess.check_call([
        VENV_PYTHON,
        os.path.join(METAL_DIR, "extract_weights.py"),
        "--model", model_path,
        "--output", METAL_DIR,
    ])
    success("non-expert weights extracted")


def export_tokenizer(model_path):
    if os.path.exists(TOKENIZER_BIN) and os.path.exists(VOCAB_BIN):
        success("tokenizer.bin and vocab.bin already exist")
        return

    tokenizer_json = os.path.join(model_path, "tokenizer.json")
    if not os.path.exists(tokenizer_json):
        error(f"tokenizer.json not found at {tokenizer_json}")
        sys.exit(1)

    status("exporting tokenizer.bin and vocab.bin")

    subprocess.check_call([
        VENV_PYTHON,
        os.path.join(METAL_DIR, "export_tokenizer.py"),
        tokenizer_json,
        TOKENIZER_BIN,
    ])

    with open(tokenizer_json, "r", encoding="utf-8") as f:
        t = json.load(f)
    vocab = t["model"]["vocab"]
    added = {tok["content"]: tok["id"] for tok in t["added_tokens"]}
    all_tokens = {**vocab, **added}
    max_id = max(all_tokens.values())
    num_entries = max_id + 1

    with open(VOCAB_BIN, "wb") as f:
        f.write(struct.pack("<I", num_entries))
        f.write(struct.pack("<I", max_id))
        by_id = {v: k for k, v in all_tokens.items()}
        for i in range(num_entries):
            token_str = by_id.get(i, "")
            b = token_str.encode("utf-8")
            f.write(struct.pack("<H", len(b)))
            if b:
                f.write(b)

    success(f"exported tokenizer.bin + vocab.bin ({num_entries} tokens)")


def repack_experts(model_path, auto_yes):
    pe_dir = packed_experts_dir(model_path)
    done = count_packed_layers(model_path)

    if done == NUM_LAYERS:
        success(f"all {NUM_LAYERS} expert layers already repacked")
        ensure_packed_experts_symlink(pe_dir)
        return

    if done > 0:
        warn(f"{done}/{NUM_LAYERS} layers already repacked, resuming")

    remaining = NUM_LAYERS - done
    needed_gb = remaining * LAYER_SIZE / (1024 ** 3)
    free = free_space_gb(model_path)

    status(f"repacking expert weights — {remaining} layers, ~{needed_gb:.0f} GB")

    tight = free < needed_gb + 20
    cleanup_as_we_go = False
    if tight:
        warn(f"only {free:.0f} GB free but need ~{needed_gb:.0f} GB")
        warn("can repack in batches and delete HF cache shards to free space")
        if ask("delete HF safetensor shards after each layer is repacked?", auto_yes=auto_yes):
            cleanup_as_we_go = True
        elif not ask("continue without cleanup? (may run out of space)", default_yes=False, auto_yes=auto_yes):
            sys.exit(1)

    layers_to_do = [i for i in range(NUM_LAYERS)
                    if not os.path.exists(os.path.join(pe_dir or "", f"layer_{i:02d}.bin"))]

    if cleanup_as_we_go:
        repack_with_cleanup(model_path, layers_to_do, pe_dir)
    else:
        layer_spec = f"{layers_to_do[0]}-{layers_to_do[-1]}" if len(layers_to_do) > 1 else str(layers_to_do[0])
        subprocess.check_call([
            VENV_PYTHON,
            os.path.join(REPO_DIR, "repack_experts.py"),
            "--index", EXPERT_INDEX,
            "--layers", layer_spec,
        ])

    final_count = count_packed_layers(model_path)
    if final_count == NUM_LAYERS:
        success(f"all {NUM_LAYERS} layers repacked")
    else:
        error(f"only {final_count}/{NUM_LAYERS} layers repacked")
        sys.exit(1)

    ensure_packed_experts_symlink(pe_dir)


def repack_with_cleanup(model_path, layers, pe_dir):
    with open(EXPERT_INDEX) as f:
        idx = json.load(f)
    expert_reads = idx["expert_reads"]

    all_shard_files = set()
    shard_to_layers = {}
    for layer_idx in range(NUM_LAYERS):
        layer_key = str(layer_idx)
        if layer_key not in expert_reads:
            continue
        for comp_info in expert_reads[layer_key].values():
            fname = comp_info["file"]
            all_shard_files.add(fname)
            shard_to_layers.setdefault(fname, set()).add(layer_idx)

    completed_layers = set(range(NUM_LAYERS)) - set(layers)

    for i, layer_idx in enumerate(layers):
        print(f"\n  [{i + 1}/{len(layers)}] repacking layer {layer_idx}...")
        subprocess.check_call([
            VENV_PYTHON,
            os.path.join(REPO_DIR, "repack_experts.py"),
            "--index", EXPERT_INDEX,
            "--layers", str(layer_idx),
        ])
        completed_layers.add(layer_idx)

        for fname, needed_by in shard_to_layers.items():
            if needed_by.issubset(completed_layers):
                shard_path = os.path.join(model_path, fname)
                if os.path.islink(shard_path):
                    blob_path = os.path.realpath(shard_path)
                    if os.path.exists(blob_path):
                        size_mb = os.path.getsize(blob_path) / (1024 ** 2)
                        os.remove(blob_path)
                        os.remove(shard_path)
                        print(f"    deleted {fname} ({size_mb:.0f} MB) — no longer needed")
                elif os.path.exists(shard_path):
                    size_mb = os.path.getsize(shard_path) / (1024 ** 2)
                    os.remove(shard_path)
                    print(f"    deleted {fname} ({size_mb:.0f} MB) — no longer needed")


def ensure_packed_experts_symlink(pe_dir):
    link_path = os.path.join(METAL_DIR, "packed_experts")
    if os.path.islink(link_path):
        if os.readlink(link_path) == pe_dir:
            return
        os.unlink(link_path)
    elif os.path.exists(link_path):
        return

    os.symlink(pe_dir, link_path)
    success(f"symlinked metal_infer/packed_experts -> {pe_dir}")


def build():
    if os.path.exists(INFER_BIN) and os.path.exists(CHAT_BIN):
        success("C/Metal engine already built")
        return

    status("building C/Metal inference engine")
    subprocess.check_call(["make", "-C", METAL_DIR, "all"])
    subprocess.check_call(["make", "-C", METAL_DIR, "chat"])
    success("build complete")


def verify():
    status("verifying setup")
    if not os.path.exists(INFER_BIN):
        error("infer binary not found")
        return False
    if not os.path.exists(MODEL_WEIGHTS_BIN):
        error("model_weights.bin not found")
        return False

    pe_link = os.path.join(METAL_DIR, "packed_experts")
    if not os.path.exists(pe_link):
        error("packed_experts not found in metal_infer/")
        return False

    layer0 = os.path.join(pe_link, "layer_00.bin")
    if not os.path.exists(layer0):
        error("packed_experts/layer_00.bin not found")
        return False

    if not os.path.exists(os.path.join(METAL_DIR, "tokenizer.bin")):
        error("tokenizer.bin not found")
        return False
    if not os.path.exists(os.path.join(METAL_DIR, "vocab.bin")):
        error("vocab.bin not found")
        return False

    success("all files in place")
    return True


def main():
    auto_yes = "--yes" in sys.argv or "-y" in sys.argv

    if "--status" in sys.argv:
        print_status()
        sys.exit(0)

    print(f"\n{BOLD}Flash-MoE Setup{RESET}")
    print(f"run a 397B parameter LLM locally on your mac\n")

    check_platform()
    check_disk_space(auto_yes)

    setup_venv()
    model_path = download_model(auto_yes)
    update_expert_index(model_path)
    extract_weights(model_path)
    export_tokenizer(model_path)
    repack_experts(model_path, auto_yes)
    build()

    if verify():
        print(f"\n{GREEN}{BOLD}setup complete!{RESET}\n")
        print(f"  start the server:")
        print(f"    cd {METAL_DIR}")
        print(f"    ./infer --serve 8000")
        print(f"\n  then in another terminal:")
        print(f"    cd {METAL_DIR}")
        print(f"    ./chat")
        print(f"\n  OpenAI-compatible API at http://localhost:8000/v1\n")
    else:
        print(f"\n{RED}setup incomplete — check errors above{RESET}")
        sys.exit(1)


if __name__ == "__main__":
    main()
