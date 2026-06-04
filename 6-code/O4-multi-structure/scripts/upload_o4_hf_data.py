"""Upload O4 multi-structure HDF5 data to a Hugging Face dataset.

The script is designed for large local files:

- token is read from `HF_TOKEN`;
- proxy environment variables are cleared by default;
- a staging folder with hardlinks is created, avoiding a second 20+ GB copy;
- only files missing on the remote are staged;
- `HfApi.upload_large_folder` is used for resumable uploads.

Example from project root:
    $env:HF_TOKEN = "<token>"
    python 6-code/O4-multi-structure/scripts/upload_o4_hf_data.py --num_workers 4
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys

from huggingface_hub import HfApi


PROJECT_ROOT = Path(__file__).resolve().parents[3]
O4_DIR = Path(__file__).resolve().parents[1]
DEFAULT_BASE_DATA = Path(r"D:\BaiduNetdiskDownload\SesimicTransformerData")
DEFAULT_STAGE = PROJECT_ROOT / ".codex" / "hf_upload_stage"
REMOTE_PREFIX = "MDOF/knet-250/Data/fno"


def clear_proxy_env() -> None:
    for key in [
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ]:
        os.environ.pop(key, None)
    os.environ["NO_PROXY"] = "*"
    os.environ["no_proxy"] = "*"


def load_structure_files() -> list[str]:
    config = json.loads((O4_DIR / "configs" / "o4_structures.json").read_text(encoding="utf-8"))
    return [meta["filename"] for meta in config.values()]


def hardlink_or_copy(src: Path, dst: Path, allow_copy: bool) -> None:
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        if not allow_copy:
            raise
        shutil.copy2(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload O4 HDF5 files to Hugging Face.")
    parser.add_argument("--repo_id", default="JasonXF/SeFNO")
    parser.add_argument("--repo_type", default="dataset")
    parser.add_argument("--base_data_dir", default=str(DEFAULT_BASE_DATA))
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--stage_dir", default=str(DEFAULT_STAGE))
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--allow_copy", action="store_true", help="Copy if hardlinks are not supported.")
    parser.add_argument("--keep_proxy", action="store_true", help="Do not clear proxy environment variables.")
    args = parser.parse_args()

    if not os.environ.get("HF_TOKEN"):
        raise RuntimeError("HF_TOKEN is required in the environment. The token is not stored by this script.")
    if not args.keep_proxy:
        clear_proxy_env()
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")

    api = HfApi()
    data_dir = Path(args.data_dir) if args.data_dir else Path(args.base_data_dir) / "MDOF" / "knet-250" / "Data"
    stage_dir = Path(args.stage_dir)
    remote_files = set(api.list_repo_files(repo_id=args.repo_id, repo_type=args.repo_type))

    target_files = load_structure_files()
    missing = [name for name in target_files if f"{REMOTE_PREFIX}/{name}" not in remote_files]
    print(f"Remote already has {len(target_files) - len(missing)}/{len(target_files)} O4 files.", flush=True)
    if not missing:
        print("All O4 HDF5 files are already present on Hugging Face.", flush=True)
        return

    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    stage_remote_dir = stage_dir / "MDOF" / "knet-250" / "Data" / "fno"
    stage_remote_dir.mkdir(parents=True, exist_ok=True)

    print("Staging missing files:", flush=True)
    for name in missing:
        src = data_dir / name
        dst = stage_remote_dir / name
        if not src.exists():
            raise FileNotFoundError(src)
        hardlink_or_copy(src, dst, allow_copy=args.allow_copy)
        print(f"  {name}: {src.stat().st_size / 1024**3:.2f} GiB", flush=True)

    print("\nStarting resumable upload_large_folder.", flush=True)
    print(f"Repo: {args.repo_id} ({args.repo_type})", flush=True)
    print(f"Stage: {stage_dir}", flush=True)
    print(f"Workers: {args.num_workers}", flush=True)
    sys.stdout.flush()

    api.upload_large_folder(
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        folder_path=stage_dir,
        num_workers=args.num_workers,
        print_report=True,
        print_report_every=30,
    )

    print("\nRemote check:", flush=True)
    final_files = set(api.list_repo_files(repo_id=args.repo_id, repo_type=args.repo_type))
    for name in target_files:
        remote = f"{REMOTE_PREFIX}/{name}"
        print(f"  {'OK' if remote in final_files else 'MISSING'} {remote}", flush=True)
    print("O4 HF data upload script completed.", flush=True)


if __name__ == "__main__":
    main()
