#!/usr/bin/env python3
import shutil
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
RESULTS = ROOT / 'results'

def merge_dirs(src: Path, dst: Path):
    dst.mkdir(parents=True, exist_ok=True)
    for root, dirs, files in os.walk(src):
        rel = Path(root).relative_to(src)
        target_root = dst / rel
        target_root.mkdir(parents=True, exist_ok=True)
        for f in files:
            s = Path(root) / f
            t = target_root / f
            if t.exists():
                # overwrite
                t.unlink()
            shutil.copy2(s, t)

def main():
    to_move = [
        'amcl',
        'ngps_only',
        'rtabmap',
        'spf_lidar',
    ]
    for name in to_move:
        s = DATA / name
        d = RESULTS / name
        if not s.exists():
            print(f"{s} does not exist, skipping")
            continue
        if d.exists():
            print(f"Merging {s} -> {d}")
            merge_dirs(s, d)
            print(f"Removing source {s}")
            shutil.rmtree(s)
        else:
            print(f"Moving {s} -> {d}")
            shutil.move(str(s), str(d))

    print("Done. Remaining in data/:")
    for p in sorted(DATA.iterdir()):
        print(' -', p.name)

if __name__ == '__main__':
    main()
