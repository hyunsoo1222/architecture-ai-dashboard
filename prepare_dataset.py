"""
Steel + Wood Plank 데이터셋 통합 및 레이블 정규화 스크립트
- Steel-Rod  → class 0
- Wood-Plank → class 1
- 폴리곤 레이블 → bbox (cx cy w h) 변환
- 80:20 train/valid 분할
"""

import os
import shutil
import random
from pathlib import Path

BASE = Path(r"C:\Users\User\Desktop\건축AI 모듈2 7조")
STEEL_DIR = BASE / "Steel.yolov8"
WOOD_DIR  = BASE / "Wood Plank.yolov8"
OUT_DIR   = BASE / "merged_dataset"


def polygon_to_bbox(coords: list[float]) -> tuple[float, float, float, float]:
    """x1 y1 x2 y2 ... → cx cy w h (normalized)"""
    xs = coords[0::2]
    ys = coords[1::2]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2
    w  = x_max - x_min
    h  = y_max - y_min
    return cx, cy, w, h


def normalize_label_file(src: Path, dst: Path, remap_class: int | None = None):
    """레이블 파일을 읽어 bbox 포맷으로 정규화 후 저장"""
    lines_out = []
    for line in src.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        cls = int(parts[0])
        if remap_class is not None:
            cls = remap_class
        vals = list(map(float, parts[1:]))
        if len(vals) == 4:
            # 이미 bbox
            cx, cy, w, h = vals
        elif len(vals) >= 6 and len(vals) % 2 == 0:
            # 폴리곤 → bbox
            cx, cy, w, h = polygon_to_bbox(vals)
        else:
            # 홀수 좌표 등 비정상 → 건너뜀
            continue
        lines_out.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    dst.write_text("\n".join(lines_out), encoding="utf-8")


def collect_pairs(img_dir: Path, lbl_dir: Path):
    """이미지-레이블 쌍 수집"""
    pairs = []
    for img in sorted(img_dir.iterdir()):
        if img.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        lbl = lbl_dir / (img.stem + ".txt")
        if lbl.exists():
            pairs.append((img, lbl))
    return pairs


def split_and_copy(pairs, class_id: int, prefix: str, out_dir: Path, ratio=0.8):
    random.shuffle(pairs)
    n_train = int(len(pairs) * ratio)
    splits = {"train": pairs[:n_train], "valid": pairs[n_train:]}
    for split, items in splits.items():
        for img, lbl in items:
            dst_img = out_dir / split / "images" / f"{prefix}_{img.name}"
            dst_lbl = out_dir / split / "labels" / f"{prefix}_{img.stem}.txt"
            shutil.copy2(img, dst_img)
            normalize_label_file(lbl, dst_lbl, remap_class=class_id)
    return len(splits["train"]), len(splits["valid"])


def main():
    random.seed(42)

    # 출력 폴더 초기화
    for split in ("train", "valid"):
        for sub in ("images", "labels"):
            (OUT_DIR / split / sub).mkdir(parents=True, exist_ok=True)

    # Steel (class 0)
    steel_pairs = collect_pairs(
        STEEL_DIR / "train" / "images",
        STEEL_DIR / "train" / "labels",
    )
    s_tr, s_val = split_and_copy(steel_pairs, class_id=0, prefix="steel", out_dir=OUT_DIR)
    print(f"[Steel]      train={s_tr}  valid={s_val}")

    # Wood Plank (class 1)
    wood_pairs = collect_pairs(
        WOOD_DIR / "train" / "images",
        WOOD_DIR / "train" / "labels",
    )
    w_tr, w_val = split_and_copy(wood_pairs, class_id=1, prefix="wood", out_dir=OUT_DIR)
    print(f"[Wood Plank] train={w_tr}  valid={w_val}")

    # data.yaml 작성
    yaml_text = f"""\
train: {OUT_DIR / 'train' / 'images'}
val:   {OUT_DIR / 'valid' / 'images'}

nc: 2
names: ['Steel-Rod', 'Wood-Plank']
"""
    (OUT_DIR / "data.yaml").write_text(yaml_text, encoding="utf-8")
    print(f"\n[완료] data.yaml → {OUT_DIR / 'data.yaml'}")
    print(f"       총 train={s_tr + w_tr}장  valid={s_val + w_val}장")


if __name__ == "__main__":
    main()
