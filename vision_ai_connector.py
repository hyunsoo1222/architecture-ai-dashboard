"""
Vision AI → DB 연동 모듈
YOLOv8 추론 결과를 Vision AI DB 형식으로 변환
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# ============================================================
# bbox 수량 환산 계수 (프로젝트 기준값)
# ============================================================
BBOX_TO_QTY = {
    "rebar":     {"material_name": "철근",   "unit": "kg",  "qty_per_bbox": 50},
    "remicon":   {"material_name": "레미콘", "unit": "m³",  "qty_per_bbox": 6},
    "steel":     {"material_name": "철골",   "unit": "kg",  "qty_per_bbox": 200},
    "bolt":      {"material_name": "볼트",   "unit": "EA",  "qty_per_bbox": 10},
}

# ============================================================
# YOLOv8 결과 파싱
# ============================================================

def parse_yolo_results(results, image_path: str, location: str) -> dict:
    """
    YOLOv8 model.predict() 결과 객체를 받아 Vision AI DB 한 행으로 변환.

    Parameters
    ----------
    results   : YOLOv8 Results 객체 (model.predict() 반환값의 첫 번째 요소)
    image_path: 이미지 파일 경로 (날짜 추출에 사용)
    location  : 촬영 위치 (예: "B1-기초")
    """
    # 파일명에서 날짜 파싱 시도 (예: 20240305_B1_001.jpg)
    stem = Path(image_path).stem
    try:
        capture_date = datetime.strptime(stem[:8], "%Y%m%d").strftime("%Y-%m-%d")
    except ValueError:
        capture_date = datetime.today().strftime("%Y-%m-%d")

    boxes = results.boxes
    class_names = results.names  # {0: 'rebar', 1: 'remicon', ...}

    rows = []
    for class_id, class_key in class_names.items():
        if class_key not in BBOX_TO_QTY:
            continue

        mask = boxes.cls.cpu().numpy() == class_id
        count = int(mask.sum())
        if count == 0:
            continue

        conf_scores = boxes.conf.cpu().numpy()[mask]
        info = BBOX_TO_QTY[class_key]

        rows.append({
            "capture_date":   capture_date,
            "location":       location,
            "material_name":  info["material_name"],
            "unit":           info["unit"],
            "bbox_count":     count,
            "confidence_avg": float(np.mean(conf_scores).round(4)),
            "image_count":    1,
            "actual_qty":     count * info["qty_per_bbox"],
        })

    return rows


def parse_yolo_txt(txt_path: str, class_map: dict,
                   image_path: str, location: str) -> list[dict]:
    """
    YOLOv8 --save-txt 결과(.txt)를 파싱.
    각 줄: class_id cx cy w h confidence

    Parameters
    ----------
    txt_path  : YOLO 결과 텍스트 파일 경로
    class_map : {class_id(int): label(str)}  예) {0: 'rebar', 1: 'remicon'}
    """
    stem = Path(image_path).stem
    try:
        capture_date = datetime.strptime(stem[:8], "%Y%m%d").strftime("%Y-%m-%d")
    except ValueError:
        capture_date = datetime.today().strftime("%Y-%m-%d")

    detections: dict[str, list] = {}
    txt_file = Path(txt_path)
    if not txt_file.exists():
        return []

    with open(txt_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            class_id = int(parts[0])
            conf = float(parts[5]) if len(parts) >= 6 else 1.0
            label = class_map.get(class_id)
            if label and label in BBOX_TO_QTY:
                detections.setdefault(label, []).append(conf)

    rows = []
    for label, confs in detections.items():
        info = BBOX_TO_QTY[label]
        rows.append({
            "capture_date":   capture_date,
            "location":       location,
            "material_name":  info["material_name"],
            "unit":           info["unit"],
            "bbox_count":     len(confs),
            "confidence_avg": float(np.mean(confs).round(4)),
            "image_count":    1,
            "actual_qty":     len(confs) * info["qty_per_bbox"],
        })
    return rows


# ============================================================
# 여러 이미지 결과를 누적 → Vision AI DB 생성
# ============================================================

class VisionDBBuilder:
    """
    현장 이미지 여러 장의 YOLO 결과를 누적하여
    날짜 × 위치 × 자재명 단위로 집계된 Vision AI DB를 반환.
    """

    def __init__(self):
        self._records: list[dict] = []

    def add_from_txt(self, txt_path: str, class_map: dict,
                     image_path: str, location: str):
        self._records.extend(
            parse_yolo_txt(txt_path, class_map, image_path, location)
        )

    def add_from_results(self, results, image_path: str, location: str):
        self._records.extend(
            parse_yolo_results(results, image_path, location)
        )

    def build(self) -> pd.DataFrame:
        if not self._records:
            return pd.DataFrame()

        df = pd.DataFrame(self._records)
        df["capture_date"] = pd.to_datetime(df["capture_date"])

        # 날짜 × 위치 × 자재명으로 집계
        agg = df.groupby(
            ["capture_date", "location", "material_name", "unit"],
            as_index=False,
        ).agg(
            bbox_count=("bbox_count", "sum"),
            confidence_avg=("confidence_avg", "mean"),
            image_count=("image_count", "sum"),
            actual_qty=("actual_qty", "sum"),
        )
        agg["confidence_avg"] = agg["confidence_avg"].round(4)
        return agg


# ============================================================
# 데모: 가상 YOLO txt 결과로 동작 확인
# ============================================================

if __name__ == "__main__":
    import tempfile, os

    CLASS_MAP = {0: "rebar", 1: "remicon", 2: "steel", 3: "bolt"}

    # 가상 YOLO txt 파일 생성 (실제 환경에서는 model.predict()로 대체)
    fake_detections = [
        "0 0.5 0.3 0.1 0.2 0.92",  # rebar
        "0 0.6 0.4 0.1 0.2 0.89",
        "0 0.2 0.7 0.1 0.2 0.94",
        "1 0.8 0.5 0.2 0.3 0.85",  # remicon
        "1 0.3 0.2 0.2 0.3 0.88",
    ]

    builder = VisionDBBuilder()

    # 이미지 3장 시뮬레이션
    test_cases = [
        ("20240305_B1_001.jpg", "B1-기초", fake_detections),
        ("20240305_B1_002.jpg", "B1-기초", fake_detections[:3]),
        ("20240403_B1_001.jpg", "B1-골조", fake_detections),
    ]

    for img_name, loc, det_lines in test_cases:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt",
                                         delete=False) as f:
            f.write("\n".join(det_lines))
            tmp_path = f.name
        builder.add_from_txt(tmp_path, CLASS_MAP, img_name, loc)
        os.unlink(tmp_path)

    vision_df = builder.build()
    print("[ Vision AI DB - YOLOv8 결과 집계 ]")
    print(vision_df.to_string(index=False))
    vision_df.to_csv("vision_ai_db.csv", index=False, encoding="utf-8-sig")
    print("\n→ 'vision_ai_db.csv' 저장 완료")
