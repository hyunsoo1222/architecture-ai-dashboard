"""
YOLOv8 추론 → Vision AI DB → EVMS 전체 파이프라인
사용법: python inference_pipeline.py --model best.pt --source 현장사진/ --location B1-기초
"""

import argparse
import pandas as pd
from pathlib import Path
from datetime import datetime
from vision_ai_connector import VisionDBBuilder

def run_pipeline(model_path: str, source: str, location: str,
                 conf_threshold: float = 0.5) -> pd.DataFrame:
    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit("ultralytics 미설치: pip install ultralytics")

    model   = YOLO(model_path)
    builder = VisionDBBuilder()
    source_path = Path(source)

    # 단일 이미지 / 폴더 모두 처리
    if source_path.is_file():
        image_paths = [source_path]
    else:
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        image_paths = [p for p in source_path.rglob("*") if p.suffix.lower() in exts]

    if not image_paths:
        raise SystemExit(f"이미지 없음: {source}")

    print(f"[추론 시작] 모델: {model_path} | 이미지: {len(image_paths)}장 | 위치: {location}")

    for img_path in sorted(image_paths):
        results = model.predict(str(img_path), conf=conf_threshold, verbose=False)[0]
        builder.add_from_results(results, str(img_path), location)

        # 이미지별 간단 로그
        names = results.names
        counts = {}
        for cls_id in results.boxes.cls.cpu().numpy().astype(int):
            label = names[cls_id]
            counts[label] = counts.get(label, 0) + 1
        print(f"  {img_path.name}: {counts}")

    vision_df = builder.build()
    print(f"\n[Vision AI DB 생성 완료] {len(vision_df)}행")
    print(vision_df.to_string(index=False))

    out_path = f"vision_ai_db_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    vision_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"→ '{out_path}' 저장 완료")
    return vision_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 현장 이미지 추론 파이프라인")
    parser.add_argument("--model",    default="best.pt",   help="학습된 가중치 경로")
    parser.add_argument("--source",   required=True,       help="이미지 파일 또는 폴더")
    parser.add_argument("--location", default="현장",      help="촬영 위치 (예: B1-기초)")
    parser.add_argument("--conf",     default=0.5, type=float, help="신뢰도 임계값")
    args = parser.parse_args()

    run_pipeline(args.model, args.source, args.location, args.conf)
