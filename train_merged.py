"""
Steel-Rod + Wood-Plank 통합 YOLOv8 학습 스크립트
"""

from pathlib import Path
from ultralytics import YOLO

DATA_YAML = Path(r"C:\Users\User\Desktop\건축AI 모듈2 7조\merged_dataset\data.yaml")

model = YOLO("yolov8n.pt")   # nano 모델 (GPU 없어도 빠름, 성능 필요 시 yolov8s.pt 사용)

results = model.train(
    data=str(DATA_YAML),
    epochs=50,
    imgsz=640,
    batch=16,
    name="steel_wood_merged",
    project=str(Path(r"C:\Users\User\Desktop\건축AI 모듈2 7조\runs\detect")),
    patience=10,         # early stopping
    exist_ok=True,
    workers=4,
)

print("\n=== 학습 완료 ===")
print(f"모델 저장 경로: {results.save_dir}")

# 검증
metrics = model.val()
print(f"\nmAP50:     {metrics.box.map50:.4f}")
print(f"mAP50-95:  {metrics.box.map:.4f}")
