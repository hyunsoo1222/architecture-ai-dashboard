from ultralytics import YOLO
import cv2
import os
from pathlib import Path

def count_steel_rods(model_path, image_dir, conf=0.5):
    """학습된 모델로 철근 개수 산출"""
    model = YOLO(model_path)
    image_dir = Path(image_dir)

    image_files = list(image_dir.glob("*.jpg")) + \
                  list(image_dir.glob("*.jpeg")) + \
                  list(image_dir.glob("*.png"))

    print(f"\n{'='*50}")
    print(f"🔍 철근 개수 산출 결과")
    print(f"{'='*50}")

    total = 0
    for img_path in image_files:
        results = model(str(img_path), conf=conf, verbose=False)
        count = len(results[0].boxes)
        total += count

        annotated = results[0].plot()
        out_path = image_dir / f"result_{img_path.name}"
        cv2.imwrite(str(out_path), annotated)

        print(f"  📸 {img_path.name:<30} → 철근 {count:>4}개")

    print(f"{'='*50}")
    print(f"  📊 총 이미지 수  : {len(image_files)}장")
    print(f"  🔩 총 철근 개수  : {total}개")
    print(f"  📈 평균 철근 수  : {total/len(image_files):.1f}개/장")
    print(f"{'='*50}")
    print(f"✅ 결과 이미지 저장 완료: {image_dir}")


def main():
    # ===================== 경로 설정 =====================
    BASE_DIR   = r"C:/Users/User/Desktop/건축AI 모듈2 7조"
    DATA_YAML  = r"C:/Users/User/Desktop/건축AI 모듈2 7조/Steel.yolov8/data.yaml"
    IMAGE_DIR  = r"C:/Users/User/Desktop/건축AI 모듈2 7조/Steel.yolov8/train/images"
    SAVE_DIR   = r"C:/Users/User/Desktop/건축AI 모듈2 7조/runs"  # ← 저장 경로 명확히 지정
    MODEL_NAME = "steel_v1"

    # ===================== 모델 로드 =====================
    model = YOLO("yolov8n.pt")

    # ===================== 학습 =====================
    results = model.train(
        data      = DATA_YAML,
        epochs    = 50,
        imgsz     = 640,
        batch     = 8,
        workers   = 4,
        device    = 0,
        project   = SAVE_DIR,   # ← 저장 폴더
        name      = MODEL_NAME, # ← 실험 이름
        patience  = 10,
        save      = True,
        plots     = True,
    )

    print("✅ 학습 완료!")

    # ===================== 철근 개수 산출 =====================
    best_model = f"{SAVE_DIR}/{MODEL_NAME}/weights/best.pt"
    print(f"📂 모델 경로: {best_model}")

    if os.path.exists(best_model):
        count_steel_rods(
            model_path = best_model,
            image_dir  = IMAGE_DIR,
            conf       = 0.5,
        )
    else:
        print(f"⚠️ 모델 파일을 찾을 수 없어요: {best_model}")


if __name__ == '__main__':
    main()