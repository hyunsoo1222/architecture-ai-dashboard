from ultralytics import YOLO
import os

def main():
    # ===================== 경로 설정 =====================
    DATA_YAML  = r"C:/Users/User/Desktop/건축AI 모듈2 7조/Steel.yolov8/data.yaml"
    SAVE_DIR   = r"C:/Users/User/Desktop/건축AI 모듈2 7조/runs"
    MODEL_NAME = "steel_v1"

    # ===================== 모델 로드 =====================
    model = YOLO("yolov8n.pt")

    # ===================== 학습 (데이터 증강 포함) =====================
    results = model.train(
        data      = DATA_YAML,
        epochs    = 20,
        imgsz     = 640,
        batch     = 8,
        workers   = 4,
        device    = 0,
        project   = SAVE_DIR,
        name      = MODEL_NAME,
        patience  = 10,
        save      = True,
        plots     = True,

        # ===================== 데이터 증강 파라미터 =====================
        fliplr      = 0.5,
        flipud      = 0.1,
        scale       = 0.5,
        translate   = 0.1,
        degrees     = 10.0,
        shear       = 2.0,
        perspective = 0.0005,

        hsv_h = 0.015,
        hsv_s = 0.7,
        hsv_v = 0.4,

        mosaic     = 1.0,
        mixup      = 0.3,
        copy_paste = 0.5,

        erasing = 0.4,
    )

    print("✅ 학습 완료!")

    # ===================== ONNX 변환 =====================
    best_pt_path = os.path.join(SAVE_DIR, MODEL_NAME, "weights", "best.pt")

    print(f"\n🔄 ONNX 변환 시작: {best_pt_path}")

    best_model = YOLO(best_pt_path)
    export_path = best_model.export(
        format  = "onnx",
        imgsz   = 640,
        opset   = 12,        # onnxruntime-web 호환 버전
        simplify = True,     # 모델 구조 단순화 (추론 속도 향상)
        dynamic  = False,    # 고정 입력 크기 (웹 환경 권장)
    )

    print(f"✅ ONNX 변환 완료!")
    print(f"📂 PT  저장 경로: {best_pt_path}")
    print(f"📂 ONNX 저장 경로: {export_path}")


if __name__ == '__main__':
    main()