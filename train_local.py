import torch
from pathlib import Path
from ultralytics import YOLO

def train_local():
    print("=" * 50)
    print("SISTEM DETEKSI LUBANG JALAN - LOCAL TRAINING")
    print("=" * 50)

    cuda_available = torch.cuda.is_available()
    print(f"CUDA tersedia : {cuda_available}")
    if cuda_available:
        print(f"GPU           : {torch.cuda.get_device_name(0)}")
        print(f"VRAM          : {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        device = 0
    else:
        print("GPU tidak ditemukan. Menggunakan CPU untuk training (akan berjalan lambat).")
        device = "cpu"

    project_dir = Path(__file__).resolve().parent
    yaml_path = project_dir / "data" / "data.yaml"
    weights_path = project_dir / "yolov8n.pt"

    if not yaml_path.exists():
        raise FileNotFoundError(f"File konfigurasi {yaml_path} tidak ditemukan!")
    if not weights_path.exists():
        print(f"Base model yolov8n.pt tidak ditemukan di {weights_path}, mengunduh otomatis dari Ultralytics...")

    print(f"Menggunakan data config : {yaml_path}")
    print(f"Menggunakan base weights: {weights_path}")

    print("\nMemuat model...")
    model = YOLO(str(weights_path))

    EPOCHS = 30
    IMGSZ = 640
    BATCH_SIZE = 8
    OPTIMIZER = "AdamW"
    PATIENCE = 15
    LR0 = 0.001
    WORKERS = 4
    PROJECT_NAME = "pothole_detection"
    RUN_NAME = "yolov8n_ep30_tuned"

    print("\nParameter Training:")
    print(f"  - Epochs       : {EPOCHS}")
    print(f"  - Image Size   : {IMGSZ}")
    print(f"  - Batch Size   : {BATCH_SIZE}")
    print(f"  - Learning Rate: {LR0}")
    print(f"  - Device       : {device}")
    print(f"  - Project      : {PROJECT_NAME}/{RUN_NAME}")
    print("=" * 50)

    print("\nMemulai training model...")
    model.train(
        data=str(yaml_path),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH_SIZE,
        patience=PATIENCE,
        lr0=LR0,
        optimizer=OPTIMIZER,
        workers=WORKERS,
        device=device,
        project=PROJECT_NAME,
        name=RUN_NAME,
        exist_ok=True,
        pretrained=True,
        augment=True,
        cache=True,
        plots=True,
        save=True,
        verbose=True,
    )

    output_dir = project_dir / "runs" / "detect" / PROJECT_NAME / RUN_NAME
    print("\nTraining selesai secara lokal!")
    print(f"Hasil disimpan di: {output_dir}")

if __name__ == "__main__":
    train_local()