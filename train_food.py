import os
from ultralytics import YOLO

def train():
    # Load base YOLOv8 nano segmentation model
    model = YOLO("yolov8n-seg.pt")

    # Path to your verified dataset configuration
    data_path = r"C:\Users\Lenovo\Desktop\VFCT - Copy\food_dataset\fruits-.v1i.coco-segmentation\data.yaml"

    # Verify path exists before triggering trainer
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Configuration file not found at: {data_path}")

    print(f"Starting transfer learning with dataset: {data_path}")

    # Run fine-tuning on CPU (Intel Core Ultra 5)
    model.train(
        data=data_path,
        epochs=20,
        imgsz=640,
        batch=4,
        device="cpu",
        workers=0,       # Prevents Windows multiprocessing socket hangs
        project="food_training_runs",
        name="fruits_seg",
        save=True
    )

    print("\nTraining completed successfully!")
    print("Fine-tuned weights saved at: food_training_runs/fruits_seg/weights/best.pt")

if __name__ == "__main__":
    train()