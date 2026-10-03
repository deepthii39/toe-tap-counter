from pathlib import Path
import shutil

from ultralytics import YOLO

IMGSZ = 640  # must match kInputSize in main.dart


def main():
    model = YOLO("yolo26n.pt")  # detection model (downloads on first run)

    # end2end=False -> output shape [1, 84, 8400], which main.dart expects
    model.export(format="tflite", imgsz=IMGSZ, end2end=False)

    # Find the exported float32 file and copy it into the Flutter assets
    matches = list(Path("yolo26n_saved_model").rglob("*float32.tflite"))
    if not matches:
        raise FileNotFoundError("No *_float32.tflite found in yolo26n_saved_model")

    dst = Path("assets") / "football.tflite"
    dst.parent.mkdir(exist_ok=True)
    shutil.copy(matches[0], dst)
    print(f"Copied {matches[0]} -> {dst}")


if __name__ == "__main__":  # required on Windows
    main()