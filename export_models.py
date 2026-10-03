from pathlib import Path
import shutil

from ultralytics import YOLO


IMGSZ = 320


def export(weights: str, out_name: str):
    model = YOLO(weights)

    # Export classic raw YOLO output.
    model.export(
        format="tflite",
        imgsz=IMGSZ,
        end2end=False,
    )

    stem = Path(weights).stem

    # Some Ultralytics versions create the model here:
    # <model>_saved_model/*float32.tflite
    matches = list(
        Path(f"{stem}_saved_model").rglob("*float32.tflite")
    )

    # Other versions create:
    # <model>.tflite
    if not matches:
        direct = Path(f"{stem}.tflite")
        if direct.exists():
            matches = [direct]

    if not matches:
        raise FileNotFoundError(
            f"No TFLite model found for {stem}"
        )

    dst = Path("assets") / out_name
    dst.parent.mkdir(exist_ok=True)

    shutil.copy(matches[0], dst)

    print(f"Copied {matches[0]} -> {dst}")


def main():
    export(
        "yolo26n-pose.pt",
        "yolo26n-pose.tflite",
    )

    export(
        "yolo26n.pt",
        "football.tflite",
    )


if __name__ == "__main__":
    main()