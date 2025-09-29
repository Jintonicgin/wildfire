from ultralytics import YOLO
import argparse, os, shutil

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data", default="datasets/wildfire_yolo/wildfire.yaml")
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--patience", type=int, default=20)
    args=ap.parse_args()

    m=YOLO(args.model)
    r=m.train(data=args.data, imgsz=args.imgsz, epochs=args.epochs, batch=args.batch, patience=args.patience)
    best=r.save_dir+"/weights/best0.pt"
    os.makedirs("ML/weights", exist_ok=True)
    shutil.copy2(best, "ML/weights/best0.pt")
    print("saved:", "ML/weights/best0.pt")

if __name__=="__main__":
    main()