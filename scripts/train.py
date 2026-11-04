"""Train a compact single-class detector, then export a browser-compatible ONNX."""
import argparse
import shutil
from pathlib import Path
from ultralytics import YOLO


if __name__ == '__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',default='data/magnolia/magnolia.yaml')
    ap.add_argument('--epochs',type=int,default=30)
    ap.add_argument('--device',default='cpu')
    ap.add_argument('--batch',type=int,default=8)
    ap.add_argument('--weights',default='yolo11n.pt')
    ap.add_argument('--project',default='runs')
    args=ap.parse_args()
    model=YOLO(args.weights)
    model.train(data=args.data,epochs=args.epochs,imgsz=640,batch=args.batch,
                device=args.device,workers=0,project=args.project,name='magnolia',
                seed=42,deterministic=True,patience=12,cache=False,amp=False,
                mosaic=0.5,close_mosaic=5,fliplr=0.5,flipud=0.5,degrees=15,
                translate=0.1,scale=0.3,plots=True,exist_ok=False)
    best=Path(model.trainer.best)
    Path('models').mkdir(exist_ok=True)
    shutil.copy2(best,'models/magnolia_detector.pt')
    trained=YOLO(str(best))
    exported=trained.export(format='onnx',imgsz=640,batch=1,dynamic=False,
                           opset=17,simplify=True,nms=False,device='cpu')
    shutil.copy2(exported,'models/magnolia_detector.onnx')
    shutil.copy2(best.parent.parent/'results.csv','reports/training_history.csv')
    for name in ['results.png','confusion_matrix.png','BoxPR_curve.png']:
        p=best.parent.parent/name
        if p.exists():shutil.copy2(p,Path('reports')/name)
