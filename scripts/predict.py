"""Batch Magnolia quantification with ONNX; no PyTorch required for inference."""
import argparse
import csv
import json
from pathlib import Path
import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw


def union_area(boxes):
    xs=sorted(set(v for b in boxes for v in (b[0],b[2])))
    area=0.
    for left,right in zip(xs,xs[1:]):
        intervals=sorted((b[1],b[3]) for b in boxes if b[0]<right and b[2]>left)
        total=start=end=0.
        for lo,hi in intervals:
            if lo>end:total+=end-start;start,end=lo,hi
            else:end=max(end,hi)
        area+=(right-left)*(total+end-start)
    return area


def detect(session,image,confidence=.15,iou=.45):
    image=image.convert('RGB');w,h=image.size;r=min(640/w,640/h)
    rw,rh=round(w*r),round(h*r);left,top=round((640-rw)/2-.1),round((640-rh)/2-.1)
    pixels=np.full((640,640,3),114,dtype=np.uint8)
    pixels[top:top+rh,left:left+rw]=cv2.resize(np.asarray(image),(rw,rh),interpolation=cv2.INTER_LINEAR)
    tensor=np.ascontiguousarray(pixels.transpose(2,0,1)[None],dtype=np.float32)/255
    output=session.run(None,{session.get_inputs()[0].name:tensor})[0]
    if output.shape!=(1,5,8400):raise ValueError(f'Unexpected model output {output.shape}')
    candidates=output[0].T;candidates=candidates[candidates[:,4]>=confidence]
    boxes=[]
    for cx,cy,bw,bh,score in candidates:
        x1,x2=max(0,float((cx-bw/2-left)/r)),min(w,float((cx+bw/2-left)/r))
        y1,y2=max(0,float((cy-bh/2-top)/r)),min(h,float((cy+bh/2-top)/r))
        if x2>x1 and y2>y1:boxes.append([x1,y1,x2,y2,float(score)])
    kept=[]
    for b in sorted(boxes,key=lambda b:-b[4]):
        duplicate=False
        for a in kept:
            intersection=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
            union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection
            if intersection/union>iou:duplicate=True;break
        if not duplicate:kept.append(b)
    return kept


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('source',type=Path);ap.add_argument('--model',default='models/magnolia_detector.onnx');ap.add_argument('--output',type=Path,default=Path('predictions'));ap.add_argument('--confidence',type=float,default=.15)
    args=ap.parse_args()
    if not 0<=args.confidence<=1:ap.error('--confidence must be between 0 and 1')
    files=[args.source] if args.source.is_file() else sorted(p for p in args.source.iterdir() if p.suffix.lower() in ['.jpg','.jpeg','.png','.webp'] and not p.name.startswith('.') and '_annotated' not in p.stem)
    if not files:ap.error('No supported images found')
    args.output.mkdir(parents=True,exist_ok=True)
    session=ort.InferenceSession(args.model,providers=['CPUExecutionProvider']);reports=[]
    for p in files:
        try:
            image=Image.open(p).convert('RGB')
            boxes=detect(session,image,args.confidence)
            record=dict(image=p.name,width=image.width,height=image.height,count=len(boxes),
                        box_coverage_percent=100*union_area(boxes)/(image.width*image.height),
                        confidence_threshold=args.confidence,nms_iou=.45,
                        detections=[dict(id=i+1,label='Magnolia',x1=b[0],y1=b[1],x2=b[2],y2=b[3],confidence=b[4]) for i,b in enumerate(boxes)])
            (args.output/(p.stem+'.json')).write_text(json.dumps(record,indent=2))
            draw=ImageDraw.Draw(image)
            for i,b in enumerate(boxes):
                draw.rectangle(b[:4],outline='#bcff72',width=3);draw.text((b[0]+2,b[1]+2),str(i+1),fill='white',stroke_width=2,stroke_fill='#143e2c')
            image.save(args.output/(p.stem+'-detections.jpg'))
            reports.append({k:record[k] for k in ['image','count','box_coverage_percent','confidence_threshold']})
            print(p.name,len(boxes),'plants')
        except OSError as e:
            raise SystemExit(f'Cannot fully decode {p}: {e}. For the supplied truncated dataset, first run prepare_dataset.py and predict on the recovered crops.')
    with (args.output/'summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=reports[0].keys(),lineterminator="\n");writer.writeheader();writer.writerows(reports)
