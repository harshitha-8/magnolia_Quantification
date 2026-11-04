"""Evaluate untouched test frames, count errors and exported ONNX parity."""
import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from PIL import Image
from ultralytics import YOLO


if __name__ == '__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--data',default='data/magnolia/magnolia.yaml');ap.add_argument('--device',default='cpu');ap.add_argument('--confidence',type=float,default=.15)
    args=ap.parse_args()
    import yaml
    config=yaml.safe_load(Path(args.data).read_text());root=Path(config['path'])
    model=YOLO('models/magnolia_detector.pt')
    metrics=model.val(data=args.data,split='test',imgsz=640,device=args.device,workers=0,batch=8,
                      project='runs/evaluation',name='test',plots=True)
    files=sorted((root/'images'/'test').glob('*.jpg'))
    rows=[]
    for result in model.predict(source=[str(p) for p in files],imgsz=640,conf=args.confidence,iou=.45,
                                device=args.device,rect=False,verbose=False,stream=True):
        p=Path(result.path)
        truth=len([l for l in (root/'labels'/'test'/p.with_suffix('.txt').name).read_text().splitlines() if l.strip()])
        predicted=len(result.boxes)
        rows.append(dict(image=p.name,annotated_count=truth,predicted_count=predicted,
                         error=predicted-truth,absolute_error=abs(predicted-truth)))
    Path('reports').mkdir(exist_ok=True)
    with open('reports/test_counts.csv','w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0].keys(),lineterminator="\n");writer.writeheader();writer.writerows(rows)
    example=files[0]
    shutil.copy2(example,'examples/magnolia-test.jpg')
    shutil.copy2(root/'labels'/'test'/example.with_suffix('.txt').name,'examples/magnolia-test.txt')
    prediction=model.predict(str(example),imgsz=640,conf=args.confidence,iou=.45,device='cpu',rect=False,verbose=False)[0]
    prediction.save(filename='reports/example-prediction.jpg')
    im=Image.open(example).convert('RGB');w,h=im.size;r=min(640/w,640/h)
    rw,rh=round(w*r),round(h*r);left,top=round((640-rw)/2-.1),round((640-rh)/2-.1)
    # OpenCV interpolation matches Ultralytics preprocessing for numerical parity.
    import cv2
    pixels=np.full((640,640,3),114,dtype=np.uint8)
    pixels[top:top+rh,left:left+rw]=cv2.resize(np.asarray(im),(rw,rh),interpolation=cv2.INTER_LINEAR)
    tensor=np.ascontiguousarray(pixels.transpose(2,0,1)[None],dtype=np.float32)/255
    exported=Path('models/magnolia_detector.onnx');onnx.checker.check_model(str(exported))
    session=ort.InferenceSession(str(exported),providers=['CPUExecutionProvider'])
    output=session.run(None,{session.get_inputs()[0].name:tensor})[0]
    import torch
    pytorch=YOLO('models/magnolia_detector.pt').model.eval()
    with torch.no_grad():reference=pytorch(torch.from_numpy(tensor))[0].numpy()
    max_error=float(np.max(np.abs(reference-output)))
    if output.shape!=(1,5,8400):raise AssertionError(f'Unexpected ONNX shape {output.shape}')
    np.testing.assert_allclose(output,reference,atol=.003,rtol=.001)
    # Small fixture exercises actual model output through the browser geometry code.
    sparse=output[0][:,output[0,4]>=.1]
    Path('tests/onnx-fixture.json').write_text(json.dumps(dict(width=w,height=h,dims=[1,5,sparse.shape[1]],
        data=sparse.reshape(-1).tolist(),expected_count=len(prediction.boxes),threshold=args.confidence)))
    report=dict(test_images=len(rows),confidence_threshold=args.confidence,nms_iou=.45,
                precision=float(metrics.box.mp),recall=float(metrics.box.mr),map50=float(metrics.box.map50),map50_95=float(metrics.box.map),
                count_mae=float(np.mean([r['absolute_error'] for r in rows])),
                count_bias=float(np.mean([r['error'] for r in rows])),
                total_annotated=sum(r['annotated_count'] for r in rows),total_predicted=sum(r['predicted_count'] for r in rows),
                onnx_max_absolute_error=max_error,onnx_input_shape=[1,3,640,640],onnx_output_shape=list(output.shape),
                onnx_sha256=hashlib.sha256(exported.read_bytes()).hexdigest(),
                example_source=example.name,example_annotated_count=rows[0]['annotated_count'],example_predicted_count=rows[0]['predicted_count'],
                caveats=['Same-nursery sequential-frame split, not independent-field validation.',
                         'All source JPEGs truncated; evaluated only on recovered top crops.',
                         f'Precision/recall are Ultralytics validation metrics; count MAE uses fixed {args.confidence} confidence.',
                         'Bounding boxes provide counts and box coverage, not leaf/canopy segmentation or physical area.'])
    Path('reports/evaluation.json').write_text(json.dumps(report,indent=2)+'\n')
    manifest=dict(name='Magnolia plant detector',architecture='YOLO11n',classes=['Magnolia'],
                  input=dict(name=session.get_inputs()[0].name,shape=[1,3,640,640],dtype='float32',
                             preprocessing='RGB, letterbox to 640x640 with pad=114, divide by 255, NCHW'),
                  output=dict(name=session.get_outputs()[0].name,shape=list(output.shape),
                              format='cx,cy,width,height,class_probability; pixel coordinates in letterboxed input; no objectness channel'),
                  postprocessing=dict(confidence=args.confidence,nms_iou=.45,coverage='union of boxes / image area'),
                  opset=17,sha256=report['onnx_sha256'],training_epochs=30,seed=42,
                  license='AGPL-3.0; see THIRD_PARTY_NOTICES.md',limitations=report['caveats'])
    Path('models/model.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(report,indent=2))
