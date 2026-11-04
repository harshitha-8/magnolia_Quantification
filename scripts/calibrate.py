"""Choose a counting confidence cutoff using validation frames only."""
import argparse
import json
from pathlib import Path
import numpy as np
import yaml
from ultralytics import YOLO

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--data',default='data/magnolia/magnolia.yaml');ap.add_argument('--device',default='cpu');args=ap.parse_args()
    root=Path(yaml.safe_load(Path(args.data).read_text())['path'])
    files=sorted((root/'images'/'val').glob('*.jpg'))
    records=[]
    for result in YOLO('models/magnolia_detector.pt').predict([str(p) for p in files],conf=.1,iou=.45,imgsz=640,rect=False,device=args.device,verbose=False,stream=True):
        truth=len([l for l in (root/'labels'/'val'/Path(result.path).with_suffix('.txt').name).read_text().splitlines() if l.strip()])
        records.append((truth,result.boxes.conf.cpu().numpy()))
    choices=[]
    for t in [.1,.15,.2,.25,.3,.35,.4,.45,.5]:
        errors=[int(np.sum(scores>=t))-truth for truth,scores in records]
        choices.append(dict(threshold=t,count_mae=float(np.mean(np.abs(errors))),count_bias=float(np.mean(errors))))
    selected=min(choices,key=lambda r:(r['count_mae'],-r['threshold']))['threshold']
    report=dict(split='validation only',images=len(files),criterion='lowest count MAE; higher threshold wins ties',selected_confidence=selected,candidates=choices)
    Path('reports/threshold_selection.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
