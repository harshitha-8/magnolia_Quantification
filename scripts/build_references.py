from pathlib import Path
import hashlib,json,csv
import argparse
ap=argparse.ArgumentParser(description='Bundle provided boxes for explicitly labeled reference views, not predictions.')
ap.add_argument('--source',required=True,type=Path)
ap.add_argument('--root',default=Path(__file__).resolve().parents[1],type=Path)
args=ap.parse_args();root=args.root.resolve();source=args.source
images=[];hashes={}
for row in csv.DictReader((root/'reports/dataset_manifest.csv').open()):
 if hashlib.sha256((source/row['filename']).read_bytes()).hexdigest()!=row['sha256']:
  raise ValueError(f"Source image changed since audit: {row['filename']}; rerun prepare_dataset.py first")
 w,h,ch=int(row['original_width']),int(row['original_height']),int(row['recovered_height'])
 boxes=[]
 for line in (source/Path(row['filename']).with_suffix('.txt')).read_text().splitlines():
  if not line.strip():continue
  _,x,y,bw,bh=map(float,line.split())
  if bw<=0 or bh<=0:continue
  x1,x2=max(0,(x-bw/2)*w),min(w,(x+bw/2)*w)
  y1,y2=max(0,(y-bh/2)*h),min(ch,(y+bh/2)*h)
  if x2-x1>=4 and y2-y1>=4:boxes.append([round(x1/w,8),round(y1/ch,8),round(x2/w,8),round(y2/ch,8)])
 assert len(boxes)==int(row['visible_boxes'])
 ident=len(images);images.append({'filename':row['filename'],'originalWidth':w,'originalHeight':h,'cropHeight':ch,'boxes':boxes})
 hashes[row['sha256']]={'id':ident,'kind':'original'}
 p=root/'data/magnolia/images'/row['split']/row['filename']
 if p.exists():hashes[hashlib.sha256(p.read_bytes()).hexdigest()]={'id':ident,'kind':'crop'}
(root/'references').mkdir(exist_ok=True)
(root/'references/annotations.json').write_text(json.dumps({'version':1,'description':'Provided YOLO annotations clipped to recoverable top image regions. Reference boxes are not predictions.','images':images,'hashes':hashes},separators=(',',':'))+'\n')
print('Reference images:',len(images),'recognized hashes:',len(hashes))
