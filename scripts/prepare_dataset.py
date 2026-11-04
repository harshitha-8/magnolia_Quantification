"""Audit YOLO boxes and salvage truncated JPEGs without training on gray fill."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageFile


def prepare(source, output, reports):
    output, reports = Path(output).resolve(), Path(reports)
    reports.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in Path(source).glob('*.jpg')
                   if not p.name.startswith('.') and '_annotated' not in p.stem)
    if not files:
        raise ValueError('No original JPEG images found')
    records, seen = [], set()
    for i, path in enumerate(files):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest in seen:
            raise ValueError(f'Duplicate image bytes: {path.name}')
        seen.add(digest)
        ImageFile.LOAD_TRUNCATED_IMAGES = False
        truncated = False
        try:
            with Image.open(path) as im:
                im.load()
        except OSError:
            truncated = True
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        with Image.open(path) as im:
            im = im.convert('RGB')
        width, height = im.size
        crop_height = height
        if truncated:
            pixels = np.asarray(im)
            # Pillow fills unavailable JPEG scanlines with exactly RGB(128,128,128).
            gray_rows = np.all(pixels == 128, axis=(1, 2))
            non_gray = np.flatnonzero(~gray_rows)
            if not len(non_gray):
                raise ValueError(f'No recoverable image content: {path}')
            # Remove the partially decoded scanline/MCU at the transition too.
            crop_height = max(0, (int(non_gray[-1]) + 1) // 16 * 16 - 32)
            if crop_height < 128:
                raise ValueError(f'Too little recoverable content: {path}')
        label_path = path.with_suffix('.txt')
        if not label_path.exists():
            raise ValueError(f'Missing annotations: {label_path}')
        labels = []
        original_count = 0
        degenerate_count = 0
        for line in label_path.read_text().splitlines():
            if not line.strip():
                continue
            row = [float(v) for v in line.split()]
            if len(row) != 5 or row[0] != 0 or not all(np.isfinite(row)):
                raise ValueError(f'Invalid single-class YOLO box in {label_path}: {line}')
            _, x, y, w, h = row
            if not all(0 <= v <= 1 for v in (x,y,w,h)):
                raise ValueError(f'Invalid normalized coordinates: {line}')
            original_count += 1
            if w == 0 or h == 0:
                degenerate_count += 1
                continue
            x1, x2 = max(0, (x-w/2)*width), min(width, (x+w/2)*width)
            y1, y2 = max(0, (y-h/2)*height), min(crop_height, (y+h/2)*height)
            # Retain any non-negligible visible fragment and clip to the crop.
            # Dropping partial plants would teach the detector to ignore them.
            if x2-x1 >= 4 and y2-y1 >= 4:
                labels.append(f'0 {(x1+x2)/2/width:.8f} {(y1+y2)/2/crop_height:.8f} {(x2-x1)/width:.8f} {(y2-y1)/crop_height:.8f}')
        # Contiguous acquisition blocks, with ten-image exclusion gaps.
        # This is stronger than random adjacent frames but NOT a new-field test.
        a, b, gap = int(len(files)*.70), int(len(files)*.85), min(10,len(files)//20)
        split = 'train' if i < a else 'excluded' if i < a+gap else 'val' if i < b else 'excluded' if i < b+gap else 'test'
        if split != 'excluded':
            for kind in ('images','labels'):
                (output/kind/split).mkdir(parents=True, exist_ok=True)
            crop = im.crop((0,0,width,crop_height))
            crop.thumbnail((1280,1280))
            crop.save(output/'images'/split/path.name, quality=95)
            (output/'labels'/split/(path.stem+'.txt')).write_text('\n'.join(labels)+'\n')
        records.append(dict(filename=path.name, split=split, sha256=digest, truncated=truncated,
                            original_width=width, original_height=height, recovered_height=crop_height,
                            original_boxes=original_count, degenerate_boxes=degenerate_count, visible_boxes=len(labels)))
    with (reports/'dataset_manifest.csv').open('w', newline='') as f:
        writer=csv.DictWriter(f,fieldnames=records[0].keys(),lineterminator="\n");writer.writeheader();writer.writerows(records)
    summary = dict(images=len(records), truncated_images=sum(r['truncated'] for r in records),
                   degenerate_boxes_removed=sum(r['degenerate_boxes'] for r in records),
                   splits={s:sum(r['split']==s for r in records) for s in ['train','val','test','excluded']},
                   visible_boxes={s:sum(r['visible_boxes'] for r in records if r['split']==s) for s in ['train','val','test']},
                   crop_fraction_range=[min(r['recovered_height']/r['original_height'] for r in records),max(r['recovered_height']/r['original_height'] for r in records)],
                   limitation='Truncated inputs: only recoverable top regions used. Same-field acquisition split; no independent-site validation.')
    (reports/'dataset_audit.json').write_text(json.dumps(summary,indent=2)+'\n')
    (output/'magnolia.yaml').write_text(f'path: {output}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: Magnolia\n')
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--source',required=True);ap.add_argument('--output',default='data/magnolia');ap.add_argument('--reports',default='reports')
    args=ap.parse_args();prepare(args.source,args.output,args.reports)
