import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {referenceBoxes,summarize} from '../detector-core.js';
const index=JSON.parse(readFileSync(new URL('../references/annotations.json',import.meta.url)));
test('all supplied dataset references contain valid boxes and exact hash matches',()=>{
  assert.equal(index.images.length,298);
  assert.equal(Object.keys(index.hashes).length,576);
  for(const [hash,match] of Object.entries(index.hashes)){
    assert.match(hash,/^[a-f0-9]{64}$/);assert.ok(index.images[match.id]);
    assert.ok(['original','crop'].includes(match.kind));
  }
  for(const image of index.images){
    assert.ok(image.cropHeight<image.originalHeight);
    for(const [x1,y1,x2,y2] of image.boxes){
      assert.ok(x1>=0&&y1>=0&&x2<=1&&y2<=1&&x2>x1&&y2>y1);
    }
  }
});
test('reference view displays every box without fabricating model confidence',()=>{
  const sample=index.images.find(r=>r.filename==='IMG_1751.jpg');
  const boxes=referenceBoxes(sample.boxes,1280,442);
  assert.equal(boxes.length,24);
  assert.ok(boxes.every(b=>b.score===null));
  assert.equal(summarize(boxes,1280,442).meanConfidence,null);
});
