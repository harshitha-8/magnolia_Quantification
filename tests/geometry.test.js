import test from 'node:test';
import assert from 'node:assert/strict';
import {decode,letterbox,iou,unionArea,summarize} from '../detector-core.js';
test('letterbox reverses portrait and landscape coordinates',()=>{
  for(const [w,h] of [[1280,640],[640,1280],[1000,753]]){
    const l=letterbox(w,h),x=w/2,y=h/2,bw=w/4,bh=h/4;
    const raw=new Float32Array([x*l.scale+l.left,y*l.scale+l.top,bw*l.scale,bh*l.scale,.9]);
    const [box]=decode(raw,[1,5,1],w,h);
    assert.ok(Math.abs(box.x1-(x-bw/2))<.001);assert.ok(Math.abs(box.y2-(y+bh/2))<.001);
  }
});
test('NMS suppresses repeated boxes and keeps adjacent plants',()=>{
  const boxes=decode(new Float32Array([100,101,300,100,101,300,50,50,50,50,50,50,.9,.8,.7]),[1,5,3],640,640);
  assert.equal(boxes.length,2);assert.ok(boxes[0].score>boxes[1].score);
});
test('union coverage does not double count overlaps',()=>{
  const a={x1:0,y1:0,x2:10,y2:10},b={x1:5,y1:0,x2:15,y2:10};
  assert.equal(unionArea([a,b]),150);assert.equal(iou(a,b),1/3);
  assert.equal(summarize([{...a,score:.8},{...b,score:.6}],20,10).boxCoveragePercent,75);
});
test('empty predictions, thresholding and malformed outputs',()=>{
  assert.deepEqual(summarize([],640,640),{count:0,boxCoveragePercent:0,meanConfidence:null});
  assert.equal(decode(new Float32Array([50,50,20,20,.2]),[1,5,1],640,640,.5).length,0);
  assert.throws(()=>decode([], [1,84,8400],640,640),/Unexpected/);
});
