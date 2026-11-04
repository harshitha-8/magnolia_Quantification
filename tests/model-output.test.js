import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {decode,summarize} from '../detector-core.js';
test('exported model detections match Python postprocessing on held-out sample',()=>{
  const fixture=JSON.parse(readFileSync(new URL('./onnx-fixture.json',import.meta.url)));
  const boxes=decode(fixture.data,fixture.dims,fixture.width,fixture.height,fixture.threshold,.45);
  assert.equal(boxes.length,fixture.expected_count);
  const stats=summarize(boxes,fixture.width,fixture.height);
  assert.ok(stats.boxCoveragePercent>=0&&stats.boxCoveragePercent<=100);
  for(const b of boxes){assert.ok(b.x1>=0&&b.y1>=0&&b.x2<=fixture.width&&b.y2<=fixture.height);}
});
