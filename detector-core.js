/* Pure geometry shared by the app and regression tests. YOLO11: [1,5,8400]. */
export function letterbox(width, height, size=640) {
  const scale=Math.min(size/width,size/height);
  const resizedWidth=Math.round(width*scale), resizedHeight=Math.round(height*scale);
  return {scale, resizedWidth, resizedHeight, left:Math.round((size-resizedWidth)/2-0.1), top:Math.round((size-resizedHeight)/2-0.1)};
}
export function iou(a,b) {
  const intersection=Math.max(0,Math.min(a.x2,b.x2)-Math.max(a.x1,b.x1))*Math.max(0,Math.min(a.y2,b.y2)-Math.max(a.y1,b.y1));
  const union=(a.x2-a.x1)*(a.y2-a.y1)+(b.x2-b.x1)*(b.y2-b.y1)-intersection;
  return union>0 ? intersection/union : 0;
}
export function decode(data,dims,width,height,threshold=0.35,nmsThreshold=0.45) {
  if(dims.length!==3 || dims[0]!==1 || dims[1]!==5) throw new Error(`Unexpected model output: ${dims}. Expected [1,5,N].`);
  const n=dims[2], l=letterbox(width,height), candidates=[];
  for(let i=0;i<n;i++) {
    const score=data[4*n+i];
    if(!Number.isFinite(score)||score<threshold) continue;
    const cx=data[i],cy=data[n+i],w=data[2*n+i],h=data[3*n+i];
    const box={score,x1:Math.max(0,(cx-w/2-l.left)/l.scale),y1:Math.max(0,(cy-h/2-l.top)/l.scale),
      x2:Math.min(width,(cx+w/2-l.left)/l.scale),y2:Math.min(height,(cy+h/2-l.top)/l.scale)};
    if(Object.values(box).every(Number.isFinite)&&box.x2>box.x1&&box.y2>box.y1)candidates.push(box);
  }
  candidates.sort((a,b)=>b.score-a.score);
  const kept=[];
  for(const candidate of candidates) if(kept.every(b=>iou(b,candidate)<=nmsThreshold))kept.push(candidate);
  return kept;
}
export function unionArea(boxes) {
  const xs=[...new Set(boxes.flatMap(b=>[b.x1,b.x2]))].sort((a,b)=>a-b);
  let area=0;
  for(let i=1;i<xs.length;i++) {
    const intervals=boxes.filter(b=>b.x1<xs[i]&&b.x2>xs[i-1]).map(b=>[b.y1,b.y2]).sort((a,b)=>a[0]-b[0]);
    let total=0,start=0,end=0;
    for(const [lo,hi] of intervals){if(lo>end){total+=end-start;start=lo;end=hi;}else end=Math.max(end,hi);}
    total+=end-start;area+=(xs[i]-xs[i-1])*total;
  }
  return area;
}
export function summarize(boxes,width,height) {
  return {count:boxes.length,boxCoveragePercent:100*unionArea(boxes)/(width*height),meanConfidence:boxes.length?boxes.reduce((s,b)=>s+b.score,0)/boxes.length:null};
}
