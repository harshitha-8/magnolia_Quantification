import {letterbox,decode,summarize,referenceBoxes} from './detector-core.js';
const $=id=>document.getElementById(id);
const MODEL_CONFIDENCE = 0.15;
const MODEL_NMS_IOU = 0.45;
let session,image,rawOutput,boxes=[],summary,filename='image',busy=false,modelReady=false,loadSequence=0;
let activeReference=null,sourceDimensions=null,selectionSequence=0;
const referenceIndex=fetch(new URL('./references/annotations.json',location.href)).then(r=>{if(!r.ok)throw new Error('Reference index unavailable');return r.json();}).catch(()=>null);
const isReference=()=>$('result-source').value==='reference';
const canvas=$('preview'),ctx=canvas.getContext('2d');
const status=(message,error=false)=>{$('status').textContent=message;$('status').classList.toggle('error',error);};
function controls() {
  $('analyze').disabled = (!modelReady && !isReference()) || !image || busy;
  $('file').disabled = busy;
  $('sample').disabled = busy;
  $('result-source').disabled = busy;
}
async function init(){
  try{
    if(!globalThis.ort)throw new Error('ONNX runtime could not load. Reload this page.');
    ort.env.wasm.wasmPaths=new URL('./vendor/',location.href).href;
    ort.env.wasm.numThreads=1;
    session=await ort.InferenceSession.create(new URL('./models/magnolia_detector.onnx',location.href).href,{executionProviders:['wasm'],graphOptimizationLevel:'all'});
    modelReady=true;status(image?'Image ready. Select Count Magnolia plants.':'Model ready. Choose an image to begin.');
  }catch(e){status(`Model unavailable: ${e.message}. Serve this folder over HTTP; check that models/ and vendor/ are deployed.`,true);}
  controls();
}
function draw(){
  ctx.drawImage(image,0,0,canvas.width,canvas.height);
  const scale=canvas.width/image.naturalWidth;
  const reference=isReference();
  ctx.lineWidth=Math.max(1.5,canvas.width/700);ctx.font=`600 ${Math.max(12,canvas.width/80)}px system-ui`;
  boxes.forEach((b,i)=>{
    const x=b.x1*scale,y=b.y1*scale,w=(b.x2-b.x1)*scale,h=(b.y2-b.y1)*scale;
    ctx.fillStyle=reference?'rgba(242,207,155,0.055)':'rgba(181,241,207,0.055)';ctx.fillRect(x,y,w,h);
    ctx.strokeStyle=reference?'#f2cf9b':'#b5f1cf';ctx.strokeRect(x,y,w,h);
    const corner=Math.min(11,w/3,h/3);
    ctx.beginPath();
    for(const [px,py,dx,dy] of [[x,y,1,1],[x+w,y,-1,1],[x,y+h,1,-1],[x+w,y+h,-1,-1]]){
      ctx.moveTo(px+dx*corner,py);ctx.lineTo(px,py);ctx.lineTo(px,py+dy*corner);
    }
    ctx.stroke();
    const text=String(i+1).padStart(2,'0'),tw=ctx.measureText(text).width+12,th=Math.max(20,canvas.width/50);
    const labelX=Math.min(x,canvas.width-tw),labelY=Math.max(th,y);
    ctx.fillStyle='#102720';ctx.fillRect(labelX,labelY-th,tw,th);ctx.fillStyle=reference?'#f2cf9b':'#d9f5e5';ctx.fillText(text,labelX+6,labelY-5);
  });
}
async function loadImage(source,name,reference=null,kind=null){
  const sequence=++loadSequence;
  rawOutput=null;boxes=[];summary=null;activeReference=null;sourceDimensions=null;$('result-source').value='model';$('reference-option').disabled=true;$('results').hidden=true;image=null;controls();
  try{
    const next=new Image();next.src=source;await next.decode();
    if(sequence!==loadSequence)return;
    if(next.naturalWidth*next.naturalHeight>60_000_000)throw new Error('Use an image smaller than 60 megapixels.');
    sourceDimensions={width:next.naturalWidth,height:next.naturalHeight};
    image=next;filename=name;activeReference=reference;
    if(reference&&kind==='original'){
      const crop=document.createElement('canvas');crop.width=1280;crop.height=Math.round(1280*reference.cropHeight/reference.originalWidth);
      crop.getContext('2d').drawImage(next,0,0,reference.originalWidth,reference.cropHeight,0,0,crop.width,crop.height);
      const recovered=new Image();recovered.src=crop.toDataURL('image/png');await recovered.decode();
      if(sequence!==loadSequence)return;
      image=recovered;
    }
    $('reference-option').disabled=!reference;
    $('reference-status').textContent=reference?`Matched ${reference.filename}. ${reference.boxes.length} provided boxes in the recoverable region. ${kind==='original'?'Damaged lower rows excluded. ':''}Choose Supplied annotations to display every provided box. This is a reference view, not model accuracy.`:'No exact dataset match. Automatic detections are available; supplied annotations cannot be assumed for this image.';
    const scale=Math.min(1,1600/Math.max(image.naturalWidth,image.naturalHeight));
    canvas.width=Math.round(image.naturalWidth*scale);canvas.height=Math.round(image.naturalHeight*scale);
    $('empty').hidden=true;canvas.hidden=false;$('filename').textContent=name;
    $('image-size').textContent=`${image.naturalWidth} × ${image.naturalHeight} PX${kind==='original'?' · RECOVERED REGION':''}`;draw();
    status(modelReady?'Image ready. Select Count Magnolia plants.':'Image ready. Waiting for model…');
  }catch(e){if(sequence!==loadSequence)return;canvas.hidden=true;$('empty').hidden=false;status(`Could not read image: ${e.message}`,true);}
  controls();
}
async function loadFile(file){
  if(!file||busy)return;
  const selection=++selectionSequence;
  if(!['image/jpeg','image/png','image/webp'].includes(file.type)&&!(file.type===''&&/\.(jpe?g|png|webp)$/i.test(file.name))){status('Choose a JPEG, PNG or WebP image.',true);return;}
  if(file.size>40*1024*1024){status('Choose an image smaller than 40 MB.',true);return;}
  const index=await referenceIndex;
  let match=null;
  if(index&&globalThis.crypto?.subtle){
    const digest=await crypto.subtle.digest('SHA-256',await file.arrayBuffer());
    const hash=Array.from(new Uint8Array(digest),v=>v.toString(16).padStart(2,'0')).join('');match=index.hashes[hash];
  }
  if(selection!==selectionSequence)return;
  const url=URL.createObjectURL(file);
  try{await loadImage(url,file.name,match?index.images[match.id]:null,match?.kind);}finally{URL.revokeObjectURL(url);}
}
function updateResults(){
  if(!image)return;
  if(isReference()&&activeReference)boxes=referenceBoxes(activeReference.boxes,image.naturalWidth,image.naturalHeight);
  else if(rawOutput)boxes=decode(rawOutput.data,rawOutput.dims,image.naturalWidth,image.naturalHeight,MODEL_CONFIDENCE,MODEL_NMS_IOU);
  else{boxes=[];$('results').hidden=true;draw();status('Select Count Magnolia plants to run the model.');return;}
  summary=summarize(boxes,image.naturalWidth,image.naturalHeight);
  $('count').textContent=summary.count;$('coverage').textContent=`${summary.boxCoveragePercent.toFixed(1)}%`;
  $('count-label').textContent=isReference()?'Annotated plants':'Plants detected';
  $('source-label').textContent=isReference()?'Annotations displayed':'Result source';
  $('source-value').textContent=isReference()?'100%':'Model';
  $('result-note').textContent = isReference()
    ? `Showing all ${activeReference.boxes.length} supplied annotation boxes in the recovered region. Box coverage measures the combined area of these rectangles.`
    : summary.count === 0
      ? 'The model found no Magnolia plants in this image.'
      : 'Review the boxes for missed plants or duplicate detections. Box coverage is the share of the image inside the boxes, counting overlaps once.';
  $('results').hidden=false;draw();
}
async function analyze(){
  if(isReference()&&activeReference&&image){updateResults();status('All supplied boxes displayed. Reference annotations are not model predictions.');return;}
  if(!session||!image||busy)return;
  busy=true;controls();status('Detecting plants on your device…');
  await new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve)));
  let tensor,outputs;
  try{
    const size=640,l=letterbox(image.naturalWidth,image.naturalHeight),input=document.createElement('canvas');input.width=input.height=size;
    const ic=input.getContext('2d',{willReadFrequently:true});ic.fillStyle='rgb(114,114,114)';ic.fillRect(0,0,size,size);ic.drawImage(image,l.left,l.top,l.resizedWidth,l.resizedHeight);
    const pixels=ic.getImageData(0,0,size,size).data,data=new Float32Array(3*size*size);
    for(let i=0;i<size*size;i++){data[i]=pixels[4*i]/255;data[size*size+i]=pixels[4*i+1]/255;data[2*size*size+i]=pixels[4*i+2]/255;}
    tensor=new ort.Tensor('float32',data,[1,3,size,size]);
    const started=performance.now();outputs=await session.run({[session.inputNames[0]]:tensor});
    const output=outputs[session.outputNames[0]];rawOutput={data:Float32Array.from(output.data),dims:[...output.dims]};
    updateResults();status(`Complete in ${((performance.now()-started)/1000).toFixed(1)} s.`);
  }catch(e){status(`Detection failed: ${e.message}`,true);}finally{tensor?.dispose();if(outputs)Object.values(outputs).forEach(t=>t.dispose());busy=false;controls();}
}
function download(blob,suffix){const link=document.createElement('a'),url=URL.createObjectURL(blob);link.href=url;link.download=filename.replace(/\.[^.]+$/,'')+suffix;link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
function report() {
  const reference = isReference();
  return {
    image: filename,
    width: image.naturalWidth,
    height: image.naturalHeight,
    sourceDimensions,
    coordinateSpace: 'displayed working image; known truncated originals use recovered upper crop resized to 1280px width',
    resultSource: reference ? 'provided_annotations' : 'model_predictions',
    annotationDisplayPercent: reference ? 100 : null,
    model: reference ? null : 'magnolia_detector.onnx',
    confidenceThreshold: reference ? null : MODEL_CONFIDENCE,
    nmsIoU: reference ? null : MODEL_NMS_IOU,
    ...summary,
    units: 'pixels; box coverage is union area, not accuracy or canopy area',
    detections: boxes.map((box, index) => ({ id: index + 1, label: 'Magnolia', ...box })),
  };
}
$('file').addEventListener('change',e=>loadFile(e.target.files[0]));
$('sample').addEventListener('click',async()=>{const selection=++selectionSequence,index=await referenceIndex;if(selection!==selectionSequence)return;await loadImage(new URL('./examples/magnolia-test.jpg',location.href).href,'magnolia-test.jpg',index?.images.find(r=>r.filename==='IMG_1751.jpg')??null,'crop');});
$('result-source').addEventListener('change',()=>{controls();updateResults();if(isReference())status('All supplied boxes displayed. Reference annotations are not model predictions.');});
$('analyze').addEventListener('click',analyze);
$('dropzone').addEventListener('dragover',e=>{e.preventDefault();$('dropzone').classList.add('dragover');});
$('dropzone').addEventListener('dragleave',()=>$('dropzone').classList.remove('dragover'));
$('dropzone').addEventListener('drop',e=>{e.preventDefault();$('dropzone').classList.remove('dragover');loadFile(e.dataTransfer.files[0]);});
window.addEventListener('dragover',e=>e.preventDefault());window.addEventListener('drop',e=>e.preventDefault());
$('json').addEventListener('click',()=>download(new Blob([JSON.stringify(report(),null,2)],{type:'application/json'}),'-detections.json'));
$('csv').addEventListener('click',()=>{
  const r=report();
  const rows=[['image','result_source','count','box_coverage_percent','confidence_threshold','id','label','confidence','x1_px','y1_px','x2_px','y2_px']];
  const prefix=[r.image,r.resultSource,r.count,r.boxCoveragePercent,r.confidenceThreshold??''];
  if(!boxes.length)rows.push([...prefix,'','','','','','','']);
  boxes.forEach((b,i)=>rows.push([...prefix,i+1,'Magnolia',b.score??'',b.x1,b.y1,b.x2,b.y2]));
  const escape=value=>{let s=String(value);if(/^[=+@\-]/.test(s))s="'"+s;return '"'+s.replaceAll('"','""')+'"';};
  download(new Blob([rows.map(row=>row.map(escape).join(',')).join('\r\n')],{type:'text/csv;charset=utf-8'}),'-detections.csv');
});
$('png').addEventListener('click',()=>canvas.toBlob(blob=>{if(blob)download(blob,'-annotated.png');},'image/png'));
init();
