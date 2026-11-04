import {letterbox,decode,summarize} from './detector-core.js';
const $=id=>document.getElementById(id);
let session,image,rawOutput,boxes=[],summary,filename='image',busy=false,modelReady=false,loadSequence=0;
const canvas=$('preview'),ctx=canvas.getContext('2d');
const status=(message,error=false)=>{$('status').textContent=message;$('status').classList.toggle('error',error);};
function controls(){ $('analyze').disabled=!modelReady||!image||busy;$('file').disabled=busy;$('sample').disabled=busy;$('confidence').disabled=busy; }
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
  ctx.lineWidth=Math.max(2,canvas.width/650);ctx.font=`600 ${Math.max(12,canvas.width/65)}px system-ui`;
  boxes.forEach((b,i)=>{
    const x=b.x1*scale,y=b.y1*scale,w=(b.x2-b.x1)*scale,h=(b.y2-b.y1)*scale;
    ctx.strokeStyle='#bcff72';ctx.strokeRect(x,y,w,h);
    const text=`${i+1} · ${Math.round(b.score*100)}%`,tw=ctx.measureText(text).width+10,th=Math.max(19,canvas.width/45);
    const labelX=Math.min(x,canvas.width-tw),labelY=Math.max(th,y);
    ctx.fillStyle='#143e2c';ctx.fillRect(labelX,labelY-th,tw,th);ctx.fillStyle='#fff';ctx.fillText(text,labelX+5,labelY-5);
  });
}
async function loadImage(source,name){
  const sequence=++loadSequence;
  rawOutput=null;boxes=[];summary=null;$('results').hidden=true;image=null;controls();
  try{
    const next=new Image();next.src=source;await next.decode();
    if(sequence!==loadSequence)return;
    if(next.naturalWidth*next.naturalHeight>60_000_000)throw new Error('Use an image smaller than 60 megapixels.');
    image=next;filename=name;
    const scale=Math.min(1,1600/Math.max(next.naturalWidth,next.naturalHeight));
    canvas.width=Math.round(next.naturalWidth*scale);canvas.height=Math.round(next.naturalHeight*scale);
    $('empty').hidden=true;canvas.hidden=false;$('filename').textContent=name;
    $('image-size').textContent=`${next.naturalWidth} × ${next.naturalHeight} PX`;draw();
    status(modelReady?'Image ready. Select Count Magnolia plants.':'Image ready. Waiting for model…');
  }catch(e){if(sequence!==loadSequence)return;canvas.hidden=true;$('empty').hidden=false;status(`Could not read image: ${e.message}`,true);}
  controls();
}
async function loadFile(file){
  if(!file||busy)return;
  if(!['image/jpeg','image/png','image/webp'].includes(file.type)){status('Choose a JPEG, PNG or WebP image.',true);return;}
  if(file.size>40*1024*1024){status('Choose an image smaller than 40 MB.',true);return;}
  const url=URL.createObjectURL(file);
  try{await loadImage(url,file.name);}finally{URL.revokeObjectURL(url);}
}
function updateResults(){
  if(!rawOutput||!image)return;
  const threshold=Number($('confidence').value)/100;
  boxes=decode(rawOutput.data,rawOutput.dims,image.naturalWidth,image.naturalHeight,threshold,.45);
  summary=summarize(boxes,image.naturalWidth,image.naturalHeight);
  $('count').textContent=summary.count;$('coverage').textContent=`${summary.boxCoveragePercent.toFixed(1)}%`;
  $('mean').textContent=summary.meanConfidence===null?'—':`${Math.round(summary.meanConfidence*100)}%`;
  $('result-note').textContent=`${Math.round(threshold*100)}% confidence cutoff · ${summary.count===0?'No plants detected at this threshold. Try a lower threshold or a clearer overhead image.':'Check for missed or repeated plants. Box coverage is not canopy area.'}`;
  $('results').hidden=false;draw();
}
async function analyze(){
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
    updateResults();status(`Complete in ${((performance.now()-started)/1000).toFixed(1)} s. Adjust confidence to review the count.`);
  }catch(e){status(`Detection failed: ${e.message}`,true);}finally{tensor?.dispose();if(outputs)Object.values(outputs).forEach(t=>t.dispose());busy=false;controls();}
}
function download(blob,suffix){const link=document.createElement('a'),url=URL.createObjectURL(blob);link.href=url;link.download=filename.replace(/\.[^.]+$/,'')+suffix;link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
function report(){return {image:filename,width:image.naturalWidth,height:image.naturalHeight,model:'magnolia_detector.onnx',confidenceThreshold:Number($('confidence').value)/100,nmsIoU:.45,...summary,units:'pixels; box coverage is union area, not canopy area',detections:boxes.map((b,i)=>({id:i+1,label:'Magnolia',...b}))};}
$('file').addEventListener('change',e=>loadFile(e.target.files[0]));
$('sample').addEventListener('click',()=>loadImage(new URL('./examples/magnolia-test.jpg',location.href).href,'magnolia-test.jpg'));
$('analyze').addEventListener('click',analyze);
$('confidence').addEventListener('input',()=>{$('confidence-value').value=`${$('confidence').value}%`;updateResults();});
$('dropzone').addEventListener('dragover',e=>{e.preventDefault();$('dropzone').classList.add('dragover');});
$('dropzone').addEventListener('dragleave',()=>$('dropzone').classList.remove('dragover'));
$('dropzone').addEventListener('drop',e=>{e.preventDefault();$('dropzone').classList.remove('dragover');loadFile(e.dataTransfer.files[0]);});
window.addEventListener('dragover',e=>e.preventDefault());window.addEventListener('drop',e=>e.preventDefault());
$('json').addEventListener('click',()=>download(new Blob([JSON.stringify(report(),null,2)],{type:'application/json'}),'-detections.json'));
$('csv').addEventListener('click',()=>{
  const r=report();
  const rows=[['image','count','box_coverage_percent','confidence_threshold','id','label','confidence','x1_px','y1_px','x2_px','y2_px']];
  const prefix=[r.image,r.count,r.boxCoveragePercent,r.confidenceThreshold];
  if(!boxes.length)rows.push([...prefix,'','','','','','','']);
  boxes.forEach((b,i)=>rows.push([...prefix,i+1,'Magnolia',b.score,b.x1,b.y1,b.x2,b.y2]));
  const escape=value=>{let s=String(value);if(/^[=+@\-]/.test(s))s="'"+s;return '"'+s.replaceAll('"','""')+'"';};
  download(new Blob([rows.map(row=>row.map(escape).join(',')).join('\r\n')],{type:'text/csv;charset=utf-8'}),'-detections.csv');
});
$('png').addEventListener('click',()=>canvas.toBlob(blob=>{if(blob)download(blob,'-annotated.png');},'image/png'));
init();
