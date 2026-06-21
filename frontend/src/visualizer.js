// NeuraLens — Canvas Visualizer (Light Theme: White / Red / Blue)
const VIZ = {
  blue: '#1a56db', blueLight: '#dbeafe', blueDark: '#1340b0',
  red: '#dc2626', redLight: '#fee2e2', redDark: '#b91c1c',
  white: '#ffffff', gray: '#8892aa', grayLight: '#e2e6ef', grayBg: '#f8f9fc',
  text: '#0f1629', textMuted: '#4a5578',
};

function setupHiDPICanvas(canvas, displayWidth, displayHeight) {
  const dpr = window.devicePixelRatio || 1;
  canvas.width = displayWidth * dpr;
  canvas.height = displayHeight * dpr;
  canvas.style.width = displayWidth + 'px';
  canvas.style.height = displayHeight + 'px';
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  return ctx;
}

function drawGrid(ctx, W, H, step=40) {
  ctx.strokeStyle = 'rgba(15,22,41,0.04)';
  ctx.lineWidth = 1;
  for (let x = 0; x <= W; x += step) { ctx.beginPath(); ctx.moveTo(x,0); ctx.lineTo(x,H); ctx.stroke(); }
  for (let y = 0; y <= H; y += step) { ctx.beginPath(); ctx.moveTo(0,y); ctx.lineTo(W,y); ctx.stroke(); }
}

function drawAxes(ctx, cx, cy, W, H) {
  ctx.strokeStyle = 'rgba(15,22,41,0.18)';
  ctx.lineWidth = 1.5;
  ctx.beginPath(); ctx.moveTo(cx, 0); ctx.lineTo(cx, H); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(0, cy); ctx.lineTo(W, cy); ctx.stroke();
}

function txt(ctx, s, x, y, color, size=11, weight='500') {
  ctx.fillStyle = color || VIZ.textMuted;
  ctx.font = `${weight} ${size}px 'JetBrains Mono', monospace`;
  ctx.textAlign = 'left';
  ctx.fillText(s, x, y);
}

// ════════════════════════════════════════
// MAIN DISPATCH
// ════════════════════════════════════════
function renderVisualization(canvas, layerId, params, inputData) {
  const acts = ['relu','leakyrelu','prelu','elu','selu','gelu','swish','mish','sigmoid','tanh','softplus'];
  const attnTypes = ['selfattn','crossattn','mha','causal','sparse','flashattn'];
  // Taller canvas for attention heatmaps (5x5 grid needs more vertical room) and RNN gate diagrams
  const W = 600;
  const H = attnTypes.includes(layerId) ? 360 : (['lstm','gru','bilstm','rnn','convlstm'].includes(layerId) ? 340 : 320);
  const ctx = setupHiDPICanvas(canvas, W, H);
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = VIZ.white;
  ctx.fillRect(0, 0, W, H);

  if (acts.includes(layerId)) drawActivationViz(ctx, W, H, layerId, params);
  else if (['softmax','softmaxout'].includes(layerId)) drawSoftmaxViz(ctx, W, H, params);
  else if (['dense','projection','linearout'].includes(layerId)) drawDenseViz(ctx, W, H, params);
  else if (['conv2d','conv1d','depthwise','dilated','separable','pointwise','conv3d','transposed'].includes(layerId)) drawConvViz(ctx, W, H, layerId, params);
  else if (['maxpool','avgpool','globalavg','globalmaxpool','adaptivepool'].includes(layerId)) drawPoolViz(ctx, W, H, layerId, params);
  else if (['batchnorm','layernorm','instancenorm','groupnorm','rmsnorm','spectralnorm'].includes(layerId)) drawNormViz(ctx, W, H, layerId, inputData);
  else if (['selfattn','crossattn','mha','causal','sparse','flashattn'].includes(layerId)) drawAttentionViz(ctx, W, H, layerId, params);
  else if (['dropout','spatialdrop','dropblock','stochdepth'].includes(layerId)) drawDropoutViz(ctx, W, H, params);
  else if (['posenc','rope'].includes(layerId)) drawPosEncViz(ctx, W, H, layerId, params);
  else if (['lstm','gru','bilstm','rnn','convlstm'].includes(layerId)) drawRNNViz(ctx, W, H, layerId);
  else if (['residual','highway','denseconn'].includes(layerId)) drawResidualViz(ctx, W, H, layerId);
  else if (['embedding','tokenembed'].includes(layerId)) drawEmbeddingViz(ctx, W, H);
  else if (['vaesample','latent','encoder','decoder','genblock','discblock'].includes(layerId)) drawGenerativeViz(ctx, W, H, layerId);
  else if (['ffn','gatedffn'].includes(layerId)) drawFFNViz(ctx, W, H, layerId);
  else if (layerId === 'moe') drawMoEViz(ctx, W, H, params);
  else drawDefaultViz(ctx, W, H, layerId);

  return { width: W, height: H };
}

// ════════════════════════════════════════
// ACTIVATION FUNCTION CURVES
// ════════════════════════════════════════
function actFn(id, x, p={}) {
  const a = p.alpha ?? p.ia ?? 0.1, b = p.beta ?? 1;
  switch(id) {
    case 'relu': return Math.max(p.threshold ?? 0, x);
    case 'leakyrelu': return x>0?x:a*x;
    case 'prelu': return x>0?x:(p.ia??0.25)*x;
    case 'elu': return x>0?x:(p.alpha??1)*(Math.exp(x)-1);
    case 'selu': return 1.0507*(x>0?x:1.6733*(Math.exp(x)-1));
    case 'gelu': return 0.5*x*(1+Math.tanh(Math.sqrt(2/Math.PI)*(x+0.044715*x**3)));
    case 'swish': return x/(1+Math.exp(-b*x));
    case 'mish': return x*Math.tanh(Math.log(1+Math.exp(x)));
    case 'sigmoid': return 1/(1+Math.exp(-x));
    case 'tanh': return Math.tanh(x);
    case 'softplus': return Math.log(1+Math.exp(Math.min(x,20)));
    default: return x;
  }
}

function drawActivationViz(ctx, W, H, id, params) {
  drawGrid(ctx, W, H);
  const cx = W/2, cy = H/2, range = 4;
  drawAxes(ctx, cx, cy, W, H);
  const isSat = ['sigmoid','softplus'].includes(id);
  const yR = isSat ? 1 : range;

  // Shade negative-zero region for ReLU family
  if (['relu','leakyrelu','prelu','elu','selu'].includes(id)) {
    const grad = ctx.createLinearGradient(0,0,cx,0);
    grad.addColorStop(0, 'rgba(220,38,38,0.06)');
    grad.addColorStop(1, 'rgba(220,38,38,0.01)');
    ctx.fillStyle = grad; ctx.fillRect(0, 0, cx, H);
  }

  // Main function curve (BLUE)
  ctx.beginPath(); ctx.strokeStyle = VIZ.blue; ctx.lineWidth = 3;
  ctx.lineJoin = 'round'; ctx.lineCap = 'round';
  for (let px = 2; px < W-2; px++) {
    const x = (px-cx)/(W/2)*range;
    const y = actFn(id, x, params);
    const py = cy - y*(H/2)/yR;
    px===2 ? ctx.moveTo(px,py) : ctx.lineTo(px,py);
  }
  ctx.stroke();

  // Derivative curve (RED dashed)
  ctx.beginPath(); ctx.strokeStyle = VIZ.red; ctx.lineWidth = 2; ctx.setLineDash([6,5]);
  for (let px = 4; px < W-4; px++) {
    const x = (px-cx)/(W/2)*range, dx = 0.01;
    const dy = (actFn(id, x+dx, params) - actFn(id, x-dx, params)) / (2*dx);
    const py = cy - dy*(H/2)/yR;
    px===4 ? ctx.moveTo(px,py) : ctx.lineTo(px,py);
  }
  ctx.stroke(); ctx.setLineDash([]);

  // Highlight point at x=1 with value
  const hx = cx + (1/range)*(W/2);
  const hy = cy - actFn(id, 1, params)*(H/2)/yR;
  ctx.beginPath(); ctx.arc(hx, hy, 5, 0, Math.PI*2);
  ctx.fillStyle = VIZ.blue; ctx.fill();
  ctx.strokeStyle = VIZ.white; ctx.lineWidth = 2; ctx.stroke();

  txt(ctx, `f(1) = ${actFn(id,1,params).toFixed(3)}`, hx+10, hy-8, VIZ.blueDark, 11, '700');

  // Labels
  txt(ctx, 'f(x)', cx+8, 16, VIZ.textMuted, 11);
  txt(ctx, 'x →', W-30, cy-8, VIZ.textMuted, 11);
  txt(ctx, LAYERS[id]?.name.toUpperCase() || id.toUpperCase(), 10, 20, VIZ.blue, 12, '700');
}

// ════════════════════════════════════════
// SOFTMAX BAR CHART
// ════════════════════════════════════════
function drawSoftmaxViz(ctx, W, H, params) {
  const T = params.temperature ?? params.T ?? 1;
  const vals = [2.3, 0.8, -0.5, 1.6, -1.2, 0.3];
  const m = Math.max(...vals);
  const exps = vals.map(x => Math.exp((x-m)/T));
  const s = exps.reduce((a,b)=>a+b,0);
  const probs = exps.map(e => e/s);

  txt(ctx, `Softmax  (T = ${T.toFixed(2)})  —  probabilities sum to 1.0000`, 16, 24, VIZ.textMuted, 12, '600');

  const barW = 70, pad = 30, chartH = H - 90, baseY = H - 50;
  vals.forEach((v, i) => {
    const x = pad + i*(barW+14);
    // Track
    ctx.fillStyle = VIZ.grayBg;
    ctx.fillRect(x, baseY-chartH, barW, chartH);
    ctx.strokeStyle = VIZ.grayLight; ctx.lineWidth = 1.5;
    ctx.strokeRect(x, baseY-chartH, barW, chartH);
    // Filled probability bar (BLUE, highest = RED)
    const fh = probs[i]*chartH;
    const isMax = probs[i] === Math.max(...probs);
    ctx.fillStyle = isMax ? VIZ.red : VIZ.blue;
    roundRect(ctx, x, baseY-fh, barW, fh, [0,0,6,6]);
    ctx.fill();
    // Percentage label
    ctx.fillStyle = isMax ? VIZ.redDark : VIZ.blueDark;
    ctx.font = '700 13px JetBrains Mono'; ctx.textAlign = 'center';
    ctx.fillText((probs[i]*100).toFixed(1)+'%', x+barW/2, baseY-fh-8);
    // Input value label
    ctx.fillStyle = VIZ.textMuted; ctx.font = '500 11px JetBrains Mono';
    ctx.fillText('x'+(i+1)+'='+v, x+barW/2, baseY+18);
  });
  ctx.textAlign = 'left';
  txt(ctx, 'T↓ sharper (argmax-like)     T↑ flatter (uniform)', 16, H-8, VIZ.gray, 10);
}

function roundRect(ctx, x, y, w, h, r) {
  if (typeof r === 'number') r = [r,r,r,r];
  ctx.beginPath();
  ctx.moveTo(x+r[0], y);
  ctx.lineTo(x+w-r[1], y);
  ctx.arcTo(x+w, y, x+w, y+r[1], r[1]);
  ctx.lineTo(x+w, y+h-r[2]);
  ctx.arcTo(x+w, y+h, x+w-r[2], y+h, r[2]);
  ctx.lineTo(x+r[3], y+h);
  ctx.arcTo(x, y+h, x, y+h-r[3], r[3]);
  ctx.lineTo(x, y+r[0]);
  ctx.arcTo(x, y, x+r[0], y, r[0]);
  ctx.closePath();
}

// ════════════════════════════════════════
// DENSE NEURAL CONNECTIONS
// ════════════════════════════════════════
function drawDenseViz(ctx, W, H, params) {
  const ni = 6, no = params.outSize ? Math.min(params.outSize, 6) : 5;
  const ix = 90, ox = W - 90;
  const iY = i => H/2 + (i - ni/2 + 0.5) * 36;
  const oY = o => H/2 + (o - no/2 + 0.5) * 42;

  txt(ctx, 'Dense Layer — y = Wx + b', 14, 20, VIZ.textMuted, 12, '600');

  // Connections
  for (let i=0; i<ni; i++) for (let o=0; o<no; o++) {
    const seed = Math.sin((i*7+o*13)*0.5);
    const w = seed;
    ctx.strokeStyle = w>0 ? `rgba(26,86,219,${Math.abs(w)*0.5})` : `rgba(220,38,38,${Math.abs(w)*0.4})`;
    ctx.lineWidth = Math.abs(w)*2;
    ctx.beginPath(); ctx.moveTo(ix, iY(i)); ctx.lineTo(ox, oY(o)); ctx.stroke();
  }

  // Input nodes (BLUE)
  for (let i=0; i<ni; i++) {
    ctx.beginPath(); ctx.arc(ix, iY(i), 14, 0, Math.PI*2);
    ctx.fillStyle = VIZ.blueLight; ctx.fill();
    ctx.strokeStyle = VIZ.blue; ctx.lineWidth = 2; ctx.stroke();
    ctx.fillStyle = VIZ.blueDark; ctx.font = '700 11px JetBrains Mono'; ctx.textAlign = 'center';
    ctx.fillText('x'+(i+1), ix, iY(i)+4);
  }
  // Output nodes (RED)
  for (let o=0; o<no; o++) {
    ctx.beginPath(); ctx.arc(ox, oY(o), 16, 0, Math.PI*2);
    ctx.fillStyle = VIZ.redLight; ctx.fill();
    ctx.strokeStyle = VIZ.red; ctx.lineWidth = 2; ctx.stroke();
    ctx.fillStyle = VIZ.redDark; ctx.font = '700 11px JetBrains Mono'; ctx.textAlign = 'center';
    ctx.fillText('y'+(o+1), ox, oY(o)+4);
  }
  ctx.textAlign = 'left';
  txt(ctx, '● blue = input neurons', ix-30, H-30, VIZ.blue, 10);
  txt(ctx, '● red = output neurons', ox-40, H-30, VIZ.red, 10);
  txt(ctx, 'Line thickness/opacity = weight magnitude', 14, H-10, VIZ.gray, 10);
}

// ════════════════════════════════════════
// CONVOLUTION SLIDING WINDOW
// ════════════════════════════════════════
function drawConvViz(ctx, W, H, id, params) {
  const k = params.kernelSize ?? params.k ?? 3;
  const gs = 24, sx = 20, sy = 56, cols = 8, rows = 7;
  const seedMat = Array.from({length:rows},(_,r)=>Array.from({length:cols},(_,c)=>(Math.sin((r*cols+c)*1.7)+1)/2));
  const kernel = Array.from({length:k},(_,r)=>Array.from({length:k},(_,c)=>Math.sin((r*k+c+5)*2.1)*0.6));

  txt(ctx, `Input feature map (${rows}×${cols})`, sx, 40, VIZ.textMuted, 11, '600');
  seedMat.forEach((r,ri)=>r.forEach((v,ci)=>{
    ctx.fillStyle = `rgba(26,86,219,${v*0.65+0.08})`;
    ctx.fillRect(sx+ci*gs, sy+ri*gs, gs-1, gs-1);
  }));
  // Sliding window highlight
  ctx.strokeStyle = VIZ.red; ctx.lineWidth = 2.5;
  ctx.strokeRect(sx+gs, sy+gs, k*gs, k*gs);

  const kx = sx + (cols+1.6)*gs;
  txt(ctx, `Kernel (${k}×${k})`, kx, 40, VIZ.textMuted, 11, '600');
  kernel.forEach((r,ri)=>r.forEach((v,ci)=>{
    ctx.fillStyle = v>0 ? `rgba(26,86,219,${Math.abs(v)*0.7+0.1})` : `rgba(220,38,38,${Math.abs(v)*0.7+0.1})`;
    ctx.fillRect(kx+ci*gs, sy+ri*gs, gs-1, gs-1);
    ctx.fillStyle = 'rgba(15,22,41,.6)'; ctx.font = '600 8px JetBrains Mono'; ctx.textAlign='center';
    ctx.fillText(v.toFixed(1), kx+ci*gs+gs/2, sy+ri*gs+gs/2+3);
  }));

  const fx = kx + (k+1)*gs + 14;
  ctx.fillStyle = VIZ.gray; ctx.font='20px sans-serif'; ctx.textAlign='center';
  ctx.fillText('→', fx, sy+k*gs/2+8);
  ctx.textAlign = 'left';
  txt(ctx, 'Feature', fx+22, sy+k*gs/2-6, VIZ.red, 11, '700');
  txt(ctx, 'Map', fx+22, sy+k*gs/2+10, VIZ.red, 11, '700');
  ctx.fillStyle = VIZ.redLight; ctx.fillRect(fx+20, sy, k*gs, k*gs);
  ctx.strokeStyle = VIZ.red; ctx.lineWidth = 2; ctx.strokeRect(fx+20, sy, k*gs, k*gs);

  txt(ctx, 'Red outline = sliding window position', sx+gs, sy+k*gs+18, VIZ.red, 10, '600');
  if (id === 'dilated') txt(ctx, `Dilation r=${params.r??2}: effective kernel = K+(K-1)(r-1)`, sx, H-10, VIZ.gray, 10);
}

// ════════════════════════════════════════
// POOLING
// ════════════════════════════════════════
function drawPoolViz(ctx, W, H, id, params) {
  const k = params.kernelSize ?? params.k ?? 2;
  const gs = 38, n = 6, sx = 30, sy = 56;
  const grid = Array.from({length:n},(_,r)=>Array.from({length:n},(_,c)=>Math.floor((Math.sin((r*n+c)*1.3)+1)*5)));

  txt(ctx, `Input (${n}×${n})`, sx, 40, VIZ.textMuted, 11, '600');
  grid.forEach((r,ri)=>r.forEach((v,ci)=>{
    const x=sx+ci*gs, y=sy+ri*gs;
    const bi = Math.floor(ri/k)*Math.floor(n/k)+Math.floor(ci/k);
    ctx.fillStyle = bi%2===0 ? 'rgba(26,86,219,.10)' : 'rgba(220,38,38,.10)';
    ctx.fillRect(x,y,gs-2,gs-2);
    ctx.strokeStyle = VIZ.grayLight; ctx.lineWidth=1; ctx.strokeRect(x,y,gs-2,gs-2);
    ctx.fillStyle = VIZ.text; ctx.font='600 13px JetBrains Mono'; ctx.textAlign='center';
    ctx.fillText(v, x+gs/2, y+gs/2+5);
  }));

  const outSize = Math.floor(n/k);
  const ox = sx + (n+1.3)*gs, oy = sy, ogs = gs*(n/outSize)*0.82;
  txt(ctx, `Output (${outSize}×${outSize})`, ox, 40, VIZ.textMuted, 11, '600');
  for (let ri=0; ri<outSize; ri++) for (let ci=0; ci<outSize; ci++) {
    const win = [];
    for (let dr=0; dr<k; dr++) for (let dc=0; dc<k; dc++) if (ri*k+dr<n && ci*k+dc<n) win.push(grid[ri*k+dr][ci*k+dc]);
    const v = id.includes('max') ? Math.max(...win) : Math.round(win.reduce((a,b)=>a+b,0)/win.length);
    const x = ox+ci*ogs, y = oy+ri*ogs;
    ctx.fillStyle = VIZ.redLight; ctx.fillRect(x,y,ogs-3,ogs-3);
    ctx.strokeStyle = VIZ.red; ctx.lineWidth=2; ctx.strokeRect(x,y,ogs-3,ogs-3);
    ctx.fillStyle = VIZ.redDark; ctx.font='700 15px JetBrains Mono'; ctx.textAlign='center';
    ctx.fillText(v, x+ogs/2, y+ogs/2+6);
  }
  ctx.textAlign = 'left';
  txt(ctx, id.includes('max') ? `MAX of each ${k}×${k} window` : `AVERAGE of each ${k}×${k} window`, sx, H-10, VIZ.gray, 10, '600');
}

// ════════════════════════════════════════
// NORMALIZATION BEFORE/AFTER
// ════════════════════════════════════════
function drawNormViz(ctx, W, H, id, inputData) {
  const data = (inputData || [1.2,-0.8,2.1,-1.5,0.3,1.8,-2.2,0.9,-0.4,1.1]).concat([1.5,-2.4]);
  const mu = data.reduce((a,b)=>a+b,0)/data.length;
  const v = data.reduce((a,x)=>a+(x-mu)**2,0)/data.length;
  const rms = Math.sqrt(data.reduce((a,x)=>a+x*x,0)/data.length+1e-5);
  const norm = data.map(x => id==='rmsnorm' ? x/rms : (x-mu)/Math.sqrt(v+1e-5));

  const bw = 18, pad = 24, mid = H/2+10, sc = 24;
  txt(ctx, 'Before normalization', pad, 22, VIZ.textMuted, 11, '600');
  data.forEach((v2,i) => {
    const x = pad+i*(bw+4), bh = Math.abs(v2)*sc, y = v2>0?mid-bh:mid;
    ctx.fillStyle = v2>0 ? 'rgba(26,86,219,.7)' : 'rgba(220,38,38,.6)';
    roundRect(ctx, x, y, bw, bh, 3); ctx.fill();
  });

  const ox = pad+(bw+4)*data.length+30;
  txt(ctx, `After ${LAYERS[id]?.name||id}`, ox, 22, VIZ.textMuted, 11, '600');
  norm.forEach((v2,i) => {
    const x = ox+i*(bw+4), bh = Math.abs(v2)*sc, y = v2>0?mid-bh:mid;
    ctx.fillStyle = v2>0 ? 'rgba(26,86,219,.7)' : 'rgba(220,38,38,.6)';
    roundRect(ctx, x, y, bw, bh, 3); ctx.fill();
  });

  ctx.strokeStyle = 'rgba(15,22,41,.15)'; ctx.lineWidth=1.5; ctx.setLineDash([5,5]);
  ctx.beginPath(); ctx.moveTo(0,mid); ctx.lineTo(W,mid); ctx.stroke(); ctx.setLineDash([]);

  txt(ctx, `μ = ${mu.toFixed(3)}   σ = ${Math.sqrt(v).toFixed(3)}   →   normalized: μ≈0, σ≈1`, 14, H-12, VIZ.gray, 10, '600');
}

// ════════════════════════════════════════
// ATTENTION HEATMAP
// ════════════════════════════════════════
function drawAttentionViz(ctx, W, H, id, params) {
  const toks = ['I','love','deep','learning','today'];
  const n = toks.length, cs = 52, sx = (W - n*(cs+6))/2, sy = 40;
  const scores = Array.from({length:n},(_,i)=>Array.from({length:n},(_,j)=>{
    if (id==='causal' && j>i) return -1;
    return (Math.sin((i*7+j*11)*0.9)+1)/2;
  }));
  const soft = scores.map(row => {
    const valid = row.filter(x=>x>=0);
    const mx = Math.max(...(valid.length?valid:[0]));
    const e = row.map(x=>x<0?0:Math.exp(x-mx));
    const s = e.reduce((a,b)=>a+b,0)||1;
    return e.map(x=>x/s);
  });

  ctx.textAlign = 'center';
  toks.forEach((t,i) => {
    ctx.fillStyle = VIZ.blueDark; ctx.font='600 11px Inter';
    ctx.fillText(t, sx+i*(cs+6)+cs/2, sy-10);
    ctx.fillText(t, sx-26, sy+i*(cs+6)+cs/2+4);
  });

  soft.forEach((row,r) => row.forEach((v,c) => {
    const x = sx+c*(cs+6), y = sy+r*(cs+6);
    if (id==='causal' && c>r) {
      ctx.fillStyle = VIZ.grayBg;
      roundRect(ctx,x,y,cs,cs,6); ctx.fill();
      ctx.strokeStyle = VIZ.grayLight; ctx.lineWidth=1; ctx.stroke();
      ctx.fillStyle = VIZ.gray; ctx.font='16px sans-serif'; ctx.fillText('×', x+cs/2, y+cs/2+6);
    } else {
      ctx.fillStyle = `rgba(220,38,38,${v*0.85+0.05})`;
      roundRect(ctx,x,y,cs,cs,6); ctx.fill();
      ctx.fillStyle = v>0.5 ? VIZ.white : VIZ.redDark;
      ctx.font='600 11px JetBrains Mono';
      ctx.fillText(v.toFixed(2), x+cs/2, y+cs/2+4);
    }
  }));
  ctx.textAlign = 'left';
  const labels = { selfattn:'Self Attention', mha:'Multi-Head Attention', causal:'Causal Attention (future masked)', crossattn:'Cross Attention', sparse:'Sparse Attention', flashattn:'Flash Attention (identical output, tiled compute)' };
  txt(ctx, (labels[id]||id) + '  —  rows = query, columns = key', 14, H-10, VIZ.gray, 10, '600');
}

// ════════════════════════════════════════
// DROPOUT MASK GRID
// ════════════════════════════════════════
function drawDropoutViz(ctx, W, H, params) {
  const p = params.rate ?? params.p ?? params.kp ? (1-(params.kp??0.9)) : 0.5;
  const n = 48, cols = 8, gs = 36, sx = (W - cols*gs)/2, sy = 60;
  const mask = Array.from({length:n},()=>Math.random()>p);
  txt(ctx, `Dropout (rate = ${p.toFixed(2)})`, 16, 28, VIZ.textMuted, 12, '600');

  mask.forEach((active,i) => {
    const r = Math.floor(i/cols), c = i%cols, x = sx+c*gs, y = sy+r*gs;
    ctx.fillStyle = active ? 'rgba(26,86,219,.12)' : 'rgba(220,38,38,.10)';
    roundRect(ctx, x, y, gs-4, gs-4, 6); ctx.fill();
    ctx.strokeStyle = active ? VIZ.blueLight : VIZ.redLight; ctx.lineWidth = 1.5; ctx.stroke();
    ctx.textAlign='center';
    if (!active) { ctx.fillStyle = VIZ.red; ctx.font='16px Inter'; ctx.fillText('×', x+gs/2-2, y+gs/2+6); }
    else { ctx.fillStyle = VIZ.blue; ctx.font='14px Inter'; ctx.fillText('✓', x+gs/2-2, y+gs/2+5); }
  });
  ctx.textAlign='left';
  const activeCount = mask.filter(Boolean).length;
  txt(ctx, `Active: ${activeCount}/${n}  (${(activeCount/n*100).toFixed(0)}%)   Dropped: ${n-activeCount}/${n}  (${((n-activeCount)/n*100).toFixed(0)}%)`, 16, H-10, VIZ.gray, 10, '600');
}

// ════════════════════════════════════════
// POSITIONAL ENCODING HEATMAP
// ════════════════════════════════════════
function drawPosEncViz(ctx, W, H, id, params) {
  const d = Math.min(params.d ?? 64, 64), T = 22;
  txt(ctx, `${id==='rope'?'RoPE (Rotary)':'Sinusoidal'} Positional Encoding`, 16, 24, VIZ.textMuted, 12, '600');
  const cw = (W-32)/T, ch = (H-60)/16, sx=16, sy=40;
  for (let pos=0; pos<T; pos++) for (let i=0; i<16; i++) {
    const v = id==='rope'
      ? Math.cos(pos*Math.pow(10000,-2*i/d))
      : (i%2===0 ? Math.sin(pos/Math.pow(10000,i*2/d)) : Math.cos(pos/Math.pow(10000,(i*2-1)/d)));
    ctx.fillStyle = v>0 ? `rgba(26,86,219,${Math.abs(v)*0.8+0.05})` : `rgba(220,38,38,${Math.abs(v)*0.8+0.05})`;
    ctx.fillRect(sx+pos*cw+1, sy+i*ch, cw-1, ch-1);
  }
  txt(ctx, 'position →', 16, H-22, VIZ.gray, 10);
  txt(ctx, '↑ dim', 16, sy-6, VIZ.gray, 10);
  txt(ctx, '■ blue = positive value     ■ red = negative value', 16, H-8, VIZ.gray, 10);
}

// ════════════════════════════════════════
// RNN / LSTM GATE DIAGRAM
// ════════════════════════════════════════
function drawRNNViz(ctx, W, H, id) {
  const steps = 5, sx = 60, sw = (W-120)/(steps-1), ny = H/2+70;
  txt(ctx, `${id.toUpperCase()} — hidden state across ${steps} timesteps`, 16, 24, VIZ.textMuted, 12, '600');

  if (id==='lstm' || id==='gru') {
    const gates = id==='lstm'
      ? [['Forget', VIZ.red],['Input', VIZ.blue],['Cell', VIZ.red],['Output', VIZ.blue]]
      : [['Update', VIZ.blue],['Reset', VIZ.red]];
    gates.forEach(([gn,gc],gi) => {
      const gv = (Math.sin(gi*2.3)+1)/2;
      const gx = 60+(gi*(W-120)/gates.length), gy = 40;
      ctx.fillStyle = gc+'18'; roundRect(ctx,gx,gy,(W-120)/gates.length-10,32,6); ctx.fill();
      ctx.strokeStyle = gc+'66'; ctx.lineWidth=1.5; ctx.stroke();
      ctx.fillStyle = gc; ctx.font='600 10px JetBrains Mono'; ctx.textAlign='center';
      ctx.fillText(gn+` (${gv.toFixed(2)})`, gx+((W-120)/gates.length-10)/2, gy+20);
    });
  }
  ctx.textAlign='left';
  for (let t=0; t<steps; t++) {
    const x = sx+t*sw, v=(Math.sin(t*1.7)+1)/2;
    ctx.beginPath(); ctx.arc(x, ny, 18, 0, Math.PI*2);
    ctx.fillStyle = `rgba(26,86,219,${v*0.3+0.1})`; ctx.fill();
    ctx.strokeStyle = VIZ.blue; ctx.lineWidth=2; ctx.stroke();
    ctx.fillStyle = VIZ.blueDark; ctx.font='600 10px JetBrains Mono'; ctx.textAlign='center';
    ctx.fillText('h'+t, x, ny+3);
    ctx.fillStyle = VIZ.gray; ctx.fillText('x'+t, x, ny+34);
    if (t<steps-1) {
      ctx.strokeStyle = VIZ.blueLight; ctx.lineWidth=2;
      ctx.beginPath(); ctx.moveTo(x+18,ny); ctx.lineTo(x+sw-18,ny); ctx.stroke();
      ctx.fillStyle = VIZ.blue; ctx.font='14px sans-serif';
      ctx.fillText('→', x+sw/2, ny-8);
    }
  }
  ctx.textAlign='left';
  if (id==='lstm') txt(ctx, 'Cell state flows along the bottom path — the "gradient highway"', 16, H-10, VIZ.gray, 10);
}

// ════════════════════════════════════════
// RESIDUAL CONNECTION
// ════════════════════════════════════════
function drawResidualViz(ctx, W, H, id) {
  const labels = { residual:'Residual: y = F(x) + x', highway:'Highway: y = T·F(x) + (1−T)·x', denseconn:'Dense: layer receives ALL prior feature maps' };
  txt(ctx, labels[id], 16, 24, VIZ.textMuted, 12, '600');
  const sx = 70, ex = W-70, my = H/2+10;

  ctx.strokeStyle = VIZ.blue; ctx.lineWidth=2.5;
  ctx.beginPath(); ctx.moveTo(sx,my); ctx.bezierCurveTo(sx+30,my-70,ex-30,my-70,ex,my); ctx.stroke();
  ctx.fillStyle = VIZ.blueDark; ctx.font='600 11px JetBrains Mono'; ctx.textAlign='center';
  ctx.fillText(id==='residual'?'Skip (identity)':id==='highway'?'Carry (1−T)':'Prior features', W/2, my-78);

  ctx.fillStyle = VIZ.redLight;
  roundRect(ctx, sx+30, my-26, W-200, 52, 10); ctx.fill();
  ctx.strokeStyle = VIZ.red; ctx.lineWidth=2; ctx.stroke();
  ctx.fillStyle = VIZ.redDark; ctx.font='600 13px JetBrains Mono';
  ctx.fillText(id==='residual'?'F(x)':id==='highway'?'Transform T·F(x)':'Layer L output', W/2, my+6);

  ctx.beginPath(); ctx.arc(ex, my, 18, 0, Math.PI*2);
  ctx.fillStyle = VIZ.blueLight; ctx.fill(); ctx.strokeStyle = VIZ.blue; ctx.lineWidth=2.5; ctx.stroke();
  ctx.fillStyle = VIZ.blueDark; ctx.font='700 18px Inter'; ctx.fillText('+', ex, my+7);
  ctx.textAlign='left';
}

// ════════════════════════════════════════
// EMBEDDING LOOKUP TABLE
// ════════════════════════════════════════
function drawEmbeddingViz(ctx, W, H) {
  const vocab = 10, d = 14, cw = Math.floor((W-32)/vocab)-2, ch = Math.floor((H-80)/d);
  txt(ctx, "Embedding table — each column is one token's vector", 16, 24, VIZ.textMuted, 12, '600');
  const selIdx = 3, sx = 16, sy = 36;
  for (let v=0; v<vocab; v++) for (let di=0; di<d; di++) {
    const x = sx+v*(cw+2), y = sy+di*(ch+1);
    const val = (Math.sin((v*d+di)*0.7)+1)/2;
    ctx.fillStyle = v===selIdx ? `rgba(220,38,38,${val*0.6+0.15})` : `rgba(26,86,219,${val*0.18+0.03})`;
    ctx.fillRect(x+2, y+1, cw, ch);
  }
  ctx.strokeStyle = VIZ.red; ctx.lineWidth=2.5;
  ctx.strokeRect(sx+selIdx*(cw+2)+2, sy, cw, d*(ch+1));
  ctx.fillStyle = VIZ.redDark; ctx.font='600 11px JetBrains Mono'; ctx.textAlign='center';
  ctx.fillText('Token #'+(selIdx+1), sx+selIdx*(cw+2)+cw/2+2, H-30);
  ctx.textAlign='left';
  txt(ctx, 'Real example: GPT-2 uses 50,257 × 768. This shows 10 × 14 illustratively.', 16, H-10, VIZ.gray, 10);
}

// ════════════════════════════════════════
// FFN / SWIGLU
// ════════════════════════════════════════
function drawFFNViz(ctx, W, H, id) {
  txt(ctx, id==='gatedffn' ? 'SwiGLU Gated FFN: (xW₁ ⊗ SiLU(xWg)) W₂' : 'FFN Block: Activation(xW₁+b₁)W₂+b₂', 16, 24, VIZ.textMuted, 12, '600');
  const ni=8, nm=28, no=8, iX=60, mX=W/2, oX=W-60;
  const iY = i => H/2 + (i-ni/2+0.5)*28, mY = m => H/2 + (m-nm/2+0.5)*9, oY = o => H/2 + (o-no/2+0.5)*28;

  for (let i=0;i<ni;i++) for (let m=0;m<nm;m++) {
    if ((i+m)%3!==0) continue;
    ctx.strokeStyle = 'rgba(26,86,219,.08)'; ctx.lineWidth=.5;
    ctx.beginPath(); ctx.moveTo(iX,iY(i)); ctx.lineTo(mX,mY(m)); ctx.stroke();
  }
  for (let m=0;m<nm;m++) for (let o=0;o<no;o++) {
    if ((m+o)%3!==0) continue;
    ctx.strokeStyle = 'rgba(220,38,38,.08)'; ctx.lineWidth=.5;
    ctx.beginPath(); ctx.moveTo(mX,mY(m)); ctx.lineTo(oX,oY(o)); ctx.stroke();
  }
  for (let i=0;i<ni;i++) { ctx.beginPath(); ctx.arc(iX,iY(i),9,0,Math.PI*2); ctx.fillStyle=VIZ.blueLight; ctx.fill(); ctx.strokeStyle=VIZ.blue; ctx.lineWidth=1.5; ctx.stroke(); }
  for (let m=0;m<nm;m++) { ctx.beginPath(); ctx.arc(mX,mY(m),4,0,Math.PI*2); ctx.fillStyle= id==='gatedffn' ? VIZ.red : VIZ.gray; ctx.fill(); }
  for (let o=0;o<no;o++) { ctx.beginPath(); ctx.arc(oX,oY(o),9,0,Math.PI*2); ctx.fillStyle=VIZ.redLight; ctx.fill(); ctx.strokeStyle=VIZ.red; ctx.lineWidth=1.5; ctx.stroke(); }

  ctx.textAlign='center'; ctx.fillStyle=VIZ.textMuted; ctx.font='600 10px JetBrains Mono';
  ctx.fillText('d_model', iX, H-16);
  ctx.fillText(id==='gatedffn'?'d_ff (gated, 8/3·d)':'d_ff (4×d)', mX, H-16);
  ctx.fillText('d_model', oX, H-16);
  ctx.textAlign='left';
}

// ════════════════════════════════════════
// MIXTURE OF EXPERTS
// ════════════════════════════════════════
function drawMoEViz(ctx, W, H, params) {
  const ne = Math.min(params.ne ?? 8, 8), topk = params.topk ?? 2;
  txt(ctx, `Mixture of Experts — ${ne} experts, top-${topk} active per token`, 16, 24, VIZ.textMuted, 12, '600');
  const ew = 64, eh = 56, startX = (W-(ne*(ew+10)))/2;
  const activeSet = new Set();
  while(activeSet.size < topk) activeSet.add(Math.floor(Math.random()*ne));

  for (let e=0; e<ne; e++) {
    const x = startX+e*(ew+10), y = 80, active = activeSet.has(e);
    ctx.fillStyle = active ? VIZ.blueLight : VIZ.grayBg;
    roundRect(ctx,x,y,ew,eh,8); ctx.fill();
    ctx.strokeStyle = active ? VIZ.blue : VIZ.grayLight; ctx.lineWidth = active?2.5:1.5; ctx.stroke();
    ctx.fillStyle = active ? VIZ.blueDark : VIZ.gray; ctx.font='600 10px JetBrains Mono'; ctx.textAlign='center';
    ctx.fillText('Expert '+(e+1), x+ew/2, y+eh/2-6);
    ctx.fillText(active?'ACTIVE':'idle', x+ew/2, y+eh/2+10);
    if (active) { ctx.fillStyle=VIZ.red; ctx.font='16px sans-serif'; ctx.fillText('▲', x+ew/2, y-6); }
  }
  ctx.textAlign='left';
  ctx.fillStyle = VIZ.redLight; roundRect(ctx, W/2-55, H/2+50, 110, 32, 8); ctx.fill();
  ctx.strokeStyle = VIZ.red; ctx.lineWidth=1.5; ctx.stroke();
  ctx.fillStyle = VIZ.redDark; ctx.font='600 10px JetBrains Mono'; ctx.textAlign='center';
  ctx.fillText('Gating Network', W/2, H/2+70);
  ctx.textAlign='left';
  txt(ctx, `Only ${topk}/${ne} experts compute — sparse activation = efficiency at scale`, 16, H-10, VIZ.gray, 10, '600');
}

// ════════════════════════════════════════
// GENERATIVE (VAE / ENCODER / DECODER)
// ════════════════════════════════════════
function drawGenerativeViz(ctx, W, H, id) {
  if (id === 'vaesample') {
    txt(ctx, 'VAE Reparameterization: z = μ + σ·ε,  ε~N(0,I)', 16, 24, VIZ.textMuted, 12, '600');
    for (let i=0; i<180; i++) {
      const u1=Math.random(),u2=Math.random();
      const rx = Math.sqrt(-2*Math.log(u1))*Math.cos(2*Math.PI*u2);
      const ry = Math.sqrt(-2*Math.log(u1))*Math.sin(2*Math.PI*u2);
      const x = W/2+rx*55, y = H/2+ry*45;
      const r = Math.sqrt((x-W/2)**2+(y-H/2)**2);
      ctx.beginPath(); ctx.arc(x,y,2.5,0,Math.PI*2);
      ctx.fillStyle = `rgba(220,38,38,${Math.max(0,1-r/100)})`; ctx.fill();
    }
    ctx.beginPath(); ctx.arc(W/2,H/2,65,0,Math.PI*2);
    ctx.strokeStyle = VIZ.blue; ctx.lineWidth=2; ctx.setLineDash([6,5]); ctx.stroke(); ctx.setLineDash([]);
    txt(ctx, 'N(0,I) prior boundary', W/2+45, H/2-50, VIZ.blueDark, 10, '600');
  } else if (id === 'encoder') {
    const layers = [110,85,60,32,16];
    txt(ctx, 'Encoder: input → compressed latent z', 16, 24, VIZ.textMuted, 12, '600');
    layers.forEach((ww,i) => {
      const x = 40+i*(W-80)/layers.length, cy = H/2+10;
      ctx.fillStyle = `rgba(26,86,219,${0.15+i*0.1})`;
      roundRect(ctx, x, cy-ww/2, (W-80)/layers.length-10, ww, 6); ctx.fill();
      ctx.strokeStyle = VIZ.blueLight; ctx.lineWidth=1; ctx.stroke();
    });
  } else if (id === 'decoder') {
    const layers = [16,32,60,85,110];
    txt(ctx, 'Decoder: latent z → reconstructed output', 16, 24, VIZ.textMuted, 12, '600');
    layers.forEach((ww,i) => {
      const x = 40+i*(W-80)/layers.length, cy = H/2+10;
      ctx.fillStyle = `rgba(220,38,38,${0.15+(4-i)*0.1})`;
      roundRect(ctx, x, cy-ww/2, (W-80)/layers.length-10, ww, 6); ctx.fill();
      ctx.strokeStyle = VIZ.redLight; ctx.lineWidth=1; ctx.stroke();
    });
  } else drawDefaultViz(ctx, W, H, id);
}

// ════════════════════════════════════════
// DEFAULT FALLBACK
// ════════════════════════════════════════
function drawDefaultViz(ctx, W, H, id) {
  ctx.fillStyle = VIZ.grayLight;
  ctx.font='600 16px Inter'; ctx.textAlign='center';
  ctx.fillText(LAYERS[id]?.name || id, W/2, H/2-10);
  ctx.fillStyle = VIZ.gray; ctx.font='12px JetBrains Mono';
  ctx.fillText(LAYERS[id]?.short || '', W/2, H/2+16);
  ctx.textAlign = 'left';
}
