// NeuraLens — Main Application Logic
let sel = null, activeTab = 'overview', mode = 'explore';
let curParams = {}, curInput = [], archLayers = [], collapsedCats = {}, compareA = null, compareB = null;

// ══════════════════════════════════════
// SIDEBAR
// ══════════════════════════════════════
function buildSidebar(filter = '') {
  const sb = document.getElementById('sidebar');
  sb.innerHTML = '';
  CATS.forEach(cat => {
    const visible = cat.layers.filter(id => {
      if (!filter) return true;
      const nm = (LAYERS[id]?.name || id).toLowerCase();
      return nm.includes(filter.toLowerCase()) || id.includes(filter.toLowerCase());
    });
    if (!visible.length) return;
    const collapsed = collapsedCats[cat.id];

    const head = document.createElement('div');
    head.className = 'sidebar-section-head';
    head.innerHTML = `
      <span class="ssh-dot" style="background:${cat.color}"></span>
      <span class="ssh-label">${cat.name}</span>
      <span class="ssh-count">${visible.length}</span>
      <span class="ssh-chevron${collapsed?'':' open'}">›</span>`;
    head.onclick = () => { collapsedCats[cat.id] = !collapsed; buildSidebar(filter); };
    sb.appendChild(head);

    if (!collapsed) {
      visible.forEach(id => {
        const ld = LAYERS[id]; if (!ld) return;
        const el = document.createElement('div');
        const isActive = mode === 'compare' ? (id === compareA || id === compareB) : (mode === 'builder' ? false : sel === id);
        el.className = 'layer-item' + (isActive ? ' active' : '');
        el.id = 'li-' + id;
        let badge = '';
        if (mode === 'compare') {
          if (id === compareA) badge = '<span class="li-badge" style="background:var(--blue-light);color:var(--blue)">A</span>';
          else if (id === compareB) badge = '<span class="li-badge" style="background:var(--red-light);color:var(--red)">B</span>';
        }
        el.innerHTML = `<span class="li-dot" style="background:${cat.color}"></span><span class="li-name">${ld.name}</span>${badge}`;
        el.onclick = () => selectLayer(id);
        sb.appendChild(el);
      });
    }
  });
}
function searchLayers(v) { buildSidebar(v); }

// ══════════════════════════════════════
// MODE SWITCHING
// ══════════════════════════════════════
function setMode(m) {
  mode = m;
  ['explore','learn','hardware','compare','builder'].forEach(x => {
    document.getElementById('mb-'+x)?.classList.toggle('active', x === m);
  });
  buildSidebar(document.getElementById('search-inp')?.value || '');
  if (m === 'builder') { renderBuilder(); return; }
  if (m === 'compare') { renderCompare(); return; }
  // Jump straight to the mode's dedicated tab so content is visible immediately
  if (m === 'learn') activeTab = 'learn';
  else if (m === 'hardware') activeTab = 'hardware';
  else if (activeTab === 'learn' || activeTab === 'hardware') activeTab = 'overview';
  if (sel) renderLayerContent();
  else document.getElementById('content').innerHTML = document.getElementById('welcome').outerHTML;
}

// ══════════════════════════════════════
// SELECT LAYER
// ══════════════════════════════════════
async function selectLayer(id) {
  if (!LAYERS[id]) return;
  sel = id;
  // Respect current mode: stay on Learn/Hardware tab if that's the active mode
  if (mode === 'learn') activeTab = 'learn';
  else if (mode === 'hardware') activeTab = 'hardware';
  else activeTab = 'overview';
  curInput = defaultInput(id);

  // Fetch backend params if available
  const info = await fetchLayerInfo(id);
  curParams = {};
  if (info && info.params) {
    info.params.forEach(p => curParams[p.id] = p.default);
  } else {
    // local param defaults
    const localDefaults = {
      relu: {threshold:0}, leakyrelu:{alpha:0.1}, prelu:{ia:0.25}, elu:{alpha:1},
      swish:{beta:1}, softmax:{temperature:1}, softmaxout:{T:1}, dropout:{rate:0.5},
      dense:{outSize:8}, maxpool:{kernelSize:2}, avgpool:{kernelSize:2}, conv2d:{kernelSize:3},
      mha:{heads:4,seqLen:8}, moe:{ne:8,topk:2}, ffn:{expansion:4}
    };
    curParams = localDefaults[id] || {};
  }

  if (mode === 'compare') {
    // Alternate which slot gets updated: fill A first, then B; clicking again on A's layer refills A
    if (!compareA || compareA === id) { compareA = id; }
    else if (!compareB || compareB === id) { compareB = id; }
    else { compareA = id; compareB = null; } // both filled with a new layer -> restart at A
    buildSidebar(document.getElementById('search-inp')?.value || '');
    renderCompare();
    return;
  }
  if (mode === 'builder') return;
  buildSidebar(document.getElementById('search-inp')?.value || '');
  renderLayerContent();
}

// ══════════════════════════════════════
// CATEGORY COLOR HELPER
// ══════════════════════════════════════
function catColorOf(id) {
  const layer = LAYERS[id];
  const cat = CATS.find(c => c.id === layer?.cat);
  return cat?.color || '#1a56db';
}
function catNameOf(id) {
  const layer = LAYERS[id];
  const cat = CATS.find(c => c.id === layer?.cat);
  return cat?.name || '';
}

// ══════════════════════════════════════
// RENDER LAYER SHELL + TABS
// ══════════════════════════════════════
async function renderLayerContent() {
  const ld = LAYERS[sel]; if (!ld) return;
  const content = document.getElementById('content');
  const cc = catColorOf(sel);
  const info = await fetchLayerInfo(sel);
  const desc = info?.description || LEARN_CONTENT[sel]?.description || `${ld.name} is a core deep learning operation.`;
  const uses = info?.uses || LEARN_CONTENT[sel]?.uses || ['General use'];
  const cx = info?.complexity || ld.cx;

  const tabs = ['overview','math','playground','visualize','impact'];
  if (mode === 'learn') tabs.push('learn');
  if (mode === 'hardware') tabs.push('hardware');

  content.innerHTML = `
    <div class="layer-header anim-in">
      <div class="layer-title-row">
        <span class="layer-name">${ld.name}</span>
        <span class="layer-cat-badge" style="background:${cc}12;color:${cc};border-color:${cc}30">${catNameOf(sel)}</span>
        <span class="layer-cx-badge">${cx}</span>
      </div>
      <div class="layer-desc">${desc}</div>
      <div class="layer-uses">${uses.map(u=>`<span class="use-chip">${u}</span>`).join('')}</div>
    </div>
    <div class="tabs">${tabs.map(t=>`<button class="tab-btn${activeTab===t?' active':''}" onclick="switchTab('${t}')">${tabIcon(t)} ${tabLabel(t)}</button>`).join('')}</div>
    <div id="tab-panel"></div>`;
  renderActiveTab();
}
function tabLabel(t) { return {overview:'Overview',math:'Math',playground:'Playground',visualize:'Visualize',impact:'Impact',learn:'Learn',hardware:'Hardware'}[t]||t; }
function tabIcon(t) { return {overview:'◉',math:'∑',playground:'▶',visualize:'◫',impact:'⚡',learn:'◎',hardware:'⊞'}[t]||''; }
function switchTab(t) { activeTab = t; renderLayerContent(); }

function renderActiveTab() {
  const panel = document.getElementById('tab-panel'); if (!panel) return;
  if (activeTab === 'overview') renderOverview(panel);
  else if (activeTab === 'math') renderMath(panel);
  else if (activeTab === 'playground') renderPlayground(panel);
  else if (activeTab === 'visualize') renderVisualizeTab(panel);
  else if (activeTab === 'impact') renderImpact(panel);
  else if (activeTab === 'learn') renderLearn(panel);
  else if (activeTab === 'hardware') renderHardware(panel);
}

// ══════════════════════════════════════
// PARAM SLIDER BUILDER
// ══════════════════════════════════════
async function getParamDefs() {
  const info = await fetchLayerInfo(sel);
  if (info?.params) return info.params;
  const localParamDefs = {
    relu:[{id:'threshold',label:'Threshold',min:-2,max:2,default:0,step:0.1}],
    leakyrelu:[{id:'alpha',label:'Alpha',min:0.001,max:0.5,default:0.1,step:0.005}],
    prelu:[{id:'ia',label:'Init Alpha',min:0.01,max:0.5,default:0.25,step:0.01}],
    elu:[{id:'alpha',label:'Alpha',min:0.1,max:3,default:1,step:0.1}],
    swish:[{id:'beta',label:'Beta',min:0.1,max:5,default:1,step:0.1}],
    softmax:[{id:'temperature',label:'Temperature',min:0.1,max:5,default:1,step:0.1}],
    softmaxout:[{id:'T',label:'Temperature',min:0.1,max:5,default:1,step:0.1}],
    dropout:[{id:'rate',label:'Drop Rate',min:0,max:0.9,default:0.5,step:0.05}],
    dense:[{id:'outSize',label:'Output Size',min:2,max:32,default:8,step:1}],
    maxpool:[{id:'kernelSize',label:'Window',min:2,max:6,default:2,step:1}],
    avgpool:[{id:'kernelSize',label:'Window',min:2,max:6,default:2,step:1}],
    conv2d:[{id:'kernelSize',label:'Kernel Size',min:1,max:7,default:3,step:1}],
    mha:[{id:'heads',label:'Num Heads',min:1,max:16,default:4,step:1},{id:'seqLen',label:'Seq Length',min:4,max:64,default:8,step:4}],
    moe:[{id:'ne',label:'Num Experts',min:2,max:8,default:8,step:1},{id:'topk',label:'Top-K',min:1,max:4,default:2,step:1}],
    ffn:[{id:'expansion',label:'Expansion',min:1,max:8,default:4,step:1}],
  };
  return localParamDefs[sel] || [];
}

function paramSliderHTML(p, onchangeFn) {
  const val = curParams[p.id] ?? p.default;
  const pct = ((val - p.min) / (p.max - p.min) * 100).toFixed(1);
  return `<div class="param-row">
    <span class="param-label">${p.label}</span>
    <input type="range" min="${p.min}" max="${p.max}" step="${p.step}" value="${val}" style="--pct:${pct}%"
      oninput="updateParam('${p.id}',this);${onchangeFn||''}">
    <span class="param-value" id="pv-${p.id}">${val}</span>
  </div>`;
}
function updateParam(id, el) {
  curParams[id] = parseFloat(el.value);
  const pv = document.getElementById('pv-' + id);
  if (pv) {
    const v = curParams[id];
    pv.textContent = Math.abs(v) < 0.01 ? v.toExponential(2) : Math.abs(v) < 1 ? v.toFixed(3) : Math.abs(v) < 10 ? v.toFixed(2) : Math.round(v);
  }
  const pct = ((curParams[id] - parseFloat(el.min)) / (parseFloat(el.max) - parseFloat(el.min)) * 100).toFixed(1);
  el.style.setProperty('--pct', pct + '%');
}

// ══════════════════════════════════════
// TAB: OVERVIEW
// ══════════════════════════════════════
async function renderOverview(panel) {
  panel.innerHTML = `<div class="panel anim-in"><div class="loading-overlay"><div class="spinner"></div>Loading layer data...</div></div>`;
  const info = await fetchLayerInfo(sel);
  const ld = LAYERS[sel];
  const cc = catColorOf(sel);
  const equation = info?.equation || LEARN_CONTENT[sel]?.equation || ld.short;
  const equationFull = info?.equationFull || '';
  const result = await computeForward(sel, curInput, curParams);
  const hw = getHWProfile(sel, info);

  panel.innerHTML = `<div class="panel anim-in">
    <div class="card">
      <div class="card-header"><span class="card-accent" style="background:${cc}"></span><span class="card-title">Data Flow</span></div>
      <div class="flow-diagram">
        <div class="flow-box fb-input">Input<br><small style="color:var(--text-muted);font-size:10px">x ∈ ℝⁿ</small></div>
        <div class="flow-arrow">→</div>
        <div class="flow-box fb-op">${ld.name}<br><small style="color:var(--text-muted);font-size:10px">${equation.split('\n')[0].substring(0,26)}</small></div>
        <div class="flow-arrow">→</div>
        <div class="flow-box fb-output">Output<br><small style="color:var(--text-muted);font-size:10px">y ∈ ℝᵘ</small></div>
      </div>
    </div>
    <div class="grid-2">
      <div class="card">
        <div class="card-header"><span class="card-accent" style="background:${cc}"></span><span class="card-title">Core Equation</span></div>
        <div class="eq-box">${equation.replace(/\n/g,'<br>')}</div>
        ${equationFull?`<div class="eq-sub">${equationFull}</div>`:''}
      </div>
      <div class="card">
        <div class="card-header"><span class="card-accent" style="background:var(--green)"></span><span class="card-title">Quick Metrics</span></div>
        <div class="grid-2">
          <div class="metric-card"><div class="metric-label">Parameters</div><div class="metric-value text-blue">${fmtNum(result.paramCount)}</div></div>
          <div class="metric-card"><div class="metric-label">Est. FLOPs</div><div class="metric-value text-red">${fmtNum(result.flops)}</div></div>
          <div class="metric-card"><div class="metric-label">Memory (FP32)</div><div class="metric-value">${fmtBytes(result.memory?.fp32)}</div></div>
          <div class="metric-card"><div class="metric-label">GPU Fit</div><div class="metric-value text-green">${hw.gpu}%</div></div>
        </div>
      </div>
    </div>
    <div class="card">
      <div class="card-header"><span class="card-accent" style="background:var(--orange)"></span><span class="card-title">Hardware Snapshot</span><span class="card-action" onclick="setMode('hardware')">Full analysis →</span></div>
      <div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:14px">
        <span class="hw-badge hw-badge-blue">${hw.type}</span>
        <span class="hw-badge hw-badge-green">${hw.parallelism}</span>
        <span class="hw-badge hw-badge-orange">${hw.bottleneck}</span>
        ${hw.tensorCore?'<span class="hw-badge hw-badge-purple">Tensor Core ✓</span>':''}
      </div>
      <div class="grid-3">
        ${[['GPU',hw.gpu,'var(--green)'],['FPGA',hw.fpga,'var(--blue)'],['ASIC',hw.asic,'var(--red)']].map(([l,v,c])=>`
          <div><div style="display:flex;justify-content:space-between;font-size:11.5px;margin-bottom:4px"><span style="color:var(--text-secondary)">${l}</span><span class="mono bold" style="color:${c}">${v}%</span></div><div class="bar-track"><div class="bar-fill" style="width:${v}%;background:${c}"></div></div></div>
        `).join('')}
      </div>
    </div>
  </div>`;
}

// ══════════════════════════════════════
// TAB: MATH
// ══════════════════════════════════════
async function renderMath(panel) {
  panel.innerHTML = `<div class="panel anim-in"><div class="loading-overlay"><div class="spinner"></div>Loading...</div></div>`;
  const info = await fetchLayerInfo(sel);
  const ld = LAYERS[sel];
  const cc = catColorOf(sel);
  const equation = info?.equation || LEARN_CONTENT[sel]?.equation || ld.short;
  const equationFull = info?.equationFull || '';
  const params = await getParamDefs();

  panel.innerHTML = `<div class="panel anim-in">
    <div class="card">
      <div class="card-header"><span class="card-accent" style="background:${cc}"></span><span class="card-title">Primary Formula</span></div>
      <div class="eq-box" style="font-size:15px">${equation.replace(/\n/g,'<br>')}</div>
      ${equationFull?`<div class="eq-sub" style="margin-top:10px">${equationFull}</div>`:''}
    </div>
    ${params.length?`<div class="card"><div class="card-header"><span class="card-accent" style="background:var(--orange)"></span><span class="card-title">Parameters</span></div>${params.map(p=>paramSliderHTML(p)).join('')}</div>`:''}
    <div class="grid-2">
      <div class="card">
        <div class="card-header"><span class="card-accent" style="background:#0891b2"></span><span class="card-title">Complexity</span></div>
        <div class="grid-2">
          <div class="metric-card"><div class="metric-label">Time</div><div class="metric-value" style="font-size:14px;color:#0891b2">${ld.cx}</div></div>
          <div class="metric-card"><div class="metric-label">Space</div><div class="metric-value text-blue" style="font-size:14px">${ld.sp}</div></div>
        </div>
      </div>
      <div class="card">
        <div class="card-header"><span class="card-accent" style="background:var(--purple)"></span><span class="card-title">Dimension Flow</span></div>
        <div style="display:flex;align-items:center;justify-content:center;gap:10px;padding:10px 0">
          <div style="text-align:center"><div class="small text-muted">INPUT</div><div class="mono bold text-blue">x ∈ ℝⁿ</div></div>
          <div style="color:var(--text-light)">⟹</div>
          <div style="text-align:center"><div class="small text-muted">OUTPUT</div><div class="mono bold text-red">y ∈ ℝᵘ</div></div>
        </div>
      </div>
    </div>
  </div>`;
}

// ══════════════════════════════════════
// TAB: PLAYGROUND
// ══════════════════════════════════════
async function renderPlayground(panel) {
  panel.innerHTML = `<div class="panel anim-in"><div class="loading-overlay"><div class="spinner"></div>Computing forward pass...</div></div>`;
  const ld = LAYERS[sel];
  const params = await getParamDefs();
  const result = await computeForward(sel, curInput, curParams);
  const out = result.output;

  panel.innerHTML = `<div class="panel anim-in">
    ${params.length?`<div class="card"><div class="card-header"><span class="card-accent" style="background:var(--orange)"></span><span class="card-title">Parameters</span></div>${params.map(p=>paramSliderHTML(p,'refreshPlayground()')).join('')}</div>`:''}
    <div class="card">
      <div class="card-header"><span class="card-accent" style="background:var(--blue)"></span><span class="card-title">Live Tensor Computation</span><span class="card-action" onclick="randomizeInput()">⟳ Randomize Input</span></div>
      <div style="display:grid;grid-template-columns:1fr 70px 1fr;gap:14px;align-items:start">
        <div>
          <div class="small text-muted" style="margin-bottom:8px;letter-spacing:.5px;text-transform:uppercase">Input</div>
          <div class="tensor-row" id="t-in">${curInput.map((v,i)=>`<div class="tensor-cell ${v>=0?'tc-positive':'tc-negative'} tc-anim" id="in${i}" style="animation-delay:${i*20}ms">${v>=0?'+':''}${v.toFixed(1)}</div>`).join('')}</div>
        </div>
        <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;padding-top:24px;gap:4px">
          <div style="font-size:22px;color:var(--purple)">⟹</div>
          <div class="small text-muted mono" style="text-align:center">${ld.name}</div>
        </div>
        <div>
          <div class="small text-muted" style="margin-bottom:8px;letter-spacing:.5px;text-transform:uppercase">Output</div>
          <div class="tensor-row" id="t-out">${out.map((v,i)=>{const cls=v===0?'tc-zero':(v>=0?'tc-output':'tc-negative');return`<div class="tensor-cell ${cls} tc-anim" id="out${i}" style="animation-delay:${i*20+100}ms">${v.toFixed(2)}</div>`}).join('')}</div>
        </div>
      </div>
    </div>
    ${result.steps && result.steps.length ? `<div class="card">
      <div class="card-header"><span class="card-accent" style="background:var(--green)"></span><span class="card-title">Step-by-Step Trace</span></div>
      ${result.steps.map((s,i)=>`<div class="trace-step"><div class="trace-num">${i+1}</div><div class="trace-label">${s.label}</div>${s.value?`<div class="trace-values">[${Array.isArray(s.value)?s.value.join(', '):s.value}]</div>`:''}</div>`).join('')}
    </div>` : ''}
    <div class="card">
      <div class="card-header"><span class="card-accent" style="background:var(--purple)"></span><span class="card-title">Output Statistics</span></div>
      <div class="grid-3">
        <div class="metric-card"><div class="metric-label">Min</div><div class="metric-value text-red" style="font-size:15px">${result.stats.outputMin}</div></div>
        <div class="metric-card"><div class="metric-label">Max</div><div class="metric-value text-green" style="font-size:15px">${result.stats.outputMax}</div></div>
        <div class="metric-card"><div class="metric-label">Mean</div><div class="metric-value text-blue" style="font-size:15px">${result.stats.outputMean}</div></div>
      </div>
    </div>
  </div>`;
}
async function randomizeInput() { curInput = curInput.map(()=>+(Math.random()*4-2).toFixed(2)); renderPlayground(document.getElementById('tab-panel')); }
async function refreshPlayground() { if (activeTab==='playground') renderPlayground(document.getElementById('tab-panel')); }

// ══════════════════════════════════════
// TAB: VISUALIZE
// ══════════════════════════════════════
async function renderVisualizeTab(panel) {
  const params = await getParamDefs();
  panel.innerHTML = `<div class="panel anim-in">
    ${params.length?`<div class="card"><div class="card-header"><span class="card-accent" style="background:var(--orange)"></span><span class="card-title">Parameters — adjust for live updates</span></div>${params.map(p=>paramSliderHTML(p,'redrawViz()')).join('')}</div>`:''}
    <div class="card" style="padding:0;overflow:hidden">
      <div style="padding:14px 18px 0"><div class="card-header" style="margin-bottom:10px"><span class="card-accent" style="background:var(--purple)"></span><span class="card-title">Live Visualization</span></div></div>
      <div class="viz-canvas-wrap">
        <canvas class="viz-canvas" id="viz-canvas"></canvas>
      </div>
    </div>
    <div class="card ai-box" id="viz-explain"></div>
  </div>`;
  setTimeout(redrawViz, 50);
}
function redrawViz() {
  const canvas = document.getElementById('viz-canvas');
  if (!canvas) return;
  renderVisualization(canvas, sel, curParams, curInput);
  updateVizExplanation();
}
function updateVizExplanation() {
  const el = document.getElementById('viz-explain'); if (!el) return;
  const tips = {
    relu:'The red-shaded region shows where inputs are negative and get zeroed. The blue curve is f(x); the dashed red line is the derivative f′(x) — notice it jumps from 0 to 1 exactly at x=0.',
    gelu:'Notice the slight dip below zero just left of x=0 — this non-monotonic behavior is what separates GELU from ReLU. The function smoothly gates rather than hard-cutting.',
    sigmoid:'The S-curve flattens for |x|>4 — this is exactly where gradients vanish. Watch how f′(x) (dashed) is near-zero at the extremes.',
    tanh:'Zero-centered, unlike sigmoid. Maximum gradient of 1.0 occurs exactly at x=0, visible as the peak of the dashed derivative curve.',
    softmax:'Try increasing temperature — the bars flatten toward equal probability. Decrease it and the highest-scoring bar (red) dominates completely.',
    dense:'Blue connections represent positive weights, red represents negative weights. Thicker/more opaque lines mean larger magnitude.',
    conv2d:'The red outline shows the kernel\'s current position as it slides over the input. Each position computes one output pixel via element-wise multiply and sum.',
    maxpool:'Each colored region is one pooling window. The output keeps only the strongest (maximum) signal from each region.',
    avgpool:'Each colored region is averaged into a single output value — smoother than max pooling but loses peak information.',
    batchnorm:'Left side shows raw activations (unequal scale). Right side shows the same values rescaled to mean≈0, variance≈1.',
    layernorm:'Same normalization concept as BatchNorm, but computed per-sample instead of per-batch — works even with batch size 1.',
    mha:'This heatmap shows attention weights after softmax. Each row sums to 1.0 — it represents how much each query token "looks at" every key token.',
    causal:'The gray ×\'d cells are masked out — a token can never attend to future tokens. This enforces the autoregressive property needed for text generation.',
    lstm:'Each gate (forget/input/cell/output) controls a different aspect of memory. The cell state flowing along the bottom is what prevents vanishing gradients over long sequences.',
    dropout:'Green checkmarks are active neurons; red ×\'s are dropped this iteration. The pattern changes randomly every forward pass during training.',
    posenc:'Blue cells are positive values, red are negative. Each column (position) has a unique combination across all dimension rows — this is the position "fingerprint".',
    vaesample:'Red dots are sampled latent vectors. The dashed blue circle shows the N(0,I) prior that VAE training pushes the distribution toward.',
    ffn:'The wide middle layer (light dots) expands the representation 4× before contracting back — this is where most transformer FLOPs and parameters live.',
    moe:'Blue boxes are "active" experts selected for this token; gray boxes are idle and contribute zero compute — this sparsity is what makes MoE efficient at scale.',
  };
  el.innerHTML = `<div class="ai-box-header"><span class="ai-label">✦ AI Visualization Guide</span></div><div class="ai-text">${tips[sel] || 'This visualization shows ' + LAYERS[sel]?.name + ' behavior in real time. Adjust the parameters above to see how the output changes.'}</div>`;
}

// ══════════════════════════════════════
// TAB: IMPACT
// ══════════════════════════════════════
async function renderImpact(panel) {
  panel.innerHTML = `<div class="panel anim-in"><div class="loading-overlay"><div class="spinner"></div>Computing impact metrics...</div></div>`;
  const params = await getParamDefs();
  const result = await computeForward(sel, curInput, curParams);
  const info = await fetchLayerInfo(sel);
  const hw = getHWProfile(sel, info);
  const mem = result.memory || {fp32:0,fp16:0,int8:0,int4:0};

  panel.innerHTML = `<div class="panel anim-in">
    ${params.length?`<div class="card"><div class="card-header"><span class="card-accent" style="background:var(--orange)"></span><span class="card-title">Tune Parameters</span></div>${params.map(p=>paramSliderHTML(p,'refreshImpact()')).join('')}</div>`:''}
    <div class="grid-3" style="margin-bottom:14px">
      <div class="metric-card"><div class="metric-label">FLOPs</div><div class="metric-value text-blue">${fmtNum(result.flops)}</div><div class="metric-sub">floating-point ops</div></div>
      <div class="metric-card"><div class="metric-label">Parameters</div><div class="metric-value" style="color:var(--purple)">${fmtNum(result.paramCount)}</div><div class="metric-sub">trainable weights</div></div>
      <div class="metric-card"><div class="metric-label">Weight Memory</div><div class="metric-value text-red">${fmtBytes(mem.fp32)}</div><div class="metric-sub">FP32</div></div>
    </div>
    <div class="card">
      <div class="card-header"><span class="card-accent" style="background:var(--green)"></span><span class="card-title">Platform Suitability</span></div>
      ${[['GPU / NPU',hw.gpu,'var(--green)','cuBLAS, Tensor Cores'],['FPGA (Xilinx/Intel)',hw.fpga,'var(--blue)','DSP slices, BRAMs'],['ASIC / Custom NPU',hw.asic,'var(--red)','Dedicated silicon']].map(([l,v,c,n])=>`
        <div class="hw-platform-row">
          <div class="hw-platform-name">${l}<div class="hw-platform-sub">${n}</div></div>
          <div class="hw-bar-wrap"><div class="bar-track" style="height:9px"><div class="bar-fill" style="width:${v}%;background:${c}"></div></div></div>
          <div class="hw-score" style="color:${c}">${v}%</div>
        </div>
      `).join('')}
    </div>
    <div class="card">
      <div class="card-header"><span class="card-accent" style="background:var(--orange)"></span><span class="card-title">Quantization Impact</span></div>
      <div class="quant-grid">
        ${[['FP32',fmtBytes(mem.fp32),'var(--text-muted)','baseline'],['FP16',fmtBytes(mem.fp16),'var(--blue)','2× smaller'],['INT8',fmtBytes(mem.int8),'var(--green)','4× smaller'],['INT4',fmtBytes(mem.int4),'var(--red)','8× smaller']].map(([p,v,c,n])=>`
          <div class="quant-card"><div class="quant-label">${p}</div><div class="quant-val" style="color:${c}">${v}</div><div class="quant-sub">${n}</div></div>
        `).join('')}
      </div>
    </div>
    <div class="card">
      <div class="card-header"><span class="card-accent" style="background:var(--red)"></span><span class="card-title">Batch Size Scaling</span></div>
      <div class="grid-4">
        ${[1,8,32,128].map(bs=>`<div class="metric-card" style="text-align:center"><div class="metric-label">Batch ${bs}</div><div class="mono bold text-blue" style="font-size:13px">${fmtNum(result.flops*bs)}</div><div class="metric-sub">FLOPs</div></div>`).join('')}
      </div>
    </div>
  </div>`;
}
async function refreshImpact() { if (activeTab==='impact') renderImpact(document.getElementById('tab-panel')); }

// ══════════════════════════════════════
// TAB: LEARN (with AI explanations)
// ══════════════════════════════════════
async function renderLearn(panel) {
  panel.innerHTML = `<div class="panel anim-in"><div class="loading-overlay"><div class="spinner"></div>Loading learning content...</div></div>`;
  const info = await fetchLayerInfo(sel);
  const lc = LEARN_CONTENT[sel] || {};

  const levels = [
    { key:'beginner', title:'Beginner', icon:'🌱', color:'var(--green)', text: info?.beginner || lc.beginner || defaultBeginner() },
    { key:'intermediate', title:'Intermediate', icon:'⚡', color:'var(--blue)', text: info?.intermediate || lc.intermediate || defaultIntermediate() },
    { key:'hardware', title:'Hardware Engineer', icon:'🔧', color:'var(--orange)', text: info?.hardware || lc.hardware || defaultHardware() },
    { key:'research', title:'Research Level', icon:'🔬', color:'var(--red)', text: info?.research || lc.research || defaultResearch() },
  ];
  const interviewQ = info?.interviewQ || lc.interviewQ || ['Explain the mathematical operation this layer performs.','What are the tradeoffs vs alternative layers?','How would you implement this on FPGA/ASIC?'];
  const mistakes = info?.mistakes || lc.mistakes || ['Not verifying input/output shapes before training.','Misunderstanding training vs inference behavior.','Choosing hyperparameters without understanding their effect.'];

  panel.innerHTML = `<div class="panel anim-in">
    <div class="card ai-box" style="margin-bottom:16px">
      <div class="ai-box-header"><span class="ai-label">✦ AI-Generated Summary</span></div>
      <div class="ai-text">${aiSummaryFor(sel, info)}</div>
    </div>
    ${levels.map((l,i)=>`
      <div class="learn-accordion">
        <div class="learn-head" onclick="toggleLearn(${i})">
          <span class="learn-icon">${l.icon}</span>
          <span class="learn-title">${l.title}</span>
          <span class="learn-level-badge" style="background:${l.color}15;color:${l.color};border-color:${l.color}40">${l.title.toUpperCase()}</span>
          <span class="learn-chevron${i===0?' open':''}" id="lc-${i}">›</span>
        </div>
        <div class="learn-body${i===0?' open':''}" id="lb-${i}"><p>${l.text}</p></div>
      </div>
    `).join('')}
    <div class="card" style="margin-top:14px">
      <div class="card-header"><span class="card-accent" style="background:var(--orange)"></span><span class="card-title">Interview Questions</span></div>
      <ul class="iq-list">${interviewQ.map((q,i)=>`<li class="iq-item"><span class="iq-num">${i+1}</span>${q}</li>`).join('')}</ul>
    </div>
    <div class="card">
      <div class="card-header"><span class="card-accent" style="background:var(--red)"></span><span class="card-title">Common Mistakes</span></div>
      ${mistakes.map(m=>`<div class="mistake-item"><span class="mistake-icon">⚠</span>${m}</div>`).join('')}
    </div>
  </div>`;
}
function toggleLearn(i) {
  document.getElementById('lb-'+i)?.classList.toggle('open');
  document.getElementById('lc-'+i)?.classList.toggle('open');
}
function defaultBeginner() { return `${LAYERS[sel]?.name} is a fundamental deep learning operation that transforms input tensors to help the network learn useful representations.`; }
function defaultIntermediate() { return `Implementation details for ${LAYERS[sel]?.name}: ${LAYERS[sel]?.short}. Understanding the gradient behavior and numerical stability is key to using this layer correctly in practice.`; }
function defaultHardware() { return `On FPGA/ASIC, ${LAYERS[sel]?.name} maps to ${getHWProfile(sel)?.type || 'standard compute'} hardware. Parallelism: ${getHWProfile(sel)?.parallelism || 'moderate'}.`; }
function defaultResearch() { return `${LAYERS[sel]?.name} continues to see active research into more efficient variants, hardware co-design, and theoretical understanding at scale.`; }

function aiSummaryFor(id, info) {
  const ld = LAYERS[id];
  const eq = info?.equation || ld?.short || '';
  const cat = catNameOf(id);
  return `<strong>${ld?.name}</strong> is a <strong>${cat}</strong> layer defined by <code>${eq}</code>. ` +
    `It has time complexity <strong>${ld?.cx}</strong> and space complexity <strong>${ld?.sp}</strong>. ` +
    `${info?.description || LEARN_CONTENT[id]?.description || ''} Use the tabs above to explore its math, run it interactively in the Playground, see it visualized, and analyze its hardware/compute impact.`;
}

// ══════════════════════════════════════
// TAB: HARDWARE (enhanced)
// ══════════════════════════════════════
async function renderHardware(panel) {
  panel.innerHTML = `<div class="panel anim-in"><div class="loading-overlay"><div class="spinner"></div>Analyzing hardware profile...</div></div>`;
  const info = await fetchLayerInfo(sel);
  const hw = getHWProfile(sel, info);
  const lc = LEARN_CONTENT[sel] || {};
  const hwNotes = info?.hwNotes || lc.hardware || 'Standard compute mapping — see optimization notes below.';
  const quant = info?.quantization || { fp32:'100%', fp16:'50%', int8:'25%', int4:'12.5%' };

  panel.innerHTML = `<div class="panel anim-in">
    <div class="grid-2">
      <div class="card">
        <div class="card-header"><span class="card-accent" style="background:var(--blue)"></span><span class="card-title">Compute Classification</span></div>
        <div style="font-size:13px;line-height:2.2;color:var(--text-secondary)">
          Type: <span class="mono bold text-blue">${hw.type}</span><br>
          Parallelism: <span class="mono bold text-green">${hw.parallelism}</span><br>
          Bottleneck: <span class="mono bold" style="color:var(--orange)">${hw.bottleneck}</span><br>
          Bandwidth: <span class="mono bold" style="color:var(--orange)">${hw.bandwidth}</span><br>
          Tensor Cores: <span class="mono bold" style="color:${hw.tensorCore?'var(--green)':'var(--text-muted)'}">${hw.tensorCore?'✓ WMMA Applicable':'✗ Not applicable'}</span>
        </div>
      </div>
      <div class="card">
        <div class="card-header"><span class="card-accent" style="background:var(--green)"></span><span class="card-title">Platform Scores</span></div>
        ${[['GPU / NPU',hw.gpu,'var(--green)','cuBLAS, Tensor Cores'],['FPGA (Xilinx/Intel)',hw.fpga,'var(--blue)','DSP slices, BRAMs'],['ASIC / Custom',hw.asic,'var(--red)','Dedicated silicon']].map(([l,v,c,n])=>`
          <div class="hw-platform-row">
            <div class="hw-platform-name">${l}<div class="hw-platform-sub">${n}</div></div>
            <div class="hw-bar-wrap"><div class="bar-track" style="height:9px"><div class="bar-fill" style="width:${v}%;background:${c}"></div></div></div>
            <div class="hw-score" style="color:${c}">${v}%</div>
          </div>
        `).join('')}
      </div>
    </div>

    <div class="card">
      <div class="card-header"><span class="card-accent" style="background:var(--purple)"></span><span class="card-title">Memory Access Pattern</span></div>
      ${memoryPatternRows(sel)}
    </div>

    <div class="card ai-box">
      <div class="ai-box-header"><span class="ai-label">✦ AI Hardware Analysis</span></div>
      <div class="ai-text">${hwNotes}</div>
    </div>

    <div class="card">
      <div class="card-header"><span class="card-accent" style="background:#0891b2"></span><span class="card-title">FPGA Implementation Notes</span></div>
      <div style="font-size:13px;color:var(--text-secondary);line-height:1.85">${fpgaNotesFor(sel, hw)}</div>
    </div>

    <div class="card">
      <div class="card-header"><span class="card-accent" style="background:var(--red)"></span><span class="card-title">Quantization & Precision</span></div>
      <div class="quant-grid">
        ${Object.entries(quant).map(([k,v])=>`<div class="quant-card"><div class="quant-label">${k.toUpperCase()}</div><div class="quant-val text-blue" style="font-size:12px">${v}</div></div>`).join('')}
      </div>
    </div>

    <div class="card">
      <div class="card-header"><span class="card-accent" style="background:var(--orange)"></span><span class="card-title">Optimization Opportunities</span></div>
      <div style="font-size:13px;color:var(--text-secondary);line-height:1.9">${optimizationTipsFor(sel, hw)}</div>
    </div>
  </div>`;
}

function memoryPatternRows(id) {
  const patterns = {
    relu: [['Read','Sequential, 1 pass'],['Write','Sequential, 1 pass'],['Total BW','2× tensor size'],['Cache','Streaming, minimal reuse']],
    dense: [['Input','Sequential, reused per row'],['Weights','High reuse = batch_size×'],['Output','Sequential write'],['Tiling','block = √(L2/sizeof(W))']],
    conv2d: [['Input','Sliding window, K>1 non-seq'],['Weights','Reused H_out×W_out times'],['Output','Sequential by channel'],['im2col','Memory copy trades for GEMM']],
    mha: [['QKV proj','Dense, streamed 3×'],['Score matrix','n×n — HBM bottleneck!'],['FlashAttn','SRAM tiling avoids n×n'],['KV cache','2×T×d×h×L bytes/layer']],
    batchnorm: [['Fwd train','3 reads (input,γ,β)'],['Fwd inference','Zero cost if folded'],['Backward','2× input reads + grad write']],
  };
  const rows = patterns[id] || [['Read','Sequential pass over input'],['Write','Sequential pass over output'],['Pattern','Standard streaming access']];
  return rows.map(([l,v])=>`<div class="memory-pattern-row"><div class="mp-label">${l}</div><div class="mp-val">${v}</div></div>`).join('');
}

function fpgaNotesFor(id, hw) {
  const notes = {
    relu:'<strong>Resources:</strong> 1 comparator/lane, 0 DSPs, 0 BRAMs. 1-clock latency. Fuse with preceding layer to skip BRAM round-trip. SIMD: 16-64 elements/cycle.',
    dense:'<strong>Architecture:</strong> Systolic array. 8×8 PE array = 128 MACs/cycle. Weights in BRAM (36Kb/RAMB36). Accumulate via DSP48E2 chain.',
    conv2d:'<strong>Line buffer:</strong> K-1 rows in BRAM shift registers. Winograd 3×3: 2.25× DSP reduction. 64-256 MACs/cycle typical.',
    mha:'<strong>Softmax challenge:</strong> needs parallel reduction + division. n×n infeasible for n>256 on-chip. Use Flash Attention tiling with BRAM blocks.',
    batchnorm:'<strong>Inference:</strong> fold into conv — zero overhead. Training: tree reduction for mean/var. γ,β in registers.',
    lstm:'<strong>Fusion:</strong> stack [Wf;Wi;Wc;Wo] into one matrix (4H×(H+d)) — single large GEMM. Sequential h_t dependency limits pipeline depth.',
  };
  return notes[id] || `Map ${hw.type} operations to DSP slices. Use streaming AXI4-Stream interfaces. Pipeline and unroll inner loops. Fixed-point 16-bit typically sufficient for inference.`;
}

function optimizationTipsFor(id, hw) {
  const tips = {
    relu:'• Fuse with preceding conv/linear — saves one memory pass<br>• INT8 exact at zero — no rounding error at threshold<br>• In-place op (x.relu_()) saves memory allocation',
    dense:'• INT8 quantization → 4× throughput on Tensor Cores<br>• Batch multiple inferences for GPU utilization (batch ≥ 32)<br>• LoRA: decompose ΔW=AB, train only A, B (huge param savings)',
    conv2d:'• Winograd 3×3 → 2.25× fewer multiplications<br>• Depthwise+pointwise separable → ~9× fewer params<br>• Fuse conv+BN+ReLU into a single kernel',
    mha:'• Use Flash Attention for sequences >512 (3-8× faster, O(n) memory)<br>• Quantize KV cache to INT4 → 4× memory savings<br>• Grouped Query Attention → share K/V across heads',
    batchnorm:'• Fold into preceding conv at inference → zero runtime cost<br>• Use FP16/BF16 for statistics computation<br>• Replace with LayerNorm for transformer/small-batch use cases',
    dropout:'• Training only — compiles to NOP at inference<br>• GPU: Philox RNG for fast, parallel mask generation<br>• Spatial dropout → fewer RNG calls for CNNs',
  };
  return tips[id] || `• Profile before optimizing to find the true bottleneck<br>• Fuse adjacent elementwise ops into a single kernel pass<br>• Quantize to FP16/INT8 for inference efficiency<br>• This layer is ${hw.bottleneck} — optimize accordingly`;
}

// ══════════════════════════════════════
// COMPARE TAB
// ══════════════════════════════════════
async function renderCompare() {
  const content = document.getElementById('content');
  const ldA = compareA ? LAYERS[compareA] : null;
  const ldB = compareB ? LAYERS[compareB] : null;

  content.innerHTML = `
    <div class="layer-header">
      <div class="layer-title-row">
        <span class="layer-name">⇆ Layer Comparison</span>
        ${(ldA || ldB) ? `<button class="btn btn-sm" onclick="resetCompare()">🗑 Clear Selection</button>` : ''}
      </div>
      <div class="layer-desc">Click two layers in the sidebar to compare them side by side — equations, FLOPs, parameters, memory, and hardware suitability.</div>
    </div>
    <div class="panel">
      <div class="compare-grid">
        <div id="cmp-a">${ldA ? '<div class="loading-overlay"><div class="spinner"></div></div>' : compareSelectPrompt('A', compareA)}</div>
        <div class="vs-divider"><div class="vs-text">VS</div></div>
        <div id="cmp-b">${ldB ? '<div class="loading-overlay"><div class="spinner"></div></div>' : compareSelectPrompt('B', compareB)}</div>
      </div>
      ${ldA && ldB ? '<div id="cmp-diff-wrap"></div>' : ''}
    </div>`;

  if (ldA) document.getElementById('cmp-a').innerHTML = await compareCardHTML(compareA, 'A');
  if (ldB) document.getElementById('cmp-b').innerHTML = await compareCardHTML(compareB, 'B');
  if (ldA && ldB) {
    const diffWrap = document.getElementById('cmp-diff-wrap');
    if (diffWrap) diffWrap.innerHTML = await compareDiffTableHTML(compareA, compareB);
  }
}

function compareSelectPrompt(label, current) {
  return `<div class="compare-select-prompt">← Select layer ${label} from the sidebar${current?` (currently: ${LAYERS[current]?.name})`:''}</div>`;
}

function resetCompare() { compareA = null; compareB = null; buildSidebar(document.getElementById('search-inp')?.value || ''); renderCompare(); }

async function compareCardHTML(id, label) {
  const ld = LAYERS[id];
  const cc = catColorOf(id);
  const info = await fetchLayerInfo(id);
  const result = await computeForward(id, defaultInput(id), {});
  const hw = getHWProfile(id, info);
  const desc = info?.description || LEARN_CONTENT[id]?.description || '';
  const eq = info?.equation || ld.short;

  return `<div class="card compare-card selected" style="border-color:${cc}40">
    <div style="font-size:20px;font-weight:800;color:${cc};margin-bottom:6px">${ld.name}</div>
    <div style="font-size:11.5px;color:var(--text-secondary);margin-bottom:10px;line-height:1.6">${desc.substring(0,150)}${desc.length>150?'...':''}</div>
    <div class="eq-box" style="font-size:12px;border-left-color:${cc};padding:10px 14px">${eq.split('\n')[0]}</div>
    <div class="grid-2" style="margin:12px 0">
      <div class="metric-card"><div class="metric-label">FLOPs</div><div class="metric-value" style="font-size:16px;color:${cc}">${fmtNum(result.flops)}</div></div>
      <div class="metric-card"><div class="metric-label">Params</div><div class="metric-value" style="font-size:16px;color:${cc}">${fmtNum(result.paramCount)}</div></div>
    </div>
    ${[['GPU',hw.gpu,'var(--green)'],['FPGA',hw.fpga,'var(--blue)'],['ASIC',hw.asic,'var(--red)']].map(([l,v,c])=>`
      <div style="margin-bottom:7px"><div style="display:flex;justify-content:space-between;font-size:11px;margin-bottom:3px"><span class="text-muted">${l}</span><span class="mono bold" style="color:${c}">${v}%</span></div><div class="bar-track"><div class="bar-fill" style="width:${v}%;background:${c}"></div></div></div>
    `).join('')}
    <div class="small text-muted" style="margin-top:8px">${hw.bottleneck} · ${hw.parallelism} · BW: ${hw.bandwidth}</div>
  </div>`;
}

async function compareDiffTableHTML(idA, idB) {
  const ldA = LAYERS[idA], ldB = LAYERS[idB];
  const infoA = await fetchLayerInfo(idA), infoB = await fetchLayerInfo(idB);
  const resA = await computeForward(idA, defaultInput(idA), {});
  const resB = await computeForward(idB, defaultInput(idB), {});
  const hwA = getHWProfile(idA, infoA), hwB = getHWProfile(idB, infoB);

  const rows = [
    ['Time Complexity', ldA.cx, ldB.cx, null],
    ['Space Complexity', ldA.sp, ldB.sp, null],
    ['FLOPs', fmtNum(resA.flops), fmtNum(resB.flops), resA.flops < resB.flops ? 'A' : (resB.flops < resA.flops ? 'B' : null)],
    ['Parameters', fmtNum(resA.paramCount), fmtNum(resB.paramCount), resA.paramCount < resB.paramCount ? 'A' : (resB.paramCount < resA.paramCount ? 'B' : null)],
    ['GPU Suitability', hwA.gpu+'%', hwB.gpu+'%', hwA.gpu > hwB.gpu ? 'A' : (hwB.gpu > hwA.gpu ? 'B' : null)],
    ['FPGA Suitability', hwA.fpga+'%', hwB.fpga+'%', hwA.fpga > hwB.fpga ? 'A' : (hwB.fpga > hwA.fpga ? 'B' : null)],
    ['ASIC Suitability', hwA.asic+'%', hwB.asic+'%', hwA.asic > hwB.asic ? 'A' : (hwB.asic > hwA.asic ? 'B' : null)],
    ['Compute Type', hwA.type, hwB.type, null],
    ['Tensor Core', hwA.tensorCore?'✓':'✗', hwB.tensorCore?'✓':'✗', null],
    ['Bottleneck', hwA.bottleneck, hwB.bottleneck, null],
  ];

  return `<div class="card" style="margin-top:16px">
    <div class="card-header"><span class="card-accent" style="background:var(--blue)"></span><span class="card-title">Detailed Comparison</span></div>
    <table class="diff-table">
      <thead><tr><th>Metric</th><th>${ldA.name}</th><th>${ldB.name}</th></tr></thead>
      <tbody>
        ${rows.map(([metric,a,b,winner])=>`
          <tr>
            <td style="color:var(--text-muted);font-weight:600">${metric}</td>
            <td class="mono ${winner==='A'?'diff-better':(winner==='B'?'diff-worse':'')}">${a}${winner==='A'?' ✓':''}</td>
            <td class="mono ${winner==='B'?'diff-better':(winner==='A'?'diff-worse':'')}">${b}${winner==='B'?' ✓':''}</td>
          </tr>
        `).join('')}
      </tbody>
    </table>
  </div>`;
}

// ══════════════════════════════════════
// ARCHITECTURE BUILDER
// ══════════════════════════════════════
function renderBuilder() {
  const content = document.getElementById('content');
  content.innerHTML = `
    <div class="layer-header">
      <div class="layer-title-row"><span class="layer-name">⊕ Architecture Builder</span></div>
      <div class="layer-desc">Click palette items to stack layers into a network. Live metrics update as you build. Try a preset to get started fast.</div>
    </div>
    <div style="padding:16px 24px;overflow:hidden">
      <div class="builder-layout">
        <div class="palette" id="palette">
          ${CATS.map(cat=>`
            <div class="palette-cat">${cat.name}</div>
            ${cat.layers.slice(0,4).map(id=>LAYERS[id]?`<div class="palette-item" onclick="addArchLayer('${id}')"><span class="li-dot" style="background:${cat.color}"></span>${LAYERS[id].name}</div>`:'').join('')}
          `).join('')}
        </div>
        <div class="canvas-area" id="canvas-area">
          <div id="arch-nodes"></div>
          <div class="arch-empty" id="arch-empty">Click palette items on the left to build your network →</div>
        </div>
        <div class="arch-stats">
          <div style="font-weight:700;font-size:14px;margin-bottom:12px">📊 Network Stats</div>
          <div id="arch-stat-list"><div class="text-muted small">Add layers to see metrics</div></div>
          <div class="divider"></div>
          <div style="display:flex;flex-direction:column;gap:7px">
            <button class="btn btn-sm" style="width:100%;justify-content:center" onclick="clearArch()">🗑 Clear All</button>
            <button class="btn btn-sm btn-primary" style="width:100%;justify-content:center" onclick="loadPreset('llm')">✨ GPT/LLaMA Block</button>
            <button class="btn btn-sm" style="width:100%;justify-content:center" onclick="loadPreset('vit')">🖼 ViT Block</button>
            <button class="btn btn-sm" style="width:100%;justify-content:center" onclick="loadPreset('resnet')">📷 ResNet Block</button>
            <button class="btn btn-sm" style="width:100%;justify-content:center" onclick="loadPreset('cnn')">🔲 Classic CNN</button>
          </div>
        </div>
      </div>
    </div>`;
  renderArchNodes();
}

function addArchLayer(id) {
  if (!LAYERS[id]) return;
  archLayers.push({ id, ...LAYERS[id] });
  renderArchNodes(); updateArchStats();
}
function removeArchLayer(i) { archLayers.splice(i,1); renderArchNodes(); updateArchStats(); }
function clearArch() { archLayers = []; renderArchNodes(); updateArchStats(); }

function renderArchNodes() {
  const wrap = document.getElementById('arch-nodes');
  const empty = document.getElementById('arch-empty');
  if (!wrap) return;
  if (!archLayers.length) { if(empty) empty.style.display=''; wrap.innerHTML=''; return; }
  if (empty) empty.style.display = 'none';
  wrap.innerHTML = archLayers.map((l,i) => {
    const cc = catColorOf(l.id);
    const hw = getHWProfile(l.id);
    return `${i>0?'<div class="arch-connector">↓</div>':''}
    <div class="arch-node">
      <span class="li-dot" style="background:${cc}"></span>
      <div style="flex:1">
        <div class="arch-node-name">${l.name}</div>
        <div class="arch-node-eq">${l.short}</div>
      </div>
      <span class="hw-badge hw-badge-blue arch-node-hw">${hw.type}</span>
      <span class="arch-node-rm" onclick="removeArchLayer(${i})">✕</span>
    </div>`;
  }).join('');
}

function updateArchStats() {
  const el = document.getElementById('arch-stat-list'); if (!el) return;
  if (!archLayers.length) { el.innerHTML = '<div class="text-muted small">Add layers to see metrics</div>'; return; }
  let totalParams = 0, totalFlops = 0, gemmCount = 0;
  archLayers.forEach(l => {
    totalParams += estimateParams(l.id, {});
    totalFlops += estimateFlops(l.id, 64, {});
    const hw = getHWProfile(l.id);
    if (hw.type.includes('GEMM')) gemmCount++;
  });
  const rows = [
    ['Layers', archLayers.length],
    ['Total Params', fmtNum(totalParams)],
    ['Est. FLOPs', fmtNum(totalFlops)],
    ['Weight Mem', fmtBytes(totalParams*4)],
    ['GEMM ops', gemmCount],
    ['Network Depth', archLayers.length],
  ];
  el.innerHTML = rows.map(([l,v])=>`<div class="stat-row"><span class="stat-label">${l}</span><span class="stat-val">${v}</span></div>`).join('');
}

function loadPreset(type) {
  const presets = {
    llm: ['tokenembed','posenc','mha','layernorm','ffn','layernorm'],
    vit: ['tokenembed','posenc','mha','layernorm','ffn','layernorm','globalavg','dense','softmaxout'],
    resnet: ['conv2d','batchnorm','relu','conv2d','batchnorm','residual','relu','maxpool'],
    cnn: ['conv2d','batchnorm','relu','maxpool','conv2d','batchnorm','relu','globalavg','dense','softmaxout'],
  };
  const ids = presets[type] || presets.llm;
  archLayers = ids.filter(id=>LAYERS[id]).map(id => ({ id, ...LAYERS[id] }));
  renderArchNodes(); updateArchStats();
}

// ══════════════════════════════════════
// FORMATTING HELPERS
// ══════════════════════════════════════
function fmtNum(n) {
  if (!n || n === 0) return '0';
  if (n >= 1e12) return (n/1e12).toFixed(2)+'T';
  if (n >= 1e9) return (n/1e9).toFixed(2)+'B';
  if (n >= 1e6) return (n/1e6).toFixed(2)+'M';
  if (n >= 1e3) return (n/1e3).toFixed(1)+'K';
  return Math.round(n).toString();
}
function fmtBytes(b) {
  if (!b || b === 0) return '0 B';
  if (b >= 1e9) return (b/1e9).toFixed(2)+' GB';
  if (b >= 1e6) return (b/1e6).toFixed(2)+' MB';
  if (b >= 1e3) return (b/1e3).toFixed(1)+' KB';
  return Math.round(b)+' B';
}

// ══════════════════════════════════════
// INIT
// ══════════════════════════════════════
(async function init() {
  buildSidebar();
  await checkAPIHealth();
  setInterval(checkAPIHealth, 30000); // re-check every 30s
})();
