// NeuraLens — API Client
const API_BASE = '/api/layers';
let API_ONLINE = false;
let backendLayerCache = {};

async function checkAPIHealth() {
  try {
    const res = await fetch('/api/health', { signal: AbortSignal.timeout(3000) });
    const data = await res.json();
    API_ONLINE = data.status === 'ok';
  } catch (e) {
    API_ONLINE = false;
  }
  updateAPIStatusUI();
  return API_ONLINE;
}

function updateAPIStatusUI() {
  const dot = document.getElementById('api-dot');
  const label = document.getElementById('api-label');
  if (!dot || !label) return;
  if (API_ONLINE) {
    dot.className = 'api-dot online';
    label.textContent = 'API connected';
  } else {
    dot.className = 'api-dot error';
    label.textContent = 'offline mode';
  }
}

// Fetch full layer info (cached)
async function fetchLayerInfo(layerId) {
  if (backendLayerCache[layerId]) return backendLayerCache[layerId];
  if (!API_ONLINE) return null;
  try {
    const res = await fetch(`${API_BASE}/${layerId}`);
    if (!res.ok) return null;
    const data = await res.json();
    backendLayerCache[layerId] = data;
    return data;
  } catch (e) {
    return null;
  }
}

// Compute forward pass via backend, fallback to client-side simulation
async function computeForward(layerId, inputs, params = {}) {
  if (API_ONLINE) {
    try {
      const res = await fetch(`${API_BASE}/compute`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ layerId, inputs, params })
      });
      if (res.ok) return await res.json();
    } catch (e) { /* fall through to local sim */ }
  }
  return localComputeForward(layerId, inputs, params);
}

// Compare two layers via backend
async function compareViaAPI(layerA, layerB, inputs, paramsA = {}, paramsB = {}) {
  if (API_ONLINE) {
    try {
      const res = await fetch(`${API_BASE}/compare`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ layerA, layerB, inputs, paramsA, paramsB })
      });
      if (res.ok) return await res.json();
    } catch (e) { /* fall through */ }
  }
  // Local fallback
  const a = await localComputeForward(layerA, inputs, paramsA);
  const b = await localComputeForward(layerB, inputs, paramsB);
  return {
    layerA: a, layerB: b,
    comparison: {
      flopsDiff: a.flops>0 ? (((b.flops-a.flops)/a.flops)*100).toFixed(1)+'%' : 'N/A',
      paramsDiff: b.paramCount - a.paramCount,
      gpuDiff: 0, fpgaDiff: 0
    }
  };
}

// Simulate network forward pass through multiple layers
async function simulateNetwork(layers, input) {
  if (API_ONLINE) {
    try {
      const res = await fetch(`${API_BASE}/network/simulate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ layers, input })
      });
      if (res.ok) return await res.json();
    } catch (e) { /* fall through */ }
  }
  // Local fallback
  let tensor = input || Array.from({length:16},(_,i)=>Math.sin(i*0.3));
  const trace = [];
  for (const step of layers) {
    const before = [...tensor];
    const r = await localComputeForward(step.id, tensor, step.params || {});
    tensor = r.output;
    trace.push({
      layerId: step.id, layerName: LAYERS[step.id]?.name || step.id,
      inputShape: before.length, outputShape: tensor.length,
      inputSample: before.slice(0,6), outputSample: tensor.slice(0,6),
      flops: r.flops, params: r.paramCount
    });
  }
  return { trace, finalOutput: tensor.slice(0,10) };
}

// ── LOCAL FALLBACK SIMULATION ─────────────────────────────
function localComputeForward(layerId, inputs, params = {}) {
  const arr = inputs.map(Number);
  let output, steps = [];
  const sigmoid = x => 1/(1+Math.exp(-x));

  switch(layerId) {
    case 'relu': output = arr.map(x=>Math.max(params.threshold||0, x)); break;
    case 'leakyrelu': output = arr.map(x=>x>0?x:(params.alpha||0.1)*x); break;
    case 'prelu': output = arr.map(x=>x>0?x:(params.ia||0.25)*x); break;
    case 'elu': output = arr.map(x=>x>0?x:(params.alpha||1)*(Math.exp(x)-1)); break;
    case 'selu': output = arr.map(x=>1.0507*(x>0?x:1.6733*(Math.exp(x)-1))); break;
    case 'gelu': output = arr.map(x=>0.5*x*(1+Math.tanh(Math.sqrt(2/Math.PI)*(x+0.044715*x**3)))); break;
    case 'swish': output = arr.map(x=>x*sigmoid((params.beta||1)*x)); break;
    case 'mish': output = arr.map(x=>x*Math.tanh(Math.log(1+Math.exp(x)))); break;
    case 'sigmoid': output = arr.map(sigmoid); break;
    case 'tanh': output = arr.map(Math.tanh); break;
    case 'softplus': output = arr.map(x=>Math.log(1+Math.exp(Math.min(x,20)))); break;
    case 'softmax': case 'softmaxout': {
      const T = params.temperature || params.T || 1;
      const m = Math.max(...arr); const e = arr.map(x=>Math.exp((x-m)/T));
      const s = e.reduce((a,b)=>a+b,0); output = e.map(x=>x/s); break;
    }
    case 'sigmoidout': output = arr.map(sigmoid); break;
    case 'dropout': {
      const rate = params.rate || params.p || 0.5;
      output = arr.map(x=>Math.random()>rate?x/(1-rate):0); break;
    }
    case 'spatialdrop': case 'dropblock': {
      const rate = params.p || params.kp ? (1-(params.kp||0.9)) : 0.2;
      output = arr.map(x=>Math.random()>rate?x:0); break;
    }
    case 'batchnorm': case 'layernorm': case 'instancenorm': case 'groupnorm': {
      const mu = arr.reduce((a,b)=>a+b,0)/arr.length;
      const v = arr.reduce((a,x)=>a+(x-mu)**2,0)/arr.length;
      output = arr.map(x=>(x-mu)/Math.sqrt(v+1e-5)); break;
    }
    case 'rmsnorm': {
      const rms = Math.sqrt(arr.reduce((a,x)=>a+x*x,0)/arr.length+1e-5);
      output = arr.map(x=>x/rms); break;
    }
    case 'maxpool': {
      const k = params.k || params.kernelSize || 2; output = [];
      for(let i=0;i<arr.length-k+1;i+=k) output.push(Math.max(...arr.slice(i,i+k)));
      break;
    }
    case 'avgpool': {
      const k = params.k || params.kernelSize || 2; output = [];
      for(let i=0;i<arr.length-k+1;i+=k){const s=arr.slice(i,i+k);output.push(s.reduce((a,b)=>a+b,0)/s.length);}
      break;
    }
    case 'globalavg': output = [arr.reduce((a,b)=>a+b,0)/arr.length]; break;
    case 'globalmaxpool': output = [Math.max(...arr)]; break;
    case 'residual': {
      const fx = arr.map(x=>Math.max(0,x)*0.9+Math.sin(x)*0.1);
      output = arr.map((x,i)=>x+fx[i]); break;
    }
    case 'dense': case 'linearout': case 'projection': {
      const outSize = params.outSize || params.out || 8;
      output = Array.from({length:outSize},(_, o) => {
        let s=0; for(let i=0;i<arr.length;i++) s+=arr[i]*Math.sin((42+o*arr.length+i)*0.1)*0.5;
        return s + Math.sin(42+o)*0.1;
      });
      break;
    }
    default: output = arr.map(x=>x*0.9+Math.sin(x)*0.1);
  }

  const paramCount = estimateParams(layerId, params);
  const flops = estimateFlops(layerId, arr.length, params);

  return {
    layerId, layerName: LAYERS[layerId]?.name || layerId,
    input: arr, output, steps,
    stats: {
      inputMin: Math.min(...arr).toFixed(4), inputMax: Math.max(...arr).toFixed(4),
      inputMean: (arr.reduce((a,b)=>a+b,0)/arr.length).toFixed(4),
      outputMin: Math.min(...output).toFixed(4), outputMax: Math.max(...output).toFixed(4),
      outputMean: (output.reduce((a,b)=>a+b,0)/output.length).toFixed(4),
      outputStd: Math.sqrt(output.reduce((a,x)=>a+(x-output.reduce((a,b)=>a+b,0)/output.length)**2,0)/output.length).toFixed(4)
    },
    paramCount, flops,
    memory: { fp32: paramCount*4, fp16: paramCount*2, int8: paramCount, int4: Math.ceil(paramCount/2) }
  };
}

function estimateParams(layerId, params) {
  const map = {
    dense: (params.outSize||8) * 10 + (params.outSize||8),
    lstm: 4 * ((params.hidden||64)+10) * (params.hidden||64),
    mha: 4 * ((params.seqLen||8)*4) ** 2 / 4,
    ffn: 2 * 64 * 64 * (params.expansion||4),
    embedding: (params.vocab||1000) * (params.dim||64),
  };
  return Math.round(map[layerId] || 0);
}

function estimateFlops(layerId, n, params) {
  const map = {
    relu: n, leakyrelu: n, gelu: n*15, swish: n*4, sigmoid: n*3, tanh: n*4, softmax: n*3,
    dense: 2*n*(params.outSize||8), conv2d: n*n*9*32, mha: n*n*64,
    batchnorm: n*6, layernorm: n*5, dropout: n,
  };
  return Math.round(map[layerId] || n);
}

// Toast notifications
function showToast(message, type = 'info') {
  const container = document.getElementById('toast-container');
  if (!container) return;
  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  toast.textContent = message;
  container.appendChild(toast);
  setTimeout(() => toast.remove(), 3000);
}
