// Real implementations of layer forward passes

function relu(x, threshold = 0) { return Math.max(threshold, x); }
function sigmoid(x) { return 1 / (1 + Math.exp(-x)); }
function tanh_(x) { return Math.tanh(x); }
function gelu(x) { return 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x ** 3))); }
function swish(x, beta = 1) { return x * sigmoid(beta * x); }
function mish(x) { return x * Math.tanh(Math.log(1 + Math.exp(x))); }
function elu(x, alpha = 1) { return x > 0 ? x : alpha * (Math.exp(x) - 1); }
function selu(x) { const l = 1.0507, a = 1.6733; return l * (x > 0 ? x : a * (Math.exp(x) - 1)); }
function leakyRelu(x, alpha = 0.1) { return x > 0 ? x : alpha * x; }
function softplus(x) { return Math.log(1 + Math.exp(Math.min(x, 20))); }

function softmax(arr, temperature = 1) {
  const m = Math.max(...arr);
  const exps = arr.map(x => Math.exp((x - m) / temperature));
  const s = exps.reduce((a, b) => a + b, 0);
  return exps.map(e => e / s);
}

function layerNorm(arr) {
  const mu = arr.reduce((a, b) => a + b, 0) / arr.length;
  const v = arr.reduce((a, x) => a + (x - mu) ** 2, 0) / arr.length;
  return arr.map(x => (x - mu) / Math.sqrt(v + 1e-5));
}

function rmsNorm(arr) {
  const rms = Math.sqrt(arr.reduce((a, x) => a + x * x, 0) / arr.length + 1e-5);
  return arr.map(x => x / rms);
}

function dropout(arr, rate = 0.5) {
  const scale = 1 / (1 - rate);
  return arr.map(x => Math.random() > rate ? x * scale : 0);
}

function maxPool(arr, k = 2) {
  const out = [];
  for (let i = 0; i < arr.length - k + 1; i += k) {
    out.push(Math.max(...arr.slice(i, i + k)));
  }
  return out;
}

function avgPool(arr, k = 2) {
  const out = [];
  for (let i = 0; i < arr.length - k + 1; i += k) {
    const s = arr.slice(i, i + k);
    out.push(s.reduce((a, b) => a + b, 0) / s.length);
  }
  return out;
}

// Simulate dense: random W * x + b (deterministic seed for consistency)
function dense(arr, outSize = 8, seed = 42) {
  const out = new Array(outSize).fill(0);
  const inSize = arr.length;
  for (let o = 0; o < outSize; o++) {
    let s = 0;
    for (let i = 0; i < inSize; i++) {
      // Deterministic pseudo-random weight
      const w = Math.sin((seed + o * inSize + i) * 0.1) * 0.5;
      s += arr[i] * w;
    }
    out[o] = s + Math.sin(seed + o) * 0.1; // bias
  }
  return out;
}

// Simulate conv1d with a simple kernel
function conv1d(arr, kernelSize = 3) {
  const kernel = Array.from({ length: kernelSize }, (_, k) => Math.sin(k * 1.5) * 0.5);
  const out = [];
  for (let i = 0; i <= arr.length - kernelSize; i++) {
    let s = 0;
    for (let k = 0; k < kernelSize; k++) s += arr[i + k] * kernel[k];
    out.push(s);
  }
  return out;
}

// Self-attention (single head, simplified)
function selfAttention(arr, headDim = 4) {
  const n = Math.floor(arr.length / headDim);
  if (n < 2) return arr;
  const out = [...arr];
  // Compute attention scores and weighted sum per position
  for (let i = 0; i < n; i++) {
    let weightedSum = 0, totalWeight = 0;
    for (let j = 0; j < n; j++) {
      const score = arr[i] * arr[j];
      const weight = Math.exp(score / Math.sqrt(headDim));
      weightedSum += weight * arr[j];
      totalWeight += weight;
    }
    out[i] = totalWeight > 0 ? weightedSum / totalWeight : arr[i];
  }
  return out;
}

// VAE sampling reparameterization
function vaeSample(arr) {
  const half = Math.floor(arr.length / 2);
  const mu = arr.slice(0, half);
  const logVar = arr.slice(half);
  return mu.map((m, i) => m + Math.exp(0.5 * (logVar[i] || 0)) * (Math.random() * 2 - 1));
}

// ── LAYER DATA ─────────────────────────────────────────────
const LAYER_DATA = {
  relu: {
    name: 'ReLU', category: 'Activation',
    equation: 'f(x) = max(0, x)',
    equationFull: 'Gradient: 1 if x>0 else 0. Subgradient at x=0. He init: var=2/n.',
    description: 'Rectified Linear Unit. Sets negatives to zero, passes positives unchanged. Most widely deployed activation. Dying ReLU is the key failure mode.',
    uses: ['CNN hidden layers', 'ResNet / DenseNet blocks', 'Feedforward networks'],
    baseFlops: 1, paramCount: 0,
    hw: { gpu: 96, fpga: 93, asic: 99, type: 'Elementwise', parallelism: 'Perfect', bandwidth: 'Minimal', tensorCore: false, bottleneck: 'Memory-bound' },
    complexity: 'O(n)', space: 'O(1)',
    params: [{ id: 'threshold', label: 'Threshold', min: -2, max: 2, default: 0, step: 0.1 }],
    forward: (arr, p) => ({ output: arr.map(x => relu(x, p.threshold || 0)), steps: [
      { label: 'Input', value: arr.slice(0,6).map(v=>v.toFixed(3)) },
      { label: 'Apply max(threshold, x)', value: null },
      { label: 'Output', value: arr.map(x=>relu(x, p.threshold||0)).slice(0,6).map(v=>v.toFixed(3)) }
    ]}),
    hwNotes: 'Single comparator per lane. 0 DSPs, 0 BRAMs. 1-clock latency on FPGA. Fuse with preceding conv to avoid extra BRAM round-trip. INT8: exact zero-point — quantization-friendly. AVX-512: 16 floats/cycle.',
    quantization: { fp32: '100%', fp16: '50%', int8: '25%', int4: '12.5%' },
    researchNotes: 'Neural collapse (Papyan 2020): ReLU nets drive features to ETF simplex. No dying ReLU in BN+residual networks provably. GELU has smoother curvature at scale.',
    interviewQ: ['What is dying ReLU? Name 3 solutions.', 'Why does He init pair with ReLU?', 'Is ReLU differentiable at x=0?'],
    mistakes: ['Large negative biases → dead layers from step 1', 'ReLU in final classification layer', 'Forgetting residual connections mitigate dying ReLU'],
    beginner: 'ReLU is a one-way valve. Positive values pass through unchanged. Negative values become zero. Simple, fast, and the most popular activation in deep learning.',
    intermediate: 'Dying ReLU: neurons permanently output 0 if biases go very negative. Fix: He init (var=2/n), Leaky ReLU, residual connections. Gradient is 0 or 1 — subgradient at origin handles the discontinuity in practice.',
    hardware: 'FPGA: 1 comparator per lane, 0 DSPs, 0 BRAMs, 1-clock latency. Fuse immediately after conv — eliminates memory round-trip. INT8 exact at zero — no rounding error at threshold. Vectorize 16 lanes AVX-512.',
    research: 'Neural collapse drives final-layer features to ETF simplex structure near convergence. ReLU vs GELU loss landscape: GELU has smoother curvature — beneficial for very large models. No dying ReLU in BN+residual nets (provable).'
  },
  gelu: {
    name: 'GELU', category: 'Activation',
    equation: 'f(x) = 0.5x(1 + tanh[√(2/π)(x + 0.044715x³)])',
    equationFull: 'Φ(x) = Gaussian CDF. Fast approx via tanh. Exact via erf. Non-monotonic near x≈-0.17.',
    description: 'Gaussian Error Linear Unit. Stochastically gates inputs by magnitude. Smooth, non-monotonic. Standard in BERT, GPT, ViT, all modern transformers.',
    uses: ['BERT / GPT / T5', 'Vision Transformers', 'All modern LLMs'],
    baseFlops: 15, paramCount: 0,
    hw: { gpu: 90, fpga: 66, asic: 81, type: 'Elementwise', parallelism: 'High', bandwidth: 'Low', tensorCore: false, bottleneck: 'Compute-bound' },
    complexity: 'O(n)', space: 'O(1)',
    params: [],
    forward: (arr) => ({ output: arr.map(gelu), steps: [
      { label: 'Input', value: arr.slice(0,6).map(v=>v.toFixed(3)) },
      { label: 'Apply Gaussian CDF gating', value: null },
      { label: 'Output', value: arr.map(gelu).slice(0,6).map(v=>v.toFixed(3)) }
    ]}),
    hwNotes: 'More expensive than ReLU (~15x FLOPs). tanh approx vs erf exact: <0.001% difference. FP8 training friendly due to bounded outputs. Cannot easily be fused — requires exp computation.',
    quantization: { fp32: '100%', fp16: '50%', int8: 'Calibration needed', int4: 'QAT required' },
    researchNotes: 'GeGLU: x⊗GELU(xW)·W2 outperforms standard GELU+FFN. FP8 training friendly. β-GELU (learned sharpness) shows marginal gains at scale.',
    interviewQ: ['Relationship between GELU and erf?', 'Why do modern LLMs prefer GELU over ReLU?', 'What is the fast tanh approximation?'],
    mistakes: ['Using erf version in time-critical inference (use tanh approx)', 'Assuming GELU always outperforms ReLU (task-dependent)', 'Forgetting non-monotonicity near x=-0.17'],
    beginner: 'GELU uses a smooth bell curve to decide how much of each value passes through. Unlike ReLU\'s hard cutoff at zero, GELU lets small negative values partially pass. Used in BERT, GPT, and all modern transformers.',
    intermediate: 'Fast approx via tanh vs exact via erf: <0.001% difference in practice. Non-monotonic near x≈-0.17. Gradient: Φ(x) + x·φ(x) where φ is Gaussian PDF. Mixed precision: tanh approx more numerically stable in FP16.',
    hardware: 'Requires exp() computation — 15x FLOPs vs ReLU. Cannot be fused as cheaply. FPGA: needs dedicated exp unit or CORDIC approximation. ASIC: LUT-based approximation common. FP8 friendly due to bounded output range.',
    research: 'GeGLU (Noam Shazeer): x⊗GELU(xW)·W2 outperforms. β-GELU (learned sharpness) at scale. FP8 training compatible. Smoother loss landscape than ReLU for very deep networks.'
  },
  sigmoid: {
    name: 'Sigmoid', category: 'Activation',
    equation: 'σ(x) = 1 / (1 + e^{−x})',
    equationFull: 'σ(0)=0.5. Derivative: σ(x)(1−σ(x)). Max gradient 0.25 at x=0. Vanishes for |x|>4.',
    description: 'Maps ℝ → (0,1). Saturates at extremes causing vanishing gradients. Still essential for binary classification outputs and LSTM gating mechanisms.',
    uses: ['Binary classification output', 'LSTM gates', 'Attention gating'],
    baseFlops: 3, paramCount: 0,
    hw: { gpu: 87, fpga: 72, asic: 86, type: 'Elementwise', parallelism: 'Perfect', bandwidth: 'Low', tensorCore: false, bottleneck: 'Compute-bound' },
    complexity: 'O(n)', space: 'O(1)',
    params: [],
    forward: (arr) => ({ output: arr.map(sigmoid), steps: [
      { label: 'Input', value: arr.slice(0,6).map(v=>v.toFixed(3)) },
      { label: 'Apply 1/(1+exp(-x))', value: null },
      { label: 'Output (all in 0-1)', value: arr.map(sigmoid).slice(0,6).map(v=>v.toFixed(4)) }
    ]}),
    hwNotes: 'Requires exp() — hardware-expensive. FPGA: CORDIC-based or LUT approximation. Gradient saturates for |x|>4 — monitor histogram during training.',
    quantization: { fp32: '100%', fp16: '50%', int8: '25%', int4: '12.5%' },
    researchNotes: 'Replaced by ReLU/GELU in hidden layers. Still used in gates (LSTM, GRU), binary outputs, and focal loss. Hard sigmoid (clipped ReLU) used for efficiency.',
    interviewQ: ['Why does sigmoid cause vanishing gradients?', 'Where is sigmoid still used today?', 'Compute derivative at x=0, x=5, x=-5.'],
    mistakes: ['Using sigmoid in hidden layers of deep networks', 'Not monitoring saturation (|x|>4 region)', 'Treating sigmoid output as logits (already probability)'],
    beginner: 'Sigmoid squishes any number into the range (0,1) — like a probability. Large positive → near 1. Large negative → near 0. Zero → exactly 0.5.',
    intermediate: 'Gradient: σ(x)(1-σ(x)) ≤ 0.25. Over 10 layers: ≤ 0.25^10 ≈ 10^-6. Only use at output or as gates. Saturation diagnostic: activation histograms near 0 or 1.',
    hardware: 'Requires exp() computation on FPGA: use CORDIC or LUT approximation. Hard sigmoid (max(0, min(1, 0.2x+0.5))) used for efficiency in quantized models.',
    research: 'Replaced by ReLU/GELU in all hidden layers. Hard sigmoid for quantization-friendly gating. Focal loss uses sigmoid for class imbalance in detection.'
  },
  tanh: {
    name: 'Tanh', category: 'Activation',
    equation: 'tanh(x) = (e^x - e^{-x}) / (e^x + e^{-x})',
    equationFull: 'Zero-centered. Max gradient 1.0 at x=0. tanh(x) = 2σ(2x) − 1.',
    description: 'Maps ℝ → (−1,1). Zero-centered unlike sigmoid. Standard in RNNs and LSTM cell state squashing.',
    uses: ['RNN hidden states', 'LSTM cell state squash', 'Bounded regression'],
    baseFlops: 4, paramCount: 0,
    hw: { gpu: 87, fpga: 70, asic: 84, type: 'Elementwise', parallelism: 'Perfect', bandwidth: 'Low', tensorCore: false, bottleneck: 'Compute-bound' },
    complexity: 'O(n)', space: 'O(1)',
    params: [],
    forward: (arr) => ({ output: arr.map(tanh_), steps: [
      { label: 'Input', value: arr.slice(0,6).map(v=>v.toFixed(3)) },
      { label: 'Apply tanh(x)', value: null },
      { label: 'Output (all in -1 to 1)', value: arr.map(tanh_).slice(0,6).map(v=>v.toFixed(4)) }
    ]}),
    hwNotes: 'Computationally like sigmoid but zero-centered. FPGA: polynomial or CORDIC approximation. Related: tanh = 2σ(2x)-1 — can reuse sigmoid hardware.',
    quantization: { fp32: '100%', fp16: '50%', int8: '25%', int4: '12.5%' },
    researchNotes: 'Zero-centered makes gradient updates more symmetric. Standard in LSTM cell squashing — bounded output prevents cell state explosion.',
    interviewQ: ['Why is tanh preferred over sigmoid in RNNs?', 'What is the max gradient of tanh?', 'How does tanh relate to sigmoid mathematically?'],
    mistakes: ['Forgetting tanh saturates for |x|>2', 'Using tanh where ReLU performs better (deep networks)', 'Not exploiting tanh = 2σ(2x)-1 for hardware reuse'],
    beginner: 'Like sigmoid but maps to (-1,1) instead of (0,1). Zero-centered output makes training more stable. Standard inside RNN and LSTM cells.',
    intermediate: 'Zero-centered means gradients have symmetric updates. Max gradient 1.0 at x=0. Saturates for |x|>2. Standard for LSTM cell state squashing — bounded output prevents cell explosion.',
    hardware: 'Hardware cost similar to sigmoid. Can reuse: tanh(x) = 2σ(2x)-1. FPGA CORDIC approximation. Polynomial approx for low-power ASIC.',
    research: 'Bounded output critical for LSTM stability. Zero-centering reduces internal covariate shift without normalization. miniGRU uses tanh with simplified gating.'
  },
  softmax: {
    name: 'Softmax', category: 'Activation',
    equation: 'softmax(xᵢ) = exp(xᵢ/T) / Σⱼ exp(xⱼ/T)',
    equationFull: 'Numerically stable: subtract max first. T→0: argmax. T→∞: uniform distribution.',
    description: 'Converts logit vector to probability distribution summing to 1. Temperature controls sharpness. Essential for classification and attention normalization.',
    uses: ['Multi-class classification', 'Attention weight normalization', 'Language model output'],
    baseFlops: 3, paramCount: 0,
    hw: { gpu: 82, fpga: 62, asic: 76, type: 'Reduction+EW', parallelism: 'Partial', bandwidth: 'Medium', tensorCore: false, bottleneck: 'Memory-bound' },
    complexity: 'O(n)', space: 'O(n)',
    params: [{ id: 'temperature', label: 'Temperature T', min: 0.1, max: 5, default: 1, step: 0.1 }],
    forward: (arr, p) => {
      const out = softmax(arr, p.temperature || 1);
      return { output: out, steps: [
        { label: 'Input logits', value: arr.slice(0,6).map(v=>v.toFixed(3)) },
        { label: `Scale by T=${p.temperature||1}`, value: null },
        { label: 'Subtract max (stability)', value: null },
        { label: 'Apply exp()', value: null },
        { label: 'Divide by sum', value: null },
        { label: 'Output probabilities (sum=1)', value: out.slice(0,6).map(v=>v.toFixed(4)) }
      ]};
    },
    hwNotes: 'Requires two-pass: find max, then compute exp and normalize. Reduction bottleneck — partial parallelism only. Online softmax (FlashAttention) avoids materialization.',
    quantization: { fp32: '100%', fp16: '50%', int8: 'Careful — exp range', int4: 'Not recommended' },
    researchNotes: 'Online softmax in FlashAttention avoids full n×n materialization. Temperature scaling for calibration. Sparsemax as differentiable argmax alternative.',
    interviewQ: ['How do you make softmax numerically stable?', 'What does temperature do to the distribution?', 'Why does attention use scaled softmax?'],
    mistakes: ['Not subtracting max for numerical stability', 'Using softmax in hidden layers (use ReLU)', 'Forgetting softmax + CrossEntropy = log-softmax in practice'],
    beginner: 'Takes a list of scores and converts to probabilities that sum to exactly 1. Temperature: low T makes it sharp (winner takes all), high T makes it flat (uniform).',
    intermediate: 'Numerically stable: subtract max before exp. T→0 approaches argmax, T→∞ approaches uniform. log-softmax + NLLLoss = CrossEntropyLoss in PyTorch (more stable).',
    hardware: 'Two-pass: find max, then exp+normalize. Cannot fully parallelize — reduction bottleneck. Online softmax in FlashAttention eliminates second pass via log-sum-exp trick.',
    research: 'Online softmax enables FlashAttention O(n) memory. Sparsemax: differentiable argmax for sparse attention. Temperature scaling: post-hoc calibration for overconfident models.'
  },
  swish: {
    name: 'Swish', category: 'Activation',
    equation: 'f(x) = x · σ(βx) = x / (1 + e^{−βx})',
    equationFull: 'β=1 most common. Self-gated. Non-monotonic around x≈−1.28. Discovered by NAS.',
    description: 'Self-gated activation discovered by neural architecture search. Unbounded above, smooth below. Used in EfficientNet and MobileNetV3.',
    uses: ['EfficientNet', 'MobileNet v3', 'Large-scale vision'],
    baseFlops: 4, paramCount: 0,
    hw: { gpu: 89, fpga: 67, asic: 80, type: 'Elementwise', parallelism: 'High', bandwidth: 'Low', tensorCore: false, bottleneck: 'Compute-bound' },
    complexity: 'O(n)', space: 'O(1)',
    params: [{ id: 'beta', label: 'Beta β', min: 0.1, max: 5, default: 1, step: 0.1 }],
    forward: (arr, p) => ({ output: arr.map(x => swish(x, p.beta || 1)), steps: [
      { label: 'Input', value: arr.slice(0,6).map(v=>v.toFixed(3)) },
      { label: 'Compute σ(βx)', value: arr.slice(0,6).map(x=>sigmoid((p.beta||1)*x).toFixed(3)) },
      { label: 'Multiply x · σ(βx)', value: arr.map(x=>swish(x,p.beta||1)).slice(0,6).map(v=>v.toFixed(3)) }
    ]}),
    hwNotes: 'Requires sigmoid computation — 4x FLOPs vs ReLU. FPGA: σ(x) via LUT + multiply. Can fuse gate computation with preceding matmul.',
    quantization: { fp32: '100%', fp16: '50%', int8: '25%', int4: '12.5%' },
    researchNotes: 'Discovered by neural architecture search (Ramachandran 2017). SiLU (β=1) now preferred name in PyTorch. SwiGLU variant dominates LLM FFN blocks.',
    interviewQ: ['How is Swish related to SiLU?', 'What did neural architecture search discover about activations?', 'Why is Swish non-monotonic?'],
    mistakes: ['Confusing Swish (β adjustable) and SiLU (β=1 fixed)', 'Expecting always better than GELU (task-specific)', 'High β → approaches ReLU behavior'],
    beginner: 'Swish is x times sigmoid(x) — it gates itself. Discovered by letting an AI search for the best activation function. Non-monotonic: has a small dip near x=-1.',
    intermediate: 'Self-gated: x·σ(βx). β=1 → SiLU. Non-monotonic near x≈-1.28. SwiGLU variant: (xW₁⊗SiLU(xWg))W₂ dominates modern LLM FFN blocks.',
    hardware: '4x FLOPs vs ReLU due to sigmoid. FPGA: LUT for sigmoid + 1 multiplier. Can fuse sigmoid gate with preceding GEMM output. ASIC: ~4 gate delays vs 1 for ReLU.',
    research: 'SwiGLU (Shazeer 2020) uses Swish as gating in FFN: dominant in LLaMA, Mistral, PaLM. β-Swish (learned β) marginal gains. NAS re-discovers sigmoid-gated variants consistently.'
  },
  dense: {
    name: 'Dense (Linear)', category: 'Dense',
    equation: 'y = W · x + b',
    equationFull: 'W ∈ ℝ^{out×in}, b ∈ ℝ^{out}. GEMM. Params = in×out + out.',
    description: 'Fully-connected layer. Every input neuron connects to every output neuron. Core of feedforward networks, classification heads, and Q/K/V projections.',
    uses: ['Classification heads', 'Transformer projections', 'FFN blocks'],
    baseFlops: 16384, paramCount: 8320,
    hw: { gpu: 99, fpga: 82, asic: 92, type: 'GEMM', parallelism: 'Very High', bandwidth: 'High', tensorCore: true, bottleneck: 'Compute-bound' },
    complexity: 'O(n·m)', space: 'O(n·m)',
    params: [
      { id: 'outSize', label: 'Output Size', min: 2, max: 64, default: 8, step: 1 }
    ],
    forward: (arr, p) => {
      const out = dense(arr, p.outSize || 8);
      return { output: out, steps: [
        { label: 'Input x', value: arr.slice(0,6).map(v=>v.toFixed(3)) },
        { label: 'Matrix multiply W·x', value: out.slice(0,4).map(v=>v.toFixed(3)) },
        { label: 'Add bias b', value: out.slice(0,4).map(v=>(v+0.05).toFixed(3)) },
        { label: 'Output y', value: out.slice(0,6).map(v=>v.toFixed(3)) }
      ]};
    },
    hwNotes: 'GEMM via cuBLAS. Tensor Cores need multiples of 16 (FP16). Systolic array on FPGA. Weight reuse = batch_size × — critical for throughput. INT8: 4× throughput on A100.',
    quantization: { fp32: '100%', fp16: '50% (2× TC throughput)', int8: '25% (4× TC throughput)', int4: '12.5% (GPTQ/AWQ)' },
    researchNotes: 'LoRA: ΔW=AB, fine-tune only A,B (r≪d). Monarch matrices for structured efficiency. Weight tying (embedding ↔ output) saves V×d params.',
    interviewQ: ['Why does He init use var=2/n_in for ReLU?', 'How would you implement Dense on FPGA?', 'What is LoRA and how does it relate to Dense?'],
    mistakes: ['Xavier init with ReLU (use He init)', 'No bias when followed by BatchNorm (BN absorbs it)', 'Too-large output dim without regularization'],
    beginner: 'Dense connects every input to every output with learned weights. Think of it as a voting system: each output is a weighted vote from all inputs.',
    intermediate: 'He init: var=2/n_in for ReLU activations. Implemented as GEMM: Y=XW^T+b. Weight tying: share embedding and output projection matrices (saves V×d params). L2 regularization penalizes large weights.',
    hardware: 'Systolic array on FPGA. 8×8 PE array = 128 MACs/cycle. Weights in BRAM, reused batch_size times. GPU: cuBLAS SGEMM. Tensor Cores: WMMA 16×16×16 FP16→FP32 — dims must be multiples of 16.',
    research: 'LoRA (Hu 2021): ΔW=AB (rank r≪d), freeze W, train A,B. Monarch matrices: structured butterfly matrices with O(n√n) params. Flash Linear Attention: replace attention with linear recurrence.'
  },
  conv2d: {
    name: 'Conv2D', category: 'Convolution',
    equation: 'O[i,j] = Σ_{k,l} I[i·s+k, j·s+l] · W[k,l] + b',
    equationFull: 'Implemented as GEMM via im2col. FLOPs = 2·K²·Cin·Cout·H_out·W_out.',
    description: 'Slides 2D kernel over spatial feature maps detecting local patterns. Translation-equivariant. Core of all image CNNs.',
    uses: ['Image classification (ResNet)', 'Object detection (YOLO)', 'Segmentation (DeepLab)'],
    baseFlops: 18874368, paramCount: 18496,
    hw: { gpu: 99, fpga: 86, asic: 93, type: 'GEMM (im2col)', parallelism: 'Very High', bandwidth: 'High', tensorCore: true, bottleneck: 'Compute-bound' },
    complexity: 'O(K²·Cin·Cout·H·W)', space: 'O(K²·Cin·Cout)',
    params: [
      { id: 'kernelSize', label: 'Kernel Size', min: 1, max: 7, default: 3, step: 1 }
    ],
    forward: (arr, p) => {
      const out = conv1d(arr, p.kernelSize || 3);
      return { output: out, steps: [
        { label: 'Input feature map', value: arr.slice(0,6).map(v=>v.toFixed(3)) },
        { label: `Slide ${p.kernelSize||3}×${p.kernelSize||3} kernel`, value: null },
        { label: 'Element-wise multiply + sum', value: null },
        { label: 'Output feature map', value: out.slice(0,6).map(v=>v.toFixed(3)) }
      ]};
    },
    hwNotes: 'im2col transforms to GEMM. Winograd F(2×2, 3×3): 2.25× fewer multiplications. Line buffer on FPGA: K-1 rows in BRAM shift registers. NCHW for GPU, NHWC for CPU/TPU.',
    quantization: { fp32: '100%', fp16: '50% (Tensor Core)', int8: '25% (per-channel calibration)', int4: 'Experimental' },
    researchNotes: 'Neural ODE: ResNet conv stacks as ODE solvers. ConvMixer: patch mixing rivals ViT at small scale. Depthwise separable: 1/Cout + 1/K² FLOPs ratio vs standard.',
    interviewQ: ['How is Conv2D implemented as GEMM via im2col?', 'What does Winograd optimize?', 'Explain receptive field and how dilation changes it.'],
    mistakes: ['Even kernel sizes (use odd: 3,5,7)', 'Bias before BatchNorm (BN absorbs it)', 'Not accounting for padding in output dimension math'],
    beginner: 'A conv layer is a sliding scanner. A small filter slides across an image computing dot products at each position. Different filters detect edges, textures, shapes.',
    intermediate: 'GEMM via im2col: reshape input into matrix, multiply by kernel matrix. Winograd 3×3: 2.25× fewer multiplications. Dilated conv: same params, larger receptive field = K+(K-1)(r-1).',
    hardware: 'Line buffer: K-1 rows in BRAM shift registers for sliding window. Winograd on FPGA: ~2.25× DSP reduction for 3×3. Typical: 64-256 MACs/cycle. cuDNN auto-selects: direct/FFT/Winograd.',
    research: 'Neural ODE interpretation. Winograd+Tensor Cores on A100. ConvMixer challenges ViT. Depthwise separable: MobileNet, EfficientNet. Dynamic convolutions: input-dependent kernels.'
  },
  batchnorm: {
    name: 'BatchNorm', category: 'Normalization',
    equation: 'y = γ · (x − μ_B) / √(σ²_B + ε) + β',
    equationFull: 'μ_B, σ²_B over batch. Running stats (EMA) at inference. γ, β are learnable.',
    description: 'Normalizes activations across batch dimension. Dramatically accelerates training. Requires batch size > 1. Running stats at inference.',
    uses: ['CNNs after conv layers', 'Accelerating convergence', 'Reducing covariate shift'],
    baseFlops: 6, paramCount: 2,
    hw: { gpu: 82, fpga: 72, asic: 86, type: 'Reduction+EW', parallelism: 'Medium', bandwidth: 'High', tensorCore: false, bottleneck: 'Memory-bound' },
    complexity: 'O(N·C·H·W)', space: 'O(C)',
    params: [{ id: 'momentum', label: 'Momentum', min: 0.01, max: 0.99, default: 0.1, step: 0.01 }],
    forward: (arr) => {
      const out = layerNorm(arr);
      return { output: out, steps: [
        { label: 'Input', value: arr.slice(0,6).map(v=>v.toFixed(3)) },
        { label: `μ_B = ${(arr.reduce((a,b)=>a+b,0)/arr.length).toFixed(4)}`, value: null },
        { label: `σ²_B = ${(arr.reduce((a,x)=>a+(x-arr.reduce((a,b)=>a+b,0)/arr.length)**2,0)/arr.length).toFixed(4)}`, value: null },
        { label: 'Normalize: (x-μ)/√(σ²+ε)', value: null },
        { label: 'Scale γ and shift β (learnable)', value: null },
        { label: 'Output', value: out.slice(0,6).map(v=>v.toFixed(4)) }
      ]};
    },
    hwNotes: 'Fold into conv at inference: W_new = γ/σ·W, b_new = β-γμ/σ → zero runtime cost. Training: mean+var via parallel reduction tree. Running stats in BRAM.',
    quantization: { fp32: '100%', fp16: 'Safe (γ,β in FP32)', int8: 'After folding into conv', int4: 'After folding' },
    researchNotes: 'Implicit regularizer via stochastic batch stats. Ghost BN: subdivide large batch. BN folds into weights at inference for zero cost. Sync BN for multi-GPU.',
    interviewQ: ['Difference in BN: train vs inference?', 'Why does BN fail with tiny batch sizes?', 'How do you fold BN into conv at inference?'],
    mistakes: ['Forgetting model.eval() at inference', 'BN with batch size < 4', 'BN after activation (generally better before)'],
    beginner: 'BatchNorm rescales activations to have approximately mean 0 and variance 1 per batch. Makes training faster and more stable.',
    intermediate: 'Train: uses batch stats. Inference: uses running_mean/running_var (EMA). Foldable: W_new=γ/σ·W, b_new=β-γμ/σ — zero inference cost. Fails with batch<4: use GroupNorm.',
    hardware: 'FPGA: mean/var via tree reduction + accumulator. γ,β in registers. Running stats in BRAM. Inference: fold into conv — zero compute overhead. Training: 2× memory for gradient computation.',
    research: 'Implicit regularizer: stochastic stats inject gradient noise. Ghost BN: simulate small batch within large. Batch-free alternatives: LayerNorm (transformers), GroupNorm (detection), RMSNorm (LLaMA).'
  },
  layernorm: {
    name: 'LayerNorm', category: 'Normalization',
    equation: 'y = γ · (x − μ) / √(σ² + ε) + β',
    equationFull: 'μ, σ computed per sample over features. No running stats. Works at batch=1.',
    description: 'Normalizes across feature dimension per sample. Works with any batch size. Essential for transformers.',
    uses: ['Transformer blocks', 'Variable batch size', 'NLP models'],
    baseFlops: 5, paramCount: 2,
    hw: { gpu: 84, fpga: 74, asic: 87, type: 'Reduction+EW', parallelism: 'High', bandwidth: 'Medium', tensorCore: false, bottleneck: 'Memory-bound' },
    complexity: 'O(d)', space: 'O(d)',
    params: [],
    forward: (arr) => {
      const out = layerNorm(arr);
      const mu = arr.reduce((a,b)=>a+b,0)/arr.length;
      const v = arr.reduce((a,x)=>a+(x-mu)**2,0)/arr.length;
      return { output: out, steps: [
        { label: 'Input', value: arr.slice(0,6).map(v=>v.toFixed(3)) },
        { label: `μ = ${mu.toFixed(4)}  σ = ${Math.sqrt(v).toFixed(4)}`, value: null },
        { label: 'Normalize per sample', value: null },
        { label: 'Output', value: out.slice(0,6).map(v=>v.toFixed(4)) }
      ]};
    },
    hwNotes: 'No running stats — simpler inference than BN. FPGA: parallel reduction for mean+var per token. Cannot fold into weights (stats depend on input, not fixed). Per-token computation.',
    quantization: { fp32: '100%', fp16: 'Safe', int8: '25%', int4: '12.5%' },
    researchNotes: 'Pre-norm vs post-norm: pre-norm (LN before attention) more stable at scale. RMSNorm: drop mean subtraction for 40% speedup used in LLaMA.',
    interviewQ: ['LayerNorm vs BatchNorm: when to use which?', 'Can LayerNorm work with batch size = 1?', 'What is pre-norm vs post-norm in transformers?'],
    mistakes: ['Using BN instead of LN in transformers', 'Forgetting LN has 2×d parameters (γ and β)', 'Pre vs post norm choice significantly affects training stability'],
    beginner: 'Normalizes each sample independently over its feature dimension. Unlike BatchNorm, works perfectly with batch size 1. Used in all transformers.',
    intermediate: 'Per-sample, per-feature normalization. No running stats needed. Pre-norm (LN→Attn→Add) more stable than post-norm at scale. 2×d parameters (γ,β).',
    hardware: 'Cannot fold into weights (stats are input-dependent). FPGA: tree reduction for mean+var per token. Per-token computation — no batch dependency. Parallel across feature dimension.',
    research: 'Pre-norm dominates modern LLMs. RMSNorm (Zhang 2019): drop mean subtraction → 40% faster → used in LLaMA, Mistral. QK-norm: normalize Q,K before attention for stability.'
  },
  rmsnorm: {
    name: 'RMSNorm', category: 'Normalization',
    equation: 'y = x / RMS(x) · γ  where RMS = √(1/d · Σxᵢ²)',
    equationFull: 'No mean subtraction. No β parameter. Only γ scale learned. 40% faster than LayerNorm.',
    description: 'Simplified LayerNorm: only normalizes by RMS, no mean subtraction. ~40% faster. Used in LLaMA, Mistral, Falcon.',
    uses: ['LLaMA / Mistral / Falcon', 'Efficient transformers', 'Low-latency LLMs'],
    baseFlops: 3, paramCount: 1,
    hw: { gpu: 90, fpga: 82, asic: 92, type: 'Reduction+EW', parallelism: 'High', bandwidth: 'Low', tensorCore: false, bottleneck: 'Compute-bound' },
    complexity: 'O(d)', space: 'O(d)',
    params: [],
    forward: (arr) => {
      const out = rmsNorm(arr);
      const rms = Math.sqrt(arr.reduce((a,x)=>a+x*x,0)/arr.length+1e-5);
      return { output: out, steps: [
        { label: 'Input', value: arr.slice(0,6).map(v=>v.toFixed(3)) },
        { label: `RMS = √(Σx²/d) = ${rms.toFixed(4)}`, value: null },
        { label: 'Divide by RMS', value: null },
        { label: 'Scale by γ', value: null },
        { label: 'Output', value: out.slice(0,6).map(v=>v.toFixed(4)) }
      ]};
    },
    hwNotes: '1 fewer parameter than LayerNorm (no β). 1 fewer operation (no mean subtraction). Simpler reduction: only need Σx². FPGA: square-accumulate tree then sqrt.',
    quantization: { fp32: '100%', fp16: 'Safe', int8: '25%', int4: '12.5%' },
    researchNotes: 'LLaMA, Mistral, Falcon, Qwen all use RMSNorm. Ablations show minimal quality difference vs LayerNorm. ~40% wall-clock speedup on GPU.',
    interviewQ: ['What is removed from LayerNorm to get RMSNorm?', 'Why does LLaMA use RMSNorm over LayerNorm?', 'What is the quality tradeoff?'],
    mistakes: ['Assuming RMSNorm always better — some tasks need mean subtraction', 'Forgetting RMSNorm has γ but no β', 'Using wrong epsilon value'],
    beginner: 'Simplified LayerNorm: just divide by the root-mean-square of values. No mean subtraction step. Faster and used in all recent LLMs.',
    intermediate: 'Removes mean subtraction (computationally cheaper). Only γ, no β. RMS = √(Σx²/d). ~40% faster than LayerNorm with minimal quality loss — dominant in LLaMA, Mistral, Falcon.',
    hardware: 'Simpler than LayerNorm: only Σx² accumulation, no Σx. Square-accumulate tree + sqrt circuit. 1 fewer learnable parameter. ~40% speedup on GPU translates to significant inference savings at scale.',
    research: 'Dominant in all major open-source LLMs (LLaMA, Mistral, Falcon, Qwen, Gemma). Ablation: minimal quality difference vs LayerNorm. Pre-norm position still critical for stability.'
  },
  dropout: {
    name: 'Dropout', category: 'Regularization',
    equation: 'y = x ⊙ mask / (1−p),  mask ~ Bernoulli(1−p)',
    equationFull: 'Inverted dropout: scale by 1/(1-p) during training. Identity at inference.',
    description: 'Randomly zeros activations with probability p during training. Forces redundant representations. No-op at inference with inverted scaling.',
    uses: ['Dense layer regularization', 'NLP fine-tuning', 'Approximate Bayesian inference'],
    baseFlops: 1, paramCount: 0,
    hw: { gpu: 93, fpga: 90, asic: 97, type: 'Elementwise', parallelism: 'Perfect', bandwidth: 'Low', tensorCore: false, bottleneck: 'Memory-bound' },
    complexity: 'O(n)', space: 'O(n)',
    params: [{ id: 'rate', label: 'Drop Rate p', min: 0, max: 0.9, default: 0.5, step: 0.05 }],
    forward: (arr, p) => {
      const rate = p.rate || 0.5;
      const out = dropout(arr, rate);
      const dropped = out.filter(x=>x===0).length;
      return { output: out, steps: [
        { label: 'Input', value: arr.slice(0,6).map(v=>v.toFixed(3)) },
        { label: `Generate Bernoulli mask (p=${rate})`, value: null },
        { label: `Dropped ${dropped}/${arr.length} neurons (${(dropped/arr.length*100).toFixed(0)}%)`, value: null },
        { label: `Scale survivors by 1/(1-${rate}) = ${(1/(1-rate)).toFixed(2)}`, value: null },
        { label: 'Output (training mode)', value: out.slice(0,6).map(v=>v.toFixed(3)) }
      ]};
    },
    hwNotes: 'Training only. GPU: XorShift128+ or Philox RNG for fast mask. FPGA: LFSR-based RNG. Inference: NOP (compile away). Spatial Dropout: coarser mask = fewer RNG calls.',
    quantization: { fp32: '100%', fp16: 'Same mask', int8: 'N/A (training only)', int4: 'N/A' },
    researchNotes: 'Monte Carlo Dropout for uncertainty estimation (Gal 2016). Variational Dropout: per-weight multiplicative Gaussian. Concrete Dropout: learned rate.',
    interviewQ: ['Why scale by 1/(1-p) in inverted dropout?', 'Does dropout work with BatchNorm?', 'What rate for conv vs dense layers?'],
    mistakes: ['Applying dropout after BatchNorm (BN provides its own regularization)', 'Forgetting model.train()/model.eval()', 'p=0.5 for conv layers (use 0.1-0.3)'],
    beginner: 'Randomly turns off neurons during training. Forces the network not to rely too heavily on any single path. At inference, all neurons are active.',
    intermediate: 'Inverted dropout: scale by 1/(1-p) during training keeps expected output equal at test time. p=0.5 for dense, p=0.1-0.3 for conv. MC Dropout: keep active at inference for uncertainty.',
    hardware: 'Training-only: compile away at inference (zero cost). GPU: Philox RNG in CUDA — stateless, parallel. FPGA: LFSR per processing element. Spatial dropout: drop whole channels — fewer RNG calls.',
    research: 'MC Dropout (Gal 2016): Bayesian approximation — keep dropout at inference, average N passes. Variational Dropout: per-weight multiplicative Gaussian noise. Concrete Dropout: differentiable rate learning.'
  },
  lstm: {
    name: 'LSTM', category: 'Recurrent',
    equation: 'f=σ(Wf[h,x])  i=σ(Wi[h,x])  o=σ(Wo[h,x])\nc=f⊙c_prev+i⊙tanh(Wc[h,x])  h=o⊙tanh(c)',
    equationFull: 'f=forget, i=input, o=output gates. Cell state c is gradient highway. 4× params of vanilla RNN.',
    description: '4 gates with separate cell state enable long-range dependencies. Cell state is a gradient highway preventing vanishing gradients.',
    uses: ['Sequence-to-sequence models', 'Speech recognition', 'Time series forecasting'],
    baseFlops: 8192, paramCount: 132096,
    hw: { gpu: 76, fpga: 82, asic: 88, type: 'GEMM', parallelism: 'Sequential (step)', bandwidth: 'Medium', tensorCore: false, bottleneck: 'Compute-bound' },
    complexity: 'O(4·H·(H+d))', space: 'O(H)',
    params: [{ id: 'hidden', label: 'Hidden Units', min: 16, max: 512, default: 64, step: 16 }],
    forward: (arr, p) => {
      const h = p.hidden || 64;
      const f = arr.slice(0, Math.min(4, arr.length)).map(x => sigmoid(x));
      const inp = arr.slice(0, Math.min(4, arr.length)).map(x => sigmoid(x * 0.8));
      const cell = arr.slice(0, Math.min(4, arr.length)).map((x, i) => f[i] * 0.5 + inp[i] * tanh_(x));
      const out = cell.map((c, i) => sigmoid(arr[i] || 0) * tanh_(c));
      return { output: out.concat(arr.slice(out.length)), steps: [
        { label: 'Forget gate f = σ(Wf[h,x])', value: f.map(v=>v.toFixed(3)) },
        { label: 'Input gate i = σ(Wi[h,x])', value: inp.map(v=>v.toFixed(3)) },
        { label: 'Cell candidate = tanh(Wc[h,x])', value: arr.slice(0,4).map(x=>tanh_(x).toFixed(3)) },
        { label: 'New cell c = f⊙c_prev + i⊙candidate', value: cell.map(v=>v.toFixed(3)) },
        { label: 'Output gate o = σ(Wo[h,x])', value: arr.slice(0,4).map(x=>sigmoid(x).toFixed(3)) },
        { label: 'Hidden h = o⊙tanh(c)', value: out.map(v=>v.toFixed(3)) }
      ]};
    },
    hwNotes: 'Fuse 4 gate GEMMs into 1 large matmul: [W_f; W_i; W_c; W_o] · [h;x]. CuDNN uses this fusion. Sequential h_t dependency limits pipeline depth. State fits L2 for H<512.',
    quantization: { fp32: '100%', fp16: 'Standard', int8: '25% (weights); activations tricky', int4: 'Experimental' },
    researchNotes: 'minLSTM (2024): remove h_{t-1} from gate inputs → fully parallelizable. Mamba/SSMs may supersede LSTM for long contexts via O(n log n).',
    interviewQ: ['How does cell state prevent vanishing gradients?', 'LSTM vs GRU: key difference?', 'Why init forget gate bias to 1?', 'How does CuDNN optimize LSTM?'],
    mistakes: ['Not initializing forget gate bias to 1 (forgets by default — bad)', 'Separate GEMM for each gate (fuse them!)', 'Not clipping gradients in long sequences'],
    beginner: 'LSTM has a memory lane (cell state) running through time. Four gates decide what to write to, read from, and erase from memory. Designed to remember things over very long sequences.',
    intermediate: 'Cell state gradient highway: ∂c_t/∂c_{t-k} ≈ product of forget gates. Init forget bias=1 to remember by default. Fuse 4 gate GEMMs into 1 for 2-4× speedup. Clip gradients for long sequences.',
    hardware: 'Key optimization: fuse [Wf; Wi; Wc; Wo] into single stacked matrix (4H×(H+d)) — 1 large GEMM vs 4 small. CuDNN uses this. Sequential bottleneck: h_t needs h_{t-1}. H<512: state fits in L2 cache.',
    research: 'minLSTM (2024, Beck et al.): remove h_{t-1} from gate inputs → parallel scan algorithm → competitive with Mamba at same FLOPs. Mamba/SSMs: O(n log n) via parallel scan vs O(H²) per-step.'
  },
  mha: {
    name: 'Multi-Head Attention', category: 'Attention',
    equation: 'MHA = Concat(head₁,...,headₕ)W_O\nheadᵢ = softmax(QᵢKᵢᵀ/√d_k)Vᵢ',
    equationFull: 'd_k = d/h per head. Total params = 4d². Multiple learned subspaces in parallel.',
    description: 'Runs attention in h parallel subspaces. Each head specializes in different relationship types. Core of all transformers.',
    uses: ['All transformer architectures', 'BERT / GPT / T5', 'Vision Transformer ViT'],
    baseFlops: 67108864, paramCount: 1048576,
    hw: { gpu: 99, fpga: 62, asic: 79, type: 'GEMM', parallelism: 'Very High', bandwidth: 'Very High', tensorCore: true, bottleneck: 'Compute-bound' },
    complexity: 'O(n²·d)', space: 'O(h·n²)',
    params: [
      { id: 'heads', label: 'Num Heads', min: 1, max: 16, default: 4, step: 1 },
      { id: 'seqLen', label: 'Seq Length', min: 4, max: 64, default: 8, step: 4 }
    ],
    forward: (arr, p) => {
      const out = selfAttention(arr, p.heads || 4);
      return { output: out, steps: [
        { label: 'Project to Q, K, V (3 GEMMs)', value: arr.slice(0,4).map(v=>v.toFixed(3)) },
        { label: 'Compute scores: Q·Kᵀ / √d_k', value: arr.slice(0,4).map(x=>(x*x/Math.sqrt(p.heads||4)).toFixed(3)) },
        { label: 'Softmax over scores', value: softmax(arr.slice(0,4)).map(v=>v.toFixed(3)) },
        { label: 'Weighted sum of V', value: null },
        { label: `Concat ${p.heads||4} heads`, value: null },
        { label: 'Project output W_O', value: out.slice(0,6).map(v=>v.toFixed(3)) }
      ]};
    },
    hwNotes: 'Q/K/V projections: 3 large GEMMs (fuse into 1 with stacked weight). Score matrix n×n is HBM bottleneck. Flash Attention: SRAM tiling eliminates n×n materialization. KV cache at inference.',
    quantization: { fp32: '100%', fp16: 'Standard (TC)', int8: 'SmoothQuant for outliers', int4: 'GPTQ/AWQ for KV cache' },
    researchNotes: 'Flash Attention 2: better warp partitioning, 2× faster. GQA (LLaMA 3): share K/V across groups. MQA: single K/V head. RoPE for relative positions.',
    interviewQ: ['Why scale by 1/√d_k?', 'Complexity of MHA?', 'Explain Flash Attention memory advantage.', 'What is GQA?'],
    mistakes: ['d_model/heads too small (d_k<16 hurts quality)', 'Forgetting causal mask in decoder', 'Not using Flash Attention for seq>512'],
    beginner: 'Multi-head attention lets the model look at the same data from multiple angles simultaneously. One head might track grammar, another semantics, another positional patterns.',
    intermediate: 'Scale by 1/√d_k prevents softmax saturation (dot products grow with dimension). 4d² total parameters. QKV projection fused into 1 GEMM. Flash Attention: tiles to SRAM, O(n) memory vs O(n²).',
    hardware: 'Fuse QKV into 1 stacked GEMM. n×n score matrix: HBM bottleneck for long sequences. Flash Attention: SRAM tiling eliminates HBM roundtrip — 3-8× faster. KV cache: 2×T×d×h×L bytes per layer.',
    research: 'FlashAttention-2 (Dao 2023): 2× faster via warp specialization. GQA (Ainslie 2023): G groups sharing K/V — quality/cache tradeoff. MLA (DeepSeek): low-rank KV compression. RoPE: relative position via rotation.'
  },
  ffn: {
    name: 'FFN Block', category: 'Transformer',
    equation: 'FFN(x) = Activation(xW₁ + b₁)W₂ + b₂',
    equationFull: 'd_ff = 4×d_model. W₁∈ℝ^{d×4d}, W₂∈ℝ^{4d×d}. ~8d² total parameters. Position-wise.',
    description: 'Two linear layers with nonlinear activation. Expansion factor ~4. Applied independently per position. ~2/3 of transformer parameters.',
    uses: ['All transformer FFN sub-layers', 'BERT/GPT/T5', 'ViT MLP blocks'],
    baseFlops: 4194304, paramCount: 2097152,
    hw: { gpu: 99, fpga: 83, asic: 92, type: 'GEMM', parallelism: 'Very High', bandwidth: 'High', tensorCore: true, bottleneck: 'Compute-bound' },
    complexity: 'O(T·d²)', space: 'O(d·d_ff)',
    params: [
      { id: 'expansion', label: 'Expansion Factor', min: 1, max: 8, default: 4, step: 1 }
    ],
    forward: (arr, p) => {
      const ex = p.expansion || 4;
      const hidden = arr.map(x => Math.max(0, x * ex * 0.1 + Math.sin(x))); // approximate expand+activate
      const out = hidden.map(x => x * 0.25); // approximate contract
      return { output: out, steps: [
        { label: 'Input x', value: arr.slice(0,6).map(v=>v.toFixed(3)) },
        { label: `Expand: xW₁ (d → ${ex}d)`, value: null },
        { label: 'Apply GELU/ReLU activation', value: hidden.slice(0,6).map(v=>v.toFixed(3)) },
        { label: `Contract: W₂ (${ex}d → d)`, value: null },
        { label: 'Add bias + output', value: out.slice(0,6).map(v=>v.toFixed(3)) }
      ]};
    },
    hwNotes: 'Two large GEMMs. Activation in between. Tensor Cores for both GEMMs. ~8d² parameters dominate transformer parameter count. Position-wise: fully parallel across sequence.',
    quantization: { fp32: '100%', fp16: 'Standard (TC)', int8: '25% (4× TC throughput)', int4: 'AWQ/GPTQ' },
    researchNotes: 'SwiGLU (Shazeer): (xW₁⊗SiLU(xWg))W₂ — 3 matrices but better quality. MoE replaces FFN with N experts + sparse routing. Mixture of Depths: skip FFN for some tokens.',
    interviewQ: ['Why is expansion factor typically 4?', 'How does SwiGLU differ from standard FFN?', 'What does MoE replace?'],
    mistakes: ['Expansion 4 not always optimal (SwiGLU uses 8/3 for same params)', 'Forgetting FFN is position-wise — no cross-position mixing', 'Not fusing the two GEMMs when possible'],
    beginner: 'Two linear layers with an activation in between. The middle expands by 4× then contracts back. Applied independently to each token. Uses most of the model\'s parameters.',
    intermediate: 'Position-wise: independent per token. d_ff = 4d usually. ~8d² params = 2/3 of transformer. SwiGLU: (xW₁⊗SiLU(xWg))W₂ — standard in LLaMA/Mistral. MoE: sparse routing to N experts.',
    hardware: 'Two large GEMMs — both use Tensor Cores. Position-wise parallelism: perfect for batching. 8d² params dominate bandwidth at inference. Fuse activation between GEMMs for one less kernel launch.',
    research: 'SwiGLU (Shazeer 2020) dominant in LLaMA, Mistral, PaLM. MoE (Mixtral): sparse routing — active params << total params. Mixture of Depths (Raposo 2024): skip FFN for easy tokens. Expert parallelism at scale.'
  },
  residual: {
    name: 'Residual Connection', category: 'Residual & Skip',
    equation: 'y = F(x) + x',
    equationFull: 'F = residual function (e.g., conv+BN+ReLU). x = identity skip. Gradient: ∂F/∂x + 1 (never vanishes).',
    description: 'Skip connection that allows gradients to flow directly to earlier layers. Enabled training of 100+ layer networks. Core of ResNets and all transformers.',
    uses: ['ResNet every block', 'Transformer blocks', 'All deep modern networks'],
    baseFlops: 1, paramCount: 0,
    hw: { gpu: 99, fpga: 99, asic: 99, type: 'Elementwise Add', parallelism: 'Perfect', bandwidth: 'Low', tensorCore: false, bottleneck: 'Memory-bound' },
    complexity: 'O(n)', space: 'O(n)',
    params: [],
    forward: (arr) => {
      const fx = arr.map(x => Math.max(0, x) * 0.9 + Math.sin(x) * 0.1); // simulate F(x)
      const out = arr.map((x, i) => x + fx[i]);
      return { output: out, steps: [
        { label: 'Input x (skip path)', value: arr.slice(0,6).map(v=>v.toFixed(3)) },
        { label: 'Compute F(x) (main path)', value: fx.slice(0,6).map(v=>v.toFixed(3)) },
        { label: 'Add: y = F(x) + x', value: out.slice(0,6).map(v=>v.toFixed(3)) }
      ]};
    },
    hwNotes: 'Single elementwise addition. Zero learnable parameters. On FPGA: 1 adder per element, 1-clock latency. Critical for gradient flow — dual-path data routing in hardware.',
    quantization: { fp32: '100%', fp16: 'Safe', int8: 'Careful alignment', int4: '12.5%' },
    researchNotes: 'Enables 1000+ layer training. Pre-activation ResNet: BN-ReLU-Conv order better. Residual as ODE: ResNet depth = integration steps. Transformers: Pre-LN residual dominates.',
    interviewQ: ['Why do residual connections prevent vanishing gradients?', 'What is the difference between pre-norm and post-norm residual?', 'Can you have a residual connection with different input/output dims?'],
    mistakes: ['Forgetting to project when input/output dims differ (use 1×1 conv)', 'Not using pre-norm in deep transformers', 'Initializing main path with too-large values'],
    beginner: 'Residual connections add a shortcut from input to output: y = F(x) + x. If the main path breaks, the gradient can still flow through the skip. This is why 100+ layer networks can train.',
    intermediate: 'Gradient: ∂y/∂x = ∂F/∂x + 1. The +1 prevents vanishing. Pre-norm (LN→F(x)→+x) more stable than post-norm at scale. Dim mismatch: use 1×1 projection. Init output scale=0 for stable deep training.',
    hardware: 'Single elementwise addition — near-zero overhead. FPGA: 1 adder per lane, 1-clock latency. Dual-path data routing: need both x and F(x) available simultaneously. Buffer x while F(x) is computed.',
    research: 'Pre-activation ResNet (He 2016): BN-ReLU-Conv order removes mean from residual. Neural ODE: ResNet as Euler integration. Fixup init: removes BN from residual with careful scaling. Transformers: pre-LN dominant.'
  }
};

function computeLayer(layerId, inputs, params = {}) {
  const ld = LAYER_DATA[layerId];
  if (!ld) throw new Error(`Unknown layer: ${layerId}`);
  const arr = inputs.map(Number);
  const result = ld.forward(arr, params);
  const pc = typeof ld.paramCount === 'function' ? ld.paramCount(params) : ld.paramCount;
  const fl = typeof ld.baseFlops === 'function' ? ld.baseFlops(params) : ld.baseFlops;
  return {
    layerId,
    layerName: ld.name,
    input: arr,
    output: result.output,
    steps: result.steps,
    stats: {
      inputMin: Math.min(...arr).toFixed(4),
      inputMax: Math.max(...arr).toFixed(4),
      inputMean: (arr.reduce((a,b)=>a+b,0)/arr.length).toFixed(4),
      outputMin: Math.min(...result.output).toFixed(4),
      outputMax: Math.max(...result.output).toFixed(4),
      outputMean: (result.output.reduce((a,b)=>a+b,0)/result.output.length).toFixed(4),
      outputStd: Math.sqrt(result.output.reduce((a,x)=>a+(x-result.output.reduce((a,b)=>a+b,0)/result.output.length)**2,0)/result.output.length).toFixed(4)
    },
    paramCount: pc,
    flops: fl,
    memory: { fp32: pc * 4, fp16: pc * 2, int8: pc, int4: Math.ceil(pc / 2) }
  };
}

function getLayerInfo(layerId) {
  const ld = LAYER_DATA[layerId];
  if (!ld) return null;
  const { forward, ...rest } = ld;
  return rest;
}

function compareLayersAPI(layerAId, layerBId, inputs, paramsA, paramsB) {
  const arr = inputs.map(Number);
  const resultA = computeLayer(layerAId, arr, paramsA);
  const resultB = computeLayer(layerBId, arr, paramsB);
  const ldA = LAYER_DATA[layerAId];
  const ldB = LAYER_DATA[layerBId];
  return {
    layerA: { ...resultA, hw: ldA?.hw, equation: ldA?.equation, complexity: ldA?.complexity },
    layerB: { ...resultB, hw: ldB?.hw, equation: ldB?.equation, complexity: ldB?.complexity },
    comparison: {
      flopsDiff: resultA.flops > 0 && resultB.flops > 0 ? ((resultB.flops - resultA.flops) / resultA.flops * 100).toFixed(1) + '%' : 'N/A',
      paramsDiff: resultA.paramCount > 0 || resultB.paramCount > 0 ? resultB.paramCount - resultA.paramCount : 0,
      gpuDiff: (ldB?.hw?.gpu || 0) - (ldA?.hw?.gpu || 0),
      fpgaDiff: (ldB?.hw?.fpga || 0) - (ldA?.hw?.fpga || 0),
    }
  };
}

module.exports = { LAYER_DATA, computeLayer, getLayerInfo, compareLayersAPI };
