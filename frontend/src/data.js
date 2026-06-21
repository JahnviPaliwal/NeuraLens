// NeuraLens — Layer Data (Frontend)
const CATS = [
  { id:'act', name:'Activations', color:'#dc2626', layers:['relu','leakyrelu','prelu','elu','selu','gelu','swish','mish','sigmoid','tanh','softmax','softplus'] },
  { id:'dns', name:'Dense & Embedding', color:'#7c3aed', layers:['dense','embedding','projection'] },
  { id:'cnv', name:'Convolution', color:'#1a56db', layers:['conv1d','conv2d','conv3d','depthwise','pointwise','separable','dilated','transposed'] },
  { id:'poo', name:'Pooling', color:'#0891b2', layers:['maxpool','avgpool','globalavg','globalmaxpool','adaptivepool'] },
  { id:'nrm', name:'Normalization', color:'#16a34a', layers:['batchnorm','layernorm','instancenorm','groupnorm','rmsnorm','spectralnorm'] },
  { id:'reg', name:'Regularization', color:'#d97706', layers:['dropout','spatialdrop','dropblock','stochdepth'] },
  { id:'rec', name:'Recurrent', color:'#b45309', layers:['rnn','lstm','gru','bilstm','convlstm'] },
  { id:'atn', name:'Attention', color:'#dc2626', layers:['selfattn','crossattn','mha','causal','sparse','flashattn'] },
  { id:'tfm', name:'Transformer', color:'#7c3aed', layers:['tokenembed','posenc','rope','ffn','gatedffn','moe'] },
  { id:'res', name:'Residual & Skip', color:'#0891b2', layers:['residual','highway','denseconn'] },
  { id:'out', name:'Output Layers', color:'#16a34a', layers:['linearout','sigmoidout','softmaxout','ctc','crf'] },
  { id:'gen', name:'Generative', color:'#db2777', layers:['encoder','latent','decoder','vaesample','genblock','discblock'] },
];

// Full layer metadata (client-side)
const LAYERS = {
  relu:       { name:'ReLU',             cat:'act', short:'max(0,x)',           cx:'O(n)',    sp:'O(1)',    color:'#dc2626' },
  leakyrelu:  { name:'Leaky ReLU',       cat:'act', short:'x if x>0 else αx',   cx:'O(n)',    sp:'O(1)',    color:'#dc2626' },
  prelu:      { name:'PReLU',            cat:'act', short:'learned α per ch.',   cx:'O(n)',    sp:'O(C)',    color:'#dc2626' },
  elu:        { name:'ELU',              cat:'act', short:'α(eˣ−1) if x<0',     cx:'O(n)',    sp:'O(1)',    color:'#dc2626' },
  selu:       { name:'SELU',             cat:'act', short:'λ·ELU (self-norm)',   cx:'O(n)',    sp:'O(1)',    color:'#dc2626' },
  gelu:       { name:'GELU',             cat:'act', short:'x·Φ(x)',             cx:'O(n)',    sp:'O(1)',    color:'#dc2626' },
  swish:      { name:'Swish',            cat:'act', short:'x·σ(βx)',            cx:'O(n)',    sp:'O(1)',    color:'#dc2626' },
  mish:       { name:'Mish',             cat:'act', short:'x·tanh(softplus(x))', cx:'O(n)',   sp:'O(1)',    color:'#dc2626' },
  sigmoid:    { name:'Sigmoid',          cat:'act', short:'1/(1+e⁻ˣ)',          cx:'O(n)',    sp:'O(1)',    color:'#dc2626' },
  tanh:       { name:'Tanh',             cat:'act', short:'(eˣ−e⁻ˣ)/(eˣ+e⁻ˣ)', cx:'O(n)',   sp:'O(1)',    color:'#dc2626' },
  softmax:    { name:'Softmax',          cat:'act', short:'eˣⁱ/Σeˣʲ',          cx:'O(n)',    sp:'O(n)',    color:'#dc2626' },
  softplus:   { name:'Softplus',         cat:'act', short:'ln(1+eˣ)',           cx:'O(n)',    sp:'O(1)',    color:'#dc2626' },

  dense:      { name:'Dense',            cat:'dns', short:'Wx+b',               cx:'O(n·m)',  sp:'O(n·m)', color:'#7c3aed' },
  embedding:  { name:'Embedding',        cat:'dns', short:'lookup E[id]',        cx:'O(1)',    sp:'O(V·d)', color:'#7c3aed' },
  projection: { name:'Projection',       cat:'dns', short:'Wx (no bias)',        cx:'O(n·m)',  sp:'O(n·m)', color:'#7c3aed' },

  conv1d:     { name:'Conv1D',           cat:'cnv', short:'Σ I[i+k]·W[k]',      cx:'O(K·C·L)', sp:'O(K·C)', color:'#1a56db' },
  conv2d:     { name:'Conv2D',           cat:'cnv', short:'Σ I[i+k,j+l]·W[k,l]',cx:'O(K²·C·H·W)',sp:'O(K²·C)', color:'#1a56db' },
  conv3d:     { name:'Conv3D',           cat:'cnv', short:'3D sliding kernel',    cx:'O(K³·C·D·H·W)',sp:'O(K³·C)', color:'#1a56db' },
  depthwise:  { name:'Depthwise Conv',   cat:'cnv', short:'1 filter per channel', cx:'O(K²·C·H·W)',sp:'O(K²·C)', color:'#1a56db' },
  pointwise:  { name:'Pointwise Conv',   cat:'cnv', short:'1×1 conv',            cx:'O(Cin·Cout·H·W)',sp:'O(Cin·Cout)', color:'#1a56db' },
  separable:  { name:'Separable Conv',   cat:'cnv', short:'DW + PW',             cx:'O((K²+Cout)·C)',sp:'O(K²C)', color:'#1a56db' },
  dilated:    { name:'Dilated Conv',     cat:'cnv', short:'holes between weights', cx:'O(K²·C·H·W)',sp:'O(K²·C)', color:'#1a56db' },
  transposed: { name:'Transposed Conv',  cat:'cnv', short:'learnable upsample',   cx:'O(K²·C·H·W)',sp:'O(K²·C)', color:'#1a56db' },

  maxpool:    { name:'Max Pooling',      cat:'poo', short:'max in window',        cx:'O(K²·H·W)',sp:'O(1)',   color:'#0891b2' },
  avgpool:    { name:'Avg Pooling',      cat:'poo', short:'avg in window',        cx:'O(K²·H·W)',sp:'O(1)',   color:'#0891b2' },
  globalavg:  { name:'Global Avg Pool',  cat:'poo', short:'avg entire map',       cx:'O(H·W·C)', sp:'O(1)',   color:'#0891b2' },
  globalmaxpool:{ name:'Global Max Pool',cat:'poo', short:'max entire map',       cx:'O(H·W·C)', sp:'O(1)',   color:'#0891b2' },
  adaptivepool:{ name:'Adaptive Pool',   cat:'poo', short:'dynamic kernel/stride', cx:'O(H·W·C)',sp:'O(1)',   color:'#0891b2' },

  batchnorm:  { name:'BatchNorm',        cat:'nrm', short:'normalize over batch',  cx:'O(N·C·H·W)',sp:'O(C)', color:'#16a34a' },
  layernorm:  { name:'LayerNorm',        cat:'nrm', short:'normalize per sample',  cx:'O(d)',    sp:'O(d)',    color:'#16a34a' },
  instancenorm:{ name:'InstanceNorm',    cat:'nrm', short:'normalize per channel', cx:'O(C·H·W)',sp:'O(C)',   color:'#16a34a' },
  groupnorm:  { name:'GroupNorm',        cat:'nrm', short:'normalize in groups',   cx:'O(C·H·W)',sp:'O(C)',   color:'#16a34a' },
  rmsnorm:    { name:'RMSNorm',          cat:'nrm', short:'x/RMS(x)·γ',           cx:'O(d)',    sp:'O(d)',    color:'#16a34a' },
  spectralnorm:{ name:'SpectralNorm',    cat:'nrm', short:'W/σ₁(W)',              cx:'O(n·m)',  sp:'O(n·m)', color:'#16a34a' },

  dropout:    { name:'Dropout',          cat:'reg', short:'mask ~ Bernoulli',      cx:'O(n)',    sp:'O(n)',    color:'#d97706' },
  spatialdrop:{ name:'Spatial Dropout',  cat:'reg', short:'drop entire channels',  cx:'O(C·H·W)',sp:'O(C)',   color:'#d97706' },
  dropblock:  { name:'DropBlock',        cat:'reg', short:'drop K×K blocks',       cx:'O(H·W·C)',sp:'O(H·W)', color:'#d97706' },
  stochdepth: { name:'Stochastic Depth', cat:'reg', short:'skip entire block',     cx:'O(n)',    sp:'O(1)',    color:'#d97706' },

  rnn:        { name:'RNN',              cat:'rec', short:'h=tanh(Wh·h+Wx·x)',    cx:'O(H²+Hd)',sp:'O(H)',   color:'#b45309' },
  lstm:       { name:'LSTM',             cat:'rec', short:'4 gates + cell state',  cx:'O(4H(H+d))',sp:'O(H)', color:'#b45309' },
  gru:        { name:'GRU',              cat:'rec', short:'2 gates (update+reset)',cx:'O(3H(H+d))',sp:'O(H)', color:'#b45309' },
  bilstm:     { name:'Bi-LSTM',          cat:'rec', short:'fwd+bwd concat',        cx:'O(8H(H+d))',sp:'O(2H)', color:'#b45309' },
  convlstm:   { name:'ConvLSTM',         cat:'rec', short:'LSTM with conv gates',  cx:'O(K²H²T)',sp:'O(HHW)', color:'#b45309' },

  selfattn:   { name:'Self Attention',   cat:'atn', short:'softmax(QKᵀ/√d)V',    cx:'O(n²d)',  sp:'O(n²)',  color:'#dc2626' },
  crossattn:  { name:'Cross Attention',  cat:'atn', short:'Q_dec, K/V_enc',       cx:'O(nq·nk·d)',sp:'O(nq·nk)', color:'#dc2626' },
  mha:        { name:'Multi-Head Attn',  cat:'atn', short:'h parallel heads',     cx:'O(n²d)',  sp:'O(h·n²)', color:'#dc2626' },
  causal:     { name:'Causal Attention', cat:'atn', short:'lower-tri mask',       cx:'O(n²d/2)',sp:'O(n²/2)', color:'#dc2626' },
  sparse:     { name:'Sparse Attention', cat:'atn', short:'local+global pattern', cx:'O(n·w·d)',sp:'O(n·w)', color:'#dc2626' },
  flashattn:  { name:'Flash Attention',  cat:'atn', short:'SRAM tiling, O(n) mem',cx:'O(n²d)',  sp:'O(n)',   color:'#dc2626' },

  tokenembed: { name:'Token Embedding',  cat:'tfm', short:'E[token_id]→vec',      cx:'O(T)',    sp:'O(V·d)', color:'#7c3aed' },
  posenc:     { name:'Positional Enc.',  cat:'tfm', short:'sin/cos by position',  cx:'O(T·d)',  sp:'O(T·d)', color:'#7c3aed' },
  rope:       { name:'RoPE',             cat:'tfm', short:'rotation by position', cx:'O(T·d)',  sp:'O(d)',   color:'#7c3aed' },
  ffn:        { name:'FFN Block',        cat:'tfm', short:'Activ(xW₁)W₂',        cx:'O(T·d²)', sp:'O(d²)', color:'#7c3aed' },
  gatedffn:   { name:'Gated FFN (SwiGLU)',cat:'tfm',short:'(xW₁⊗SiLU(xWg))W₂',  cx:'O(T·d²)', sp:'O(d²)', color:'#7c3aed' },
  moe:        { name:'Mixture of Experts',cat:'tfm',short:'topK expert routing',  cx:'O(T·k/N·d²)',sp:'O(N·d²)', color:'#7c3aed' },

  residual:   { name:'Residual Conn.',   cat:'res', short:'y = F(x) + x',         cx:'O(n)',    sp:'O(n)',   color:'#0891b2' },
  highway:    { name:'Highway Network',  cat:'res', short:'T·F(x)+(1-T)·x',       cx:'O(n²)',   sp:'O(n²)', color:'#0891b2' },
  denseconn:  { name:'Dense Connection', cat:'res', short:'concat all prior maps', cx:'O(L²·k)',sp:'O(L·k)', color:'#0891b2' },

  linearout:  { name:'Linear Output',    cat:'out', short:'Wx+b (no activ.)',      cx:'O(d·out)',sp:'O(d·out)', color:'#16a34a' },
  sigmoidout: { name:'Sigmoid Output',   cat:'out', short:'σ(x) per logit',       cx:'O(out)', sp:'O(1)',   color:'#16a34a' },
  softmaxout: { name:'Softmax Output',   cat:'out', short:'prob distribution',     cx:'O(out)', sp:'O(out)', color:'#16a34a' },
  ctc:        { name:'CTC',              cat:'out', short:'alignment-free seq',    cx:'O(T·|y|·V)',sp:'O(T·|y|)', color:'#16a34a' },
  crf:        { name:'CRF',              cat:'out', short:'Viterbi decoding',      cx:'O(T·|L|²)',sp:'O(|L|²)', color:'#16a34a' },

  encoder:    { name:'Encoder',          cat:'gen', short:'x → z (compress)',      cx:'O(n→d)', sp:'O(d)',   color:'#db2777' },
  latent:     { name:'Latent Space',     cat:'gen', short:'z ~ N(μ,σ²)',           cx:'O(d)',    sp:'O(d)',   color:'#db2777' },
  decoder:    { name:'Decoder',          cat:'gen', short:'z → x̂ (expand)',       cx:'O(d→n)', sp:'O(d)',   color:'#db2777' },
  vaesample:  { name:'VAE Sampling',     cat:'gen', short:'z=μ+σ·ε, ε~N(0,I)',    cx:'O(d)',    sp:'O(d)',   color:'#db2777' },
  genblock:   { name:'Generator Block',  cat:'gen', short:'TransConv+BN+ReLU',    cx:'O(K²·C²)',sp:'O(K²C²)', color:'#db2777' },
  discblock:  { name:'Discriminator Blk',cat:'gen', short:'Conv+SpectralNorm+LReLU',cx:'O(K²·C²)',sp:'O(K²C²)', color:'#db2777' },
};

// Hardware profiles for layers not served by backend
const HW_PROFILES = {
  // Backend-covered layers (kept in sync here too, for synchronous contexts like the Builder)
  relu:        { gpu:96, fpga:93, asic:99, type:'Elementwise',    parallelism:'Perfect', bandwidth:'Minimal', tensorCore:false, bottleneck:'Memory-bound' },
  gelu:        { gpu:90, fpga:66, asic:81, type:'Elementwise',    parallelism:'High',    bandwidth:'Low',     tensorCore:false, bottleneck:'Compute-bound' },
  sigmoid:     { gpu:87, fpga:72, asic:86, type:'Elementwise',    parallelism:'Perfect', bandwidth:'Low',     tensorCore:false, bottleneck:'Compute-bound' },
  tanh:        { gpu:87, fpga:70, asic:84, type:'Elementwise',    parallelism:'Perfect', bandwidth:'Low',     tensorCore:false, bottleneck:'Compute-bound' },
  softmax:     { gpu:82, fpga:62, asic:76, type:'Reduction+EW',   parallelism:'Partial', bandwidth:'Medium',  tensorCore:false, bottleneck:'Memory-bound' },
  swish:       { gpu:89, fpga:67, asic:80, type:'Elementwise',    parallelism:'High',    bandwidth:'Low',     tensorCore:false, bottleneck:'Compute-bound' },
  dense:       { gpu:99, fpga:82, asic:92, type:'GEMM',           parallelism:'Very High',bandwidth:'High',   tensorCore:true,  bottleneck:'Compute-bound' },
  conv2d:      { gpu:99, fpga:86, asic:93, type:'GEMM (im2col)',  parallelism:'Very High',bandwidth:'High',   tensorCore:true,  bottleneck:'Compute-bound' },
  batchnorm:   { gpu:82, fpga:72, asic:86, type:'Reduction+EW',   parallelism:'Medium',  bandwidth:'High',    tensorCore:false, bottleneck:'Memory-bound' },
  layernorm:   { gpu:84, fpga:74, asic:87, type:'Reduction+EW',   parallelism:'High',    bandwidth:'Medium',  tensorCore:false, bottleneck:'Memory-bound' },
  rmsnorm:     { gpu:90, fpga:82, asic:92, type:'Reduction+EW',   parallelism:'High',    bandwidth:'Low',     tensorCore:false, bottleneck:'Compute-bound' },
  dropout:     { gpu:93, fpga:90, asic:97, type:'Elementwise',    parallelism:'Perfect', bandwidth:'Low',     tensorCore:false, bottleneck:'Memory-bound' },
  lstm:        { gpu:76, fpga:82, asic:88, type:'GEMM',           parallelism:'Sequential (step)',bandwidth:'Medium', tensorCore:false, bottleneck:'Compute-bound' },
  mha:         { gpu:99, fpga:62, asic:79, type:'GEMM',           parallelism:'Very High',bandwidth:'Very High',tensorCore:true, bottleneck:'Compute-bound' },
  ffn:         { gpu:99, fpga:83, asic:92, type:'GEMM',           parallelism:'Very High',bandwidth:'High',   tensorCore:true,  bottleneck:'Compute-bound' },
  residual:    { gpu:99, fpga:99, asic:99, type:'Elementwise Add',parallelism:'Perfect', bandwidth:'Low',     tensorCore:false, bottleneck:'Memory-bound' },
  // Layers not covered by backend (client-side fallback simulation only)
  embedding:   { gpu:72, fpga:88, asic:96, type:'Memory Lookup', parallelism:'Perfect', bandwidth:'Very High', tensorCore:false, bottleneck:'Memory-bound' },
  conv1d:      { gpu:94, fpga:86, asic:91, type:'GEMM (im2col)', parallelism:'High',     bandwidth:'Medium',   tensorCore:false, bottleneck:'Compute-bound' },
  conv3d:      { gpu:90, fpga:70, asic:82, type:'GEMM',          parallelism:'High',     bandwidth:'Very High',tensorCore:false, bottleneck:'Memory-bound' },
  depthwise:   { gpu:76, fpga:93, asic:96, type:'Elementwise Conv',parallelism:'Medium', bandwidth:'Low',      tensorCore:false, bottleneck:'Memory-bound' },
  pointwise:   { gpu:97, fpga:84, asic:93, type:'GEMM',          parallelism:'Very High',bandwidth:'High',     tensorCore:true,  bottleneck:'Compute-bound' },
  separable:   { gpu:85, fpga:88, asic:94, type:'Mixed',         parallelism:'High',     bandwidth:'Low',      tensorCore:false, bottleneck:'Mixed' },
  dilated:     { gpu:88, fpga:80, asic:88, type:'GEMM',          parallelism:'High',     bandwidth:'Medium',   tensorCore:false, bottleneck:'Compute-bound' },
  transposed:  { gpu:91, fpga:76, asic:85, type:'GEMM',          parallelism:'High',     bandwidth:'High',     tensorCore:false, bottleneck:'Compute-bound' },
  maxpool:     { gpu:90, fpga:96, asic:99, type:'Reduction',     parallelism:'Very High',bandwidth:'Minimal',  tensorCore:false, bottleneck:'Memory-bound' },
  avgpool:     { gpu:92, fpga:97, asic:99, type:'Reduction',     parallelism:'Very High',bandwidth:'Minimal',  tensorCore:false, bottleneck:'Memory-bound' },
  globalavg:   { gpu:88, fpga:97, asic:99, type:'Reduction',     parallelism:'Medium',   bandwidth:'Minimal',  tensorCore:false, bottleneck:'Memory-bound' },
  globalmaxpool:{ gpu:87,fpga:96, asic:99, type:'Reduction',     parallelism:'High',     bandwidth:'Minimal',  tensorCore:false, bottleneck:'Memory-bound' },
  adaptivepool:{ gpu:88, fpga:90, asic:95, type:'Reduction',     parallelism:'High',     bandwidth:'Low',      tensorCore:false, bottleneck:'Memory-bound' },
  instancenorm:{ gpu:80, fpga:75, asic:85, type:'Reduction+EW',  parallelism:'High',     bandwidth:'Medium',   tensorCore:false, bottleneck:'Memory-bound' },
  groupnorm:   { gpu:83, fpga:76, asic:86, type:'Reduction+EW',  parallelism:'High',     bandwidth:'Medium',   tensorCore:false, bottleneck:'Memory-bound' },
  spectralnorm:{ gpu:75, fpga:55, asic:70, type:'SVD/Power Iter',parallelism:'Medium',   bandwidth:'Low',      tensorCore:false, bottleneck:'Compute-bound' },
  spatialdrop: { gpu:92, fpga:92, asic:97, type:'Elementwise',   parallelism:'Perfect',  bandwidth:'Low',      tensorCore:false, bottleneck:'Memory-bound' },
  dropblock:   { gpu:90, fpga:88, asic:95, type:'Elementwise',   parallelism:'High',     bandwidth:'Low',      tensorCore:false, bottleneck:'Memory-bound' },
  stochdepth:  { gpu:88, fpga:85, asic:92, type:'Skip Gate',     parallelism:'High',     bandwidth:'Low',      tensorCore:false, bottleneck:'Memory-bound' },
  rnn:         { gpu:72, fpga:78, asic:84, type:'GEMM',          parallelism:'Sequential',bandwidth:'Medium',  tensorCore:false, bottleneck:'Compute-bound' },
  gru:         { gpu:78, fpga:84, asic:90, type:'GEMM',          parallelism:'Sequential',bandwidth:'Medium',  tensorCore:false, bottleneck:'Compute-bound' },
  bilstm:      { gpu:74, fpga:80, asic:86, type:'GEMM',          parallelism:'2-pass seq',bandwidth:'High',    tensorCore:false, bottleneck:'Compute-bound' },
  convlstm:    { gpu:70, fpga:78, asic:82, type:'Conv+Sequential',parallelism:'Limited', bandwidth:'Very High',tensorCore:false, bottleneck:'Mixed' },
  selfattn:    { gpu:95, fpga:65, asic:80, type:'GEMM+Softmax',  parallelism:'High',     bandwidth:'Very High',tensorCore:true,  bottleneck:'Compute-bound' },
  crossattn:   { gpu:93, fpga:62, asic:78, type:'GEMM',          parallelism:'High',     bandwidth:'Very High',tensorCore:true,  bottleneck:'Compute-bound' },
  causal:      { gpu:95, fpga:58, asic:76, type:'GEMM+masked',   parallelism:'High',     bandwidth:'Very High',tensorCore:true,  bottleneck:'Compute-bound' },
  sparse:      { gpu:72, fpga:70, asic:82, type:'Sparse GEMM',   parallelism:'Medium',   bandwidth:'Medium',   tensorCore:false, bottleneck:'Mixed' },
  flashattn:   { gpu:99, fpga:55, asic:85, type:'Tiled SRAM GEMM',parallelism:'Very High',bandwidth:'IO-opt', tensorCore:true,  bottleneck:'IO-bound (opt)' },
  tokenembed:  { gpu:74, fpga:90, asic:97, type:'Memory Lookup', parallelism:'Perfect',  bandwidth:'Very High',tensorCore:false, bottleneck:'Memory-bound' },
  posenc:      { gpu:99, fpga:99, asic:99, type:'EW (precomputed)',parallelism:'Perfect', bandwidth:'Minimal',  tensorCore:false, bottleneck:'Memory-bound' },
  rope:        { gpu:95, fpga:80, asic:90, type:'EW Rotation',   parallelism:'High',     bandwidth:'Low',      tensorCore:false, bottleneck:'Compute-bound' },
  gatedffn:    { gpu:98, fpga:80, asic:90, type:'GEMM+EW',       parallelism:'Very High',bandwidth:'High',     tensorCore:true,  bottleneck:'Compute-bound' },
  moe:         { gpu:80, fpga:50, asic:75, type:'Sparse GEMM',   parallelism:'Medium',   bandwidth:'High',     tensorCore:false, bottleneck:'Mixed' },
  highway:     { gpu:90, fpga:80, asic:88, type:'GEMM+EW',       parallelism:'High',     bandwidth:'Medium',   tensorCore:false, bottleneck:'Mixed' },
  denseconn:   { gpu:82, fpga:72, asic:80, type:'GEMM+Concat',   parallelism:'High',     bandwidth:'Very High',tensorCore:false, bottleneck:'Memory-bound' },
  linearout:   { gpu:99, fpga:82, asic:92, type:'GEMM',          parallelism:'High',     bandwidth:'Medium',   tensorCore:true,  bottleneck:'Compute-bound' },
  sigmoidout:  { gpu:92, fpga:80, asic:90, type:'Elementwise',   parallelism:'Perfect',  bandwidth:'Low',      tensorCore:false, bottleneck:'Compute-bound' },
  softmaxout:  { gpu:85, fpga:65, asic:78, type:'Reduction+EW',  parallelism:'Partial',  bandwidth:'Medium',   tensorCore:false, bottleneck:'Memory-bound' },
  ctc:         { gpu:65, fpga:60, asic:72, type:'DP Algorithm',  parallelism:'Low',      bandwidth:'Medium',   tensorCore:false, bottleneck:'Mixed' },
  crf:         { gpu:60, fpga:55, asic:68, type:'DP Viterbi',    parallelism:'Low',      bandwidth:'Low',      tensorCore:false, bottleneck:'Memory-bound' },
  encoder:     { gpu:90, fpga:78, asic:86, type:'Conv+GEMM',     parallelism:'High',     bandwidth:'High',     tensorCore:false, bottleneck:'Compute-bound' },
  latent:      { gpu:99, fpga:99, asic:99, type:'Memory Buffer', parallelism:'Perfect',  bandwidth:'Minimal',  tensorCore:false, bottleneck:'Memory-bound' },
  decoder:     { gpu:88, fpga:76, asic:84, type:'TransConv+GEMM',parallelism:'High',     bandwidth:'High',     tensorCore:false, bottleneck:'Compute-bound' },
  vaesample:   { gpu:92, fpga:75, asic:85, type:'EW+RNG',        parallelism:'Perfect',  bandwidth:'Low',      tensorCore:false, bottleneck:'Memory-bound' },
  genblock:    { gpu:86, fpga:74, asic:82, type:'TransConv+BN',  parallelism:'High',     bandwidth:'High',     tensorCore:false, bottleneck:'Compute-bound' },
  discblock:   { gpu:88, fpga:75, asic:83, type:'Conv+SpectralNorm',parallelism:'High',  bandwidth:'Medium',   tensorCore:false, bottleneck:'Compute-bound' },
  projection:  { gpu:98, fpga:80, asic:90, type:'GEMM',          parallelism:'Very High',bandwidth:'High',     tensorCore:true,  bottleneck:'Compute-bound' },
};

// Learn content for all layers not served by backend
const LEARN_CONTENT = {
  // Activations
  leakyrelu: {
    equation:'f(x) = x if x>0, else αx',
    description:'ReLU variant allowing small negative gradient (α) to prevent dead neurons.',
    uses:['GAN discriminators','Detection heads','Nets prone to dying ReLU'],
    beginner:'Like ReLU but with a tiny slope for negatives — neurons never fully die.',
    intermediate:'α typically 0.01–0.3. Gradient always nonzero. Unlike PReLU, α is fixed (not learned).',
    hardware:'FPGA: comparator + multiply for negative region. Slightly more than ReLU. Still 0 BRAMs.',
    research:'Ablations show marginal gain over ReLU in most tasks. PReLU learns per-channel α.',
    interviewQ:['What is dying ReLU and how does Leaky ReLU fix it?','How does Leaky ReLU differ from PReLU?'],
    mistakes:['Using α > 0.5 (approximates linear)','Confusing with PReLU (fixed vs learned α)'],
    hw:{ gpu:95, fpga:91, asic:98, type:'Elementwise', parallelism:'Perfect', bandwidth:'Minimal', tensorCore:false, bottleneck:'Memory-bound' }
  },
  prelu: {
    equation:'f(x) = max(0,x) + aᵢ·min(0,x)',
    description:'Like Leaky ReLU but α is learned per channel via backpropagation.',
    uses:['ResNet (He et al. 2015)','Image classification','Face recognition'],
    beginner:'Each channel learns its own negative slope during training.',
    intermediate:'α_i per channel. Init to 0.25. Adds C parameters. Can also learn per-element.',
    hardware:'Requires storing learned α per channel. Slight overhead vs Leaky ReLU.',
    research:'He 2015 shows PReLU outperforms ReLU on ImageNet. Practical gain varies.',
    interviewQ:['Why does PReLU add parameters but Leaky ReLU does not?'],
    mistakes:['Using PReLU without sufficient data (overfits easily)'],
    hw:{ gpu:94, fpga:88, asic:97, type:'Elementwise', parallelism:'Perfect', bandwidth:'Minimal', tensorCore:false, bottleneck:'Memory-bound' }
  },
  elu: {
    equation:'f(x) = x if x>0, else α(eˣ−1)',
    description:'Smooth negative region. Mean activations closer to zero reducing covariate shift.',
    uses:['Deep nets without BN','Regression','Feature extraction'],
    beginner:'Smooth version of ReLU that does not saturate to zero for negatives.',
    intermediate:'Smooth at x=0. Saturates to -α for large negatives. Reduces internal covariate shift.',
    hardware:'Requires exp() — more expensive than ReLU on FPGA. LUT or CORDIC approximation.',
    research:'Self-Normalizing Networks use SELU (scaled ELU) with LeCun init for auto-normalization.',
    interviewQ:['How does ELU reduce internal covariate shift?','When to prefer ELU over ReLU?'],
    mistakes:['Using ELU without checking saturation behavior','Large α values destabilize training'],
    hw:{ gpu:88, fpga:70, asic:82, type:'Elementwise', parallelism:'High', bandwidth:'Low', tensorCore:false, bottleneck:'Compute-bound' }
  },
  selu: {
    equation:'f(x) = λ·[x if x>0, else α(eˣ−1)]  λ≈1.0507, α≈1.6733',
    description:'Self-normalizing: activations auto-converge to μ=0, σ=1 without BatchNorm.',
    uses:['Self-Normalizing Networks','MLPs without BN','Tabular data'],
    beginner:'Scaled ELU that automatically normalizes itself — no BatchNorm needed.',
    intermediate:'Fixed constants λ, α derived to maintain normalization property. Requires LeCun init.',
    hardware:'Same cost as ELU. Multiplication by λ adds one extra op.',
    research:'Klambauer 2017. Works with LeCun normal init. Loses normalization with dropout (use AlphaDropout).',
    interviewQ:['What initialization must be used with SELU?','Why does SELU not need BatchNorm?'],
    mistakes:['Using He init with SELU (must use LeCun)','Regular dropout breaks normalization (use AlphaDropout)'],
    hw:{ gpu:87, fpga:68, asic:80, type:'Elementwise', parallelism:'High', bandwidth:'Low', tensorCore:false, bottleneck:'Compute-bound' }
  },
  mish: {
    equation:'f(x) = x·tanh(ln(1+eˣ)) = x·tanh(softplus(x))',
    description:'Self-regularized non-monotonic. Smooth everywhere. Used in YOLOv4.',
    uses:['YOLOv4 / object detection','Image classification','Benchmark tasks'],
    beginner:'Smooth, non-monotonic activation. Bounded below (~-0.31) and unbounded above.',
    intermediate:'7× FLOPs of ReLU. Non-monotonic near x=-0.31. Continuous and smooth derivatives.',
    hardware:'Expensive: requires both exp() and tanh(). FPGA needs dedicated units or LUTs.',
    research:'Misra 2019. Marginally outperforms Swish in some tasks. YOLOv4 uses Mish throughout.',
    interviewQ:['What makes Mish non-monotonic?','When does Mish outperform Swish?'],
    mistakes:['Using Mish in latency-critical inference (7x FLOPs)'],
    hw:{ gpu:86, fpga:60, asic:76, type:'Elementwise', parallelism:'High', bandwidth:'Low', tensorCore:false, bottleneck:'Compute-bound' }
  },
  softplus: {
    equation:'f(x) = ln(1 + eˣ)',
    description:'Smooth ReLU approximation. Always positive. Derivative = sigmoid(x).',
    uses:['VAE variance (always positive)','Positive output constraints','Probability modeling'],
    beginner:'A smooth version of ReLU that is always strictly positive and differentiable everywhere.',
    intermediate:'Derivative = sigmoid(x). No dying neuron. Computationally expensive for large |x|.',
    hardware:'Requires exp(). Use numerically stable: ln(1+exp(x)) → max(x, ln(1+exp(-|x|))+max(x,0)).',
    research:'Used for variance heads in VAEs. Softplus gate in some attention variants.',
    interviewQ:['What is the derivative of Softplus?','When do you use Softplus vs ReLU?'],
    mistakes:['Not using numerically stable version for large x'],
    hw:{ gpu:88, fpga:65, asic:79, type:'Elementwise', parallelism:'Perfect', bandwidth:'Low', tensorCore:false, bottleneck:'Compute-bound' }
  },
  // Embeddings
  projection: {
    equation:'y = W·x  (no bias)',
    description:'Dimensionality change without bias. For residual matching, adapter layers, and LoRA.',
    uses:['Residual dim matching','LoRA adapters','Transformer projections'],
    beginner:'A linear layer without bias used to change the size of a vector.',
    intermediate:'Bias omitted since followed by normalization. LoRA: ΔW=AB trains only low-rank A,B.',
    hardware:'Same as Dense but no bias addition. Slightly cheaper. Tensor Core applicable.',
    research:'LoRA (Hu 2021): freeze W₀, learn ΔW=AB with rank r≪d. PEFT: parameter-efficient fine-tuning.',
    interviewQ:['Why omit bias in projection layers?','What is LoRA and how does it use projections?'],
    mistakes:['Adding bias when followed immediately by LayerNorm'],
    hw:{ gpu:98, fpga:80, asic:90, type:'GEMM', parallelism:'Very High', bandwidth:'High', tensorCore:true, bottleneck:'Compute-bound' }
  },
  embedding: {
    equation:'E[token_id] → vec ∈ ℝᵈ',
    description:'Lookup table mapping discrete IDs to dense vectors. No computation — pure memory access.',
    uses:['Word/token embeddings','Categorical features','Item embeddings RecSys'],
    beginner:'A lookup table. Give it a token number and it returns a learned dense vector for that token.',
    intermediate:'O(1) forward pass — just indexing. Gradients only for used rows. Weight tying with output head saves V×d params.',
    hardware:'Pure memory lookup — bandwidth bound. Cache misses critical for large vocab. FPGA: store in DRAM, preload hot entries.',
    research:'Weight tying (Press & Wolf 2017): share embedding and output projection matrices. Reduces V×d params.',
    interviewQ:['Why is embedding lookup O(1) in FLOPs?','What is weight tying in language models?'],
    mistakes:['Not tying embedding/output weights (wastes V×d params)','Too-large embedding dim for small vocab'],
    hw:{ gpu:72, fpga:88, asic:96, type:'Memory Lookup', parallelism:'Perfect', bandwidth:'Very High', tensorCore:false, bottleneck:'Memory-bound' }
  },
};

// Helper to get category color
function getCatColor(layerId) {
  const layer = LAYERS[layerId];
  if (!layer) return '#1a56db';
  const cat = CATS.find(c => c.id === layer.cat);
  return cat ? cat.color : '#1a56db';
}

// Helper to get hardware profile (backend data or fallback)
function getHWProfile(layerId, backendData) {
  if (backendData && backendData.hw) return backendData.hw;
  return HW_PROFILES[layerId] || { gpu:80, fpga:70, asic:80, type:'Mixed', parallelism:'High', bandwidth:'Medium', tensorCore:false, bottleneck:'Mixed' };
}

// Default input vectors
function defaultInput(layerId) {
  const seeds = { relu:[-1.2,2.3,-0.5,1.8,-2.1,0.7,-0.9,3.1,-1.5,0.4], gelu:[-1,0.5,2,-0.3,1.5,-0.8,0.1,2.2], sigmoid:[-3,3,-1,1,0,-2,2,-4], tanh:[-2,2,-1,1,0,-0.5,0.5,1.5] };
  return seeds[layerId] || Array.from({length:10},(_,i)=>+(Math.sin(i*0.8+1)*2).toFixed(2));
}
