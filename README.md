# NeuraLens v2.0 — Deep Learning Layer Explorer

A full-stack, light-themed (white / red / blue) interactive explorer for 60+ deep learning layers — with a **real Express backend** computing actual forward passes, live Canvas visualizations, hardware suitability analysis, AI-generated explanations, a working layer comparator, and a drag-free architecture builder.

This is the **MERN-style rewrite** of the original single-file NeuraLens prototype: proper client/server separation, a real compute engine, and a complete redesign from dark multi-color theme to a clean white/red/blue light theme.

---

## What's new in v2 vs v1

| | v1 (prototype) | v2 (this version) |
|---|---|---|
| Architecture | Single 138KB HTML file | Express backend + modular frontend (5 files) |
| Computation | Simulated in JS only | **Real forward-pass math on the backend**, JS fallback if offline |
| Theme | Dark, multi-color per category | Light theme, **white / red / blue only** |
| Compare tab | Broken (selected same layer twice) | **Fixed** — proper A/B slot logic, sidebar badges, detailed diff table |
| Hardware tab | Static text | **Enhanced**: memory access patterns, FPGA notes, quantization table, optimization tips — all per-layer |
| Learn tab | 4-tier text | 4-tier text **+ AI-generated summary box** per layer |
| Visualizations | Canvas, dark colors | Canvas, redrawn for light theme, dynamic sizing (fixed clipping bugs) |
| Offline support | N/A (no backend) | **Full fallback**: if the API is down, the frontend simulates every layer client-side automatically |

---

## Architecture

```
neuralens/
├── package.json
├── backend/
│   ├── server.js              Express app, serves API + static frontend
│   ├── routes/
│   │   └── layers.js          REST endpoints (compute, compare, info, network simulate)
│   └── utils/
│       └── layerEngine.js     Real math: 16 layers with actual forward-pass implementations
└── frontend/
    ├── index.html             Shell, light theme CSS (white/red/blue design system)
    └── src/
        ├── data.js            69 layers metadata, categories, hardware profiles, learn content
        ├── api.js             Backend client + full offline JS fallback simulation
        ├── visualizer.js      Canvas drawing engine (20+ visualization types)
        └── app.js             App logic: sidebar, tabs, compare, builder, rendering
```

### Backend (`/backend`)

Real Express REST API. The compute engine in `layerEngine.js` implements actual forward-pass math for 16 core layers (ReLU, GELU, Sigmoid, Tanh, Softmax, Swish, Dense, Conv2D, BatchNorm, LayerNorm, RMSNorm, Dropout, LSTM, MHA, FFN, Residual) including:

- Real numeric output (not random/fake)
- Step-by-step computation traces (e.g. shows μ/σ for normalization, gate values for LSTM)
- FLOPs and parameter count as functions of the live slider parameters
- Per-layer hardware profile (GPU/FPGA/ASIC %, compute type, bottleneck, bandwidth)
- 4-tier learn content, interview questions, common mistakes — all per layer

**Endpoints:**
| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | Health check, used by frontend to detect backend availability |
| GET | `/api/layers` | List all backend-covered layers with summary stats |
| GET | `/api/layers/:id` | Full info for one layer (equation, hw, learn content, etc.) |
| POST | `/api/layers/compute` | Run a real forward pass: `{layerId, inputs, params}` |
| POST | `/api/layers/compare` | Compare two layers side by side |
| POST | `/api/layers/network/simulate` | Chain multiple layers, trace tensor shape/values through a network |

### Frontend (`/frontend`)

Vanilla JS, no framework, no build step. Talks to the backend via `fetch`; if the backend is unreachable, `api.js` automatically falls back to an equivalent client-side simulation (`localComputeForward`) so the app **always works**, even fully offline.

- **`data.js`** — metadata for all 69 layers (12 categories), hardware profiles for layers not covered by the backend engine, and learn-mode content (explanations, interview Qs, mistakes) for the extra layers.
- **`api.js`** — `checkAPIHealth()`, `computeForward()`, `compareViaAPI()`, `simulateNetwork()`, each with automatic local fallback.
- **`visualizer.js`** — Canvas-based rendering for every layer family: activation curves with live derivative overlay, softmax bar chart, dense neuron graph, conv sliding-window animation, pooling grid, normalization before/after bars, attention heatmaps (with causal masking), dropout active/dropped grid, positional-encoding heatmap, RNN/LSTM gate diagram, residual skip-path diagram, embedding lookup table, FFN/SwiGLU diagram, MoE expert routing, VAE latent scatter.
- **`app.js`** — sidebar (search + collapsible categories), 5 modes (Explore / Learn / Hardware / Compare / Builder), 7 tabs (Overview / Math / Playground / Visualize / Impact / Learn / Hardware), the Compare engine (proper A/B slot tracking + diff table), and the Architecture Builder (palette, live stats, presets).

---

## Design System

Strict **white / red / blue** palette — no other hues except small green/orange accents reserved for system status (API online indicator, "better" highlights in the Compare diff table).

```css
--white:    #ffffff
--off-white:#f8f9fc
--blue:     #1a56db   (primary actions, input tensors, GPU bars)
--blue-light:#dbeafe
--red:      #dc2626   (highlights, output tensors, ASIC bars, sliding windows)
--red-light:#fee2e2
--green:    #16a34a   (status only: API online, "better" in diffs)
--orange:   #d97706   (status only: bottleneck warnings, FPGA bars)
```

---

## Running it

```bash
npm install
npm run dev
# → NeuraLens v2.0 running at http://localhost:3000
```

Open `http://localhost:3000`. The frontend will show a green "API connected" dot in the topbar once it's successfully reached the backend. If you stop the server, the dot turns red and the app keeps working using the client-side fallback simulation.

No build step, no bundler, no framework — just Express + static files.

---

## Key Fixes From the Previous Iteration

1. **Compare tab was broken** (clicking a second layer overwrote both slots with the same value). Now uses dedicated `compareA` / `compareB` state, sidebar shows "A"/"B" badges, and there's a full diff table with per-metric winner highlighting.
2. **Hardware tab enhanced** with memory access pattern breakdown, FPGA-specific implementation notes, a quantization table (FP32/FP16/INT8/INT4), and concrete optimization tips — all tailored per layer instead of generic text.
3. **Learn tab gained an AI explanation box** at the top of every layer's learn content, auto-summarizing the equation, complexity, and description in plain language.
4. **Canvas clipping bug fixed** — attention heatmaps (5×5 grid) were being cut off at the bottom of a fixed 320px canvas; canvas height is now computed per layer-type.
5. **Builder sidebar highlighting bug fixed** — previously selecting a layer in Explore mode then switching to Builder left a stale highlight in the sidebar with no purpose.
6. **Builder hardware badges bug fixed** — layers covered by the backend engine (Conv2D, BatchNorm, Residual, etc.) were showing generic "Mixed" badges in the Builder because their real hardware profiles only existed server-side; they're now mirrored client-side for synchronous contexts.

---

## Tech Stack

`Express` · `Node.js` · `Vanilla JavaScript (ES2020+)` · `HTML5 Canvas API` · `CSS3 (Grid, Flexbox, Custom Properties)` · No frontend framework, no build tooling.
