const express = require('express');
const router = express.Router();
const { computeLayer, getLayerInfo, compareLayersAPI } = require('../utils/layerEngine');

// GET /api/layers — list all layers
router.get('/', (req, res) => {
  const { LAYER_DATA } = require('../utils/layerEngine');
  const summary = Object.entries(LAYER_DATA).map(([id, l]) => ({
    id, name: l.name, category: l.category,
    params: l.paramCount, flops: l.baseFlops,
    hwGpu: l.hw.gpu, hwFpga: l.hw.fpga, hwAsic: l.hw.asic
  }));
  res.json(summary);
});

// POST /api/layers/compute — run forward pass
router.post('/compute', (req, res) => {
  try {
    const { layerId, inputs, params } = req.body;
    if (!layerId || !inputs) return res.status(400).json({ error: 'layerId and inputs required' });
    const result = computeLayer(layerId, inputs, params || {});
    res.json(result);
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

// POST /api/layers/compare — compare two layers
router.post('/compare', (req, res) => {
  try {
    const { layerA, layerB, inputs, paramsA, paramsB } = req.body;
    const result = compareLayersAPI(layerA, layerB, inputs, paramsA || {}, paramsB || {});
    res.json(result);
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

// GET /api/layers/:id — single layer full info
router.get('/:id', (req, res) => {
  try {
    const info = getLayerInfo(req.params.id);
    if (!info) return res.status(404).json({ error: 'Layer not found' });
    res.json(info);
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

// POST /api/layers/network/simulate — full network forward pass
router.post('/network/simulate', (req, res) => {
  try {
    const { layers, input } = req.body;
    const { LAYER_DATA } = require('../utils/layerEngine');
    let tensor = input || Array.from({ length: 16 }, (_, i) => Math.sin(i * 0.3));
    const trace = [];
    for (const step of layers) {
      const before = [...tensor];
      const r = computeLayer(step.id, tensor, step.params || {});
      tensor = r.output;
      trace.push({
        layerId: step.id,
        layerName: LAYER_DATA[step.id]?.name || step.id,
        inputShape: before.length,
        outputShape: tensor.length,
        inputSample: before.slice(0, 6),
        outputSample: tensor.slice(0, 6),
        flops: r.flops,
        params: r.paramCount
      });
    }
    res.json({ trace, finalOutput: tensor.slice(0, 10) });
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

module.exports = router;
