const express = require('express');
const cors = require('cors');
const path = require('path');
const layerRoutes = require('./routes/layers');

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname, '../frontend')));

// API routes
app.use('/api/layers', layerRoutes);

// Health check
app.get('/api/health', (req, res) => {
  res.json({ status: 'ok', version: '2.0.0', timestamp: new Date().toISOString() });
});

// Serve frontend
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, '../frontend/index.html'));
});

// Only call .listen() when run directly (local dev, Hugging Face Spaces, Render, etc.)
// Vercel imports `app` and wraps it as a serverless function instead.
if (require.main === module) {
  app.listen(PORT, '0.0.0.0', () => {
    console.log(`\n  NeuraLens v2.0 running at http://localhost:${PORT}\n`);
  });
}

module.exports = app;

