// data-loader.js (UPDATED)
// - Robust CSV parser (auto-detects delimiter, skips optional header, tolerates trailing commas)
// - Normalizes pixels to [0,1], reshapes to [N,28,28,1], one-hot labels depth 10
// - FIX: tf.gather now receives an int32 Tensor for indices (avoids Uint32Array error)
// - Careful tensor disposal to prevent memory leaks

class DataLoader {
  constructor() {
    this._train = null; // { xs, ys }
    this._test  = null; // { xs, ys }
  }

  /** Load training data from a CSV file. */
  async loadTrainFromFiles(file) {
    const rows = await this._parseCsvFile(file);
    const out  = this._rowsToTensors(rows);
    this._train?.xs.dispose(); this._train?.ys.dispose();
    this._train = out;
    return out;
  }

  /** Load test data from a CSV file. */
  async loadTestFromFiles(file) {
    const rows = await this._parseCsvFile(file);
    const out  = this._rowsToTensors(rows);
    this._test?.xs.dispose(); this._test?.ys.dispose();
    this._test = out;
    return out;
  }

  /**
   * Robust CSV parser:
   * - Accepts comma, semicolon, or tab as delimiter (auto-detected).
   * - Skips an optional header row (if first token is non-numeric).
   * - Ignores blank lines and trims spaces/CR characters.
   * - Requires at least 785 tokens (label + 784 pixels); extra tokens are ignored.
   */
  _parseCsvFile(file) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onerror = () => reject(new Error('File read failed.'));
      reader.onload = () => {
        try {
          const text = typeof reader.result === 'string'
            ? reader.result
            : new TextDecoder('utf-8').decode(reader.result);

          const lines = text.split(/\r?\n/).filter(l => l.trim().length);
          if (!lines.length) throw new Error('Empty file.');

          // Auto-detect delimiter on a sample line
          const sample = lines.find(l => l.trim().length) || '';
          const count = (s, re) => (s.match(re) || []).length;
          const commas = count(sample, /,/g);
          const semis  = count(sample, /;/g);
          const tabs   = count(sample, /\t/g);
          let delim = ',';
          if (semis > commas && semis >= tabs) delim = ';';
          if (tabs  > commas && tabs  > semis) delim = '\t';

          const rows = [];
          let headerSkipped = false;

          for (let line of lines) {
            let parts = line.split(delim).map(s => s.replace(/\r/g, '').trim());
            if (parts.length && parts[parts.length - 1] === '') parts.pop(); // trailing delimiter

            if (!parts.length) continue;

            // Skip a single header row (first token non-integer)
            if (!/^-?\d+$/.test(parts[0])) {
              if (!headerSkipped) { headerSkipped = true; continue; }
              continue; // non-numeric first token on a later row → skip
            }

            if (parts.length < 785) continue;    // not enough columns
            if (parts.length > 785) parts = parts.slice(0, 785); // ignore extras

            const label = parseInt(parts[0], 10);
            if (!(label >= 0 && label <= 9)) continue;

            const px = new Array(784);
            for (let i = 0; i < 784; i++) {
              const v = parseFloat(parts[i + 1]);
              px[i] = Number.isFinite(v) ? v : 0;
            }
            rows.push({ label, pixels: px });
          }

          if (rows.length === 0) {
            throw new Error('No valid rows found. Ensure: (1) no header or first row starts with 0–9, (2) correct delimiter, (3) 785 values per row.');
          }
          resolve(rows);
        } catch (err) {
          reject(err);
        }
      };
      reader.readAsText(file);
    });
  }

  /**
   * Convert parsed rows to tensors:
   * xs: [N,28,28,1] float32 in [0,1]
   * ys: [N,10] one-hot float32
   */
  _rowsToTensors(rows) {
    return tf.tidy(() => {
      const n = rows.length;
      const buf = new Float32Array(n * 28 * 28);
      const labels = new Int32Array(n);

      let off = 0;
      for (let i = 0; i < n; i++) {
        labels[i] = rows[i].label;
        const p = rows[i].pixels;
        for (let j = 0; j < 784; j++) buf[off++] = p[j] / 255; // normalize
      }

      const xs = tf.tensor4d(buf, [n, 28, 28, 1], 'float32');
      const ys = tf.oneHot(tf.tensor1d(labels, 'int32'), 10).toFloat();
      return { xs, ys };
    });
  }

  /**
   * Split into train/val with random shuffle.
   * IMPORTANT FIX:
   * - tf.util.createShuffledIndices returns a Uint32Array.
   * - tf.gather expects indices as a Tensor (int32) or number[].
   * - We convert to an int32 Tensor to avoid the "Uint32Array" error.
   */
  splitTrainVal(xs, ys, valRatio = 0.1) {
    // No tf.tidy here because we need to return the gathered tensors; we will
    // explicitly dispose only temporary index tensors created inside.
    const n = xs.shape[0];
    const nVal = Math.max(1, Math.floor(n * valRatio));
    const shuffled = tf.util.createShuffledIndices(n); // Uint32Array

    const valIdxArr = Array.from(shuffled.slice(0, nVal));
    const trnIdxArr = Array.from(shuffled.slice(nVal));

    const valIdxT = tf.tensor1d(valIdxArr, 'int32');
    const trnIdxT = tf.tensor1d(trnIdxArr, 'int32');

    const valXs = tf.gather(xs, valIdxT);
    const valYs = tf.gather(ys, valIdxT);
    const trainXs = tf.gather(xs, trnIdxT);
    const trainYs = tf.gather(ys, trnIdxT);

    valIdxT.dispose();
    trnIdxT.dispose();

    return { trainXs, trainYs, valXs, valYs };
  }

  /**
   * Return a random batch (k) from given xs/ys.
   * Uses a plain number[] for indices which tf.gather accepts.
   */
  getRandomTestBatch(xs, ys, k = 5) {
    const n = xs.shape[0];
    const idx = new Array(k);
    for (let i = 0; i < k; i++) idx[i] = Math.floor(Math.random() * n);

    // tf.gather accepts number[] directly
    const batchXs = tf.gather(xs, idx);
    const batchYs = tf.gather(ys, idx);
    return { xs: batchXs, ys: batchYs, indices: idx };
  }

  /**
   * Draw a 28x28 grayscale tensor to a canvas with nearest-neighbor scaling.
   * Accepts [28,28], [28,28,1], or [1,28,28,1].
   */
  draw28x28ToCanvas(t, canvas, scale = 4) {
    tf.tidy(() => {
      let img = t;
      if (img.rank === 4) img = img.squeeze([0, 3]);
      if (img.rank === 3) img = img.squeeze(); // -> [28,28]

      const u8 = img.mul(255).clipByValue(0, 255).toInt().dataSync();

      const small = document.createElement('canvas');
      small.width = 28; small.height = 28;
      const ictx = small.getContext('2d');
      const id = ictx.createImageData(28, 28);
      for (let i = 0; i < 784; i++) {
        const v = u8[i], o = i * 4;
        id.data[o] = v; id.data[o+1] = v; id.data[o+2] = v; id.data[o+3] = 255;
      }
      ictx.putImageData(id, 0, 0);

      const ctx = canvas.getContext('2d');
      canvas.width = 28 * scale; canvas.height = 28 * scale;
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(small, 0, 0, canvas.width, canvas.height);
    });
  }

  /** Dispose any stored tensors to avoid memory leaks. */
  dispose() {
    if (this._train) { this._train.xs.dispose(); this._train.ys.dispose(); this._train = null; }
    if (this._test)  { this._test.xs.dispose();  this._test.ys.dispose();  this._test  = null; }
  }
}
