// data-loader.js
/**
 * Data loading + utilities for MNIST CSV.
 * Requirements satisfied:
 * - Pure browser File APIs (no network).
 * - CSV rows: label, 784 pixel ints (0–255), no header.
 * - Normalize to [0,1], reshape to [N,28,28,1], one-hot labels depth 10.
 * - Robust parsing (handles CRLF, stray spaces, trailing commas).
 * - Memory-safe with tf.tidy and explicit dispose helpers.
 */

class DataLoader {
  constructor() {
    this._train = null; // { xs, ys }
    this._test  = null; // { xs, ys }
  }

  /** Public: parse + process train CSV -> tensors */
  async loadTrainFromFiles(file) {
    const rows = await this._parseCsvFile(file);
    const out  = this._rowsToTensors(rows);
    this._train?.xs.dispose(); this._train?.ys.dispose();
    this._train = out;
    return out;
  }

  /** Public: parse + process test CSV -> tensors */
  async loadTestFromFiles(file) {
    const rows = await this._parseCsvFile(file);
    const out  = this._rowsToTensors(rows);
    this._test?.xs.dispose(); this._test?.ys.dispose();
    this._test = out;
    return out;
  }

  /**
   * CSV parsing with resiliency.
   * - Uses FileReader.readAsText (OK for MNIST sizes).
   * - Splits by /\r?\n/.
   * - Trims and ignores blank lines.
   * - Fixes common "comma escape" issues:
   *   * trailing commas => extra empty value removed
   *   * accidental spaces => trimmed
   *   * CR chars on last token => stripped
   * - Validates 785 columns (1 + 784).
   */
  _parseCsvFile(file) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onerror = () => reject(new Error('File read failed.'));
      reader.onload  = () => {
        try {
          const text = typeof reader.result === 'string'
            ? reader.result
            : new TextDecoder('utf-8').decode(reader.result);

          const lines = text.split(/\r?\n/).filter(l => l.trim().length);
          const rows = [];
          for (let line of lines) {
            // Normalize commas/spaces and remove trailing comma if present.
            // Example issue: "5,0,0,...,0," -> trailing empty token.
            let parts = line.split(',').map(s => s.replace(/\r/g,'').trim());
            if (parts.length && parts[parts.length - 1] === '') parts.pop();

            if (parts.length !== 785) continue; // skip malformed rows safely

            const label = parseInt(parts[0], 10);
            if (!(label >= 0 && label <= 9)) continue;

            // Fast path: convert pixels in place to numbers.
            const px = new Array(784);
            for (let i = 0; i < 784; i++) {
              const v = parseFloat(parts[i + 1]);
              px[i] = Number.isFinite(v) ? v : 0;
            }
            rows.push({ label, pixels: px });
          }
          if (rows.length === 0) throw new Error('No valid rows found. Check CSV format.');
          resolve(rows);
        } catch (err) {
          reject(err);
        }
      };
      // readAsText is fine for <= a few tens of MB (MNIST CSV ~ 120MB worst).
      // If needed, switch to slice/streaming; kept simple here per instructions.
      reader.readAsText(file);
    });
  }

  /**
   * Convert parsed rows -> {xs, ys} tensors.
   * xs: [N,28,28,1] float32 in [0,1]
   * ys: [N,10] one-hot
   */
  _rowsToTensors(rows) {
    return tf.tidy(() => {
      const n = rows.length;
      const buf = new Float32Array(n * 28 * 28);
      const labels = new Int32Array(n);

      let offset = 0;
      for (let i = 0; i < n; i++) {
        labels[i] = rows[i].label;
        const p = rows[i].pixels;
        for (let j = 0; j < 784; j++) buf[offset++] = p[j] / 255; // normalize
      }

      const xs = tf.tensor4d(buf, [n, 28, 28, 1], 'float32');
      const ys = tf.oneHot(tf.tensor1d(labels, 'int32'), 10).toFloat();
      return { xs, ys };
    });
  }

  /**
   * Split tensors into train/val by ratio with random shuffle.
   * Returns fresh tensors; caller owns disposal.
   */
  splitTrainVal(xs, ys, valRatio = 0.1) {
    return tf.tidy(() => {
      const n = xs.shape[0];
      const nVal = Math.max(1, Math.floor(n * valRatio));
      const idx = tf.util.createShuffledIndices(n);

      const valIdx = idx.slice(0, nVal);
      const trnIdx = idx.slice(nVal);

      const valXs = tf.gather(xs, valIdx);
      const valYs = tf.gather(ys, valIdx);
      const trainXs = tf.gather(xs, trnIdx);
      const trainYs = tf.gather(ys, trnIdx);
      return { trainXs, trainYs, valXs, valYs };
    });
  }

  /**
   * Get random batch (k) from test set (or given xs/ys).
   * Returns tensors that the caller should dispose.
   */
  getRandomTestBatch(xs, ys, k = 5) {
    return tf.tidy(() => {
      const n = xs.shape[0];
      const idx = new Array(k);
      for (let i = 0; i < k; i++) idx[i] = Math.floor(Math.random() * n);
      return { xs: tf.gather(xs, idx), ys: tf.gather(ys, idx), indices: idx };
    });
  }

  /**
   * Draw a single 28x28 grayscale tensor into a canvas with nearest-neighbor scaling.
   * Accepts shapes [28,28], [1,28,28,1], or [28,28,1].
   */
  draw28x28ToCanvas(t, canvas, scale = 4) {
    tf.tidy(() => {
      let img = t;
      if (img.rank === 4) img = img.squeeze([0, 3]); // [1,28,28,1] -> [28,28]
      if (img.rank === 3) img = img.squeeze();       // [28,28,1] -> [28,28]

      // denormalize to 0..255
      const u8 = img.mul(255).clipByValue(0,255).toInt().dataSync();

      const small = document.createElement('canvas');
      small.width = 28; small.height = 28;
      const ictx = small.getContext('2d', { willReadFrequently: false });
      const id = ictx.createImageData(28, 28);
      for (let i = 0; i < 784; i++) {
        const v = u8[i];
        const o = i * 4;
        id.data[o] = v; id.data[o+1] = v; id.data[o+2] = v; id.data[o+3] = 255;
      }
      ictx.putImageData(id, 0, 0);

      const ctx = canvas.getContext('2d');
      canvas.width = 28 * scale; canvas.height = 28 * scale;
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(small, 0, 0, canvas.width, canvas.height);
    });
  }

  /** Dispose any stored train/test tensors to avoid leaks. */
  dispose() {
    if (this._train) { this._train.xs.dispose(); this._train.ys.dispose(); this._train = null; }
    if (this._test)  { this._test.xs.dispose();  this._test.ys.dispose();  this._test  = null; }
  }
}
