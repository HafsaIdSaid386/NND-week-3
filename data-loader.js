// data-loader.js (إصـلاح: parser متسامح مع header و delimiters مختلفة)
// كيدعم: CSV بلا شبكة، label + 784 pixels، normalizing، reshape [N,28,28,1]، one-hot labels.

class DataLoader {
  constructor() {
    this._train = null; // { xs, ys }
    this._test  = null; // { xs, ys }
  }

  // تحميل Train من ملف
  async loadTrainFromFiles(file) {
    const rows = await this._parseCsvFile(file);
    const out  = this._rowsToTensors(rows);
    this._train?.xs.dispose(); this._train?.ys.dispose();
    this._train = out;
    return out;
  }

  // تحميل Test من ملف
  async loadTestFromFiles(file) {
    const rows = await this._parseCsvFile(file);
    const out  = this._rowsToTensors(rows);
    this._test?.xs.dispose(); this._test?.ys.dispose();
    this._test = out;
    return out;
  }

  /**
   * Parser متسامح:
   * - كيتعرف تلقائياً على الـ delimiter: ',' أو ';' أو tab
   * - كيسكيپّي header إلا لقا أول قيمة ماشي رقم
   * - إلى كانت السطر فيه أكثر من 785 قيمة كياخد غير الأولين (label + 784)
   * - كيحيد القيم الفارغة فالأخير (trailing delimiter)
   * - كيسجّل إحصائيات مفيدة فحالة ما لقى حتى سطر صالح
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

          // نختارو سطر عينة باش نكتاشفو delimiter
          const sample = lines.find(l => l.trim().length) || '';
          const cntComma = (sample.match(/,/g)  || []).length;
          const cntSemi  = (sample.match(/;/g)  || []).length;
          const cntTab   = (sample.match(/\t/g) || []).length;
          let delim = ',';
          if (cntSemi > cntComma && cntSemi >= cntTab) delim = ';';
          if (cntTab  > cntComma && cntTab  >  cntSemi) delim = '\t';

          const rows = [];
          let skippedHeader = false;
          let badLen = 0, badLabel = 0;

          for (let idx = 0; idx < lines.length; idx++) {
            let parts = lines[idx].split(delim).map(s => s.replace(/\r/g,'').trim());
            // إزالة قيمة فارغة فالأخير إلا كانت فاصلة/سيمي كولون زائدة
            if (parts.length && parts[parts.length - 1] === '') parts.pop();
            if (!parts.length) continue;

            // سكيپّي header: إلا أول قيمة ماشي رقم صحيح
            if (!/^-?\d+$/.test(parts[0])) {
              if (!skippedHeader) { skippedHeader = true; continue; }
              badLabel++;
              continue;
            }

            // خاص 785 قيمة على الأقل
            if (parts.length < 785) { badLen++; continue; }
            if (parts.length > 785) parts = parts.slice(0, 785);

            const label = parseInt(parts[0], 10);
            if (!(label >= 0 && label <= 9)) { badLabel++; continue; }

            const px = new Array(784);
            for (let i = 0; i < 784; i++) {
              const v = parseFloat(parts[i + 1]);
              px[i] = Number.isFinite(v) ? v : 0;
            }
            rows.push({ label, pixels: px });
          }

          if (rows.length === 0) {
            const hint = `No valid rows. headerSkipped:${skippedHeader}, badLen:${badLen}, badLabel:${badLabel}. `
              + `تأكّد: header محيّد، delimiter هو "${delim}", وكل سطر فيه 785 قيمة.`;
            throw new Error(hint);
          }
          resolve(rows);
        } catch (err) {
          reject(err);
        }
      };
      // MNIST CSV كيخدم مزيان بـ readAsText. إلى كان الملف ضخم بزاف ممكن نديرو streaming مستقبلاً.
      reader.readAsText(file);
    });
  }

  // تحويل rows -> Tensors: xs [N,28,28,1] normalized, ys one-hot [N,10]
  _rowsToTensors(rows) {
    return tf.tidy(() => {
      const n = rows.length;
      const buf = new Float32Array(n * 28 * 28);
      const labels = new Int32Array(n);

      let off = 0;
      for (let i = 0; i < n; i++) {
        labels[i] = rows[i].label;
        const p = rows[i].pixels;
        for (let j = 0; j < 784; j++) buf[off++] = p[j] / 255;
      }
      const xs = tf.tensor4d(buf, [n, 28, 28, 1], 'float32');
      const ys = tf.oneHot(tf.tensor1d(labels, 'int32'), 10).toFloat();
      return { xs, ys };
    });
  }

  // تقسيم Train/Val بنسبة valRatio (افتراضياً 0.1)
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

  // Batch عشوائي من test (افتراضياً 5) لعرض ال-preview
  getRandomTestBatch(xs, ys, k = 5) {
    return tf.tidy(() => {
      const n = xs.shape[0];
      const idx = new Array(k);
      for (let i = 0; i < k; i++) idx[i] = Math.floor(Math.random() * n);
      return { xs: tf.gather(xs, idx), ys: tf.gather(ys, idx), indices: idx };
    });
  }

  // رسم صورة 28x28 فـ canvas بسكيل nearest-neighbor
  draw28x28ToCanvas(t, canvas, scale = 4) {
    tf.tidy(() => {
      let img = t;
      if (img.rank === 4) img = img.squeeze([0, 3]);
      if (img.rank === 3) img = img.squeeze();

      const u8 = img.mul(255).clipByValue(0,255).toInt().dataSync();

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

  // تنظيف الميموري
  dispose() {
    if (this._train) { this._train.xs.dispose(); this._train.ys.dispose(); this._train = null; }
    if (this._test)  { this._test.xs.dispose();  this._test.ys.dispose();  this._test  = null; }
  }
}
