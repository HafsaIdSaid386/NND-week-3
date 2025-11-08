// app.js — FINAL (fixes Eval error: "labels must be a 1D tensor")
/**
 * MNIST (browser-only) app:
 * - Loads CSVs from file inputs (handled by DataLoader in data-loader.js)
 * - Trains a CNN with tfjs-vis live charts
 * - Evaluates on test set with confusion matrix + per-class accuracy
 * - Shows 5 random test predictions
 * - Saves/loads model using files only (no IndexedDB)
 *
 * IMPORTANT FIXES:
 * - Evaluation now converts predictions/labels to plain JS arrays BEFORE
 *   calling tfvis.metrics.confusionMatrix / per-class accuracy.
 *   This avoids the error: "labels must be a 1D tensor".
 */

class MNISTApp {
  constructor() {
    this.loader = new DataLoader();
    this.train = null; // { xs, ys }
    this.test  = null; // { xs, ys }
    this.model = null;
    this.bestValAcc = 0;
    this._bindUI();
    this._log('Ready. Upload mnist_train.csv & mnist_test.csv, then click “Load Data”.');
  }

  // ---------- Small helpers ----------
  _$(id) { return document.getElementById(id); }
  _enableBtns({ afterLoad = false, hasModel = false } = {}) {
    this._$('train').disabled    = !afterLoad;
    this._$('evaluate').disabled = !(afterLoad && hasModel);
    this._$('testFive').disabled = !(afterLoad && hasModel);
    this._$('saveModel').disabled = !hasModel;
  }
  _setDataStatus(html)   { this._$('dataStatus').innerHTML = html; }
  _setMetrics(html)      { this._$('metrics').innerHTML = html; }
  _setModelInfo(html)    { this._$('modelInfo').innerHTML = html; }
  _log(msg) {
    const el = this._$('logs'); const t = new Date().toLocaleTimeString();
    const d = document.createElement('div'); d.textContent = `[${t}] ${msg}`;
    el.appendChild(d); el.scrollTop = el.scrollHeight;
  }

  // ---------- UI wiring ----------
  _bindUI() {
    this._$('loadData').addEventListener('click', () => this.onLoadData());
    this._$('train').addEventListener('click', () => this.onTrain());
    this._$('evaluate').addEventListener('click', () => this.onEvaluate());
    this._$('testFive').addEventListener('click', () => this.onTestFive());
    this._$('saveModel').addEventListener('click', () => this.onSaveDownload());
    this._$('loadModel').addEventListener('click', () => this.onLoadFromFiles());
    this._$('reset').addEventListener('click', () => this.onReset());
    this._$('toggleVisor').addEventListener('click', () => tfvis.visor().toggle());
    this._enableBtns({ afterLoad: false, hasModel: false });
  }

  // ---------- Data ----------
  async onLoadData() {
    try {
      const trainFile = this._$('trainFile').files[0];
      const testFile  = this._$('testFile').files[0];
      if (!trainFile || !testFile) {
        alert('Please choose BOTH mnist_train.csv and mnist_test.csv.');
        return;
      }
      this._log('Loading training CSV…');
      this.train = await this.loader.loadTrainFromFiles(trainFile);
      await tf.nextFrame();

      this._log('Loading test CSV…');
      this.test  = await this.loader.loadTestFromFiles(testFile);
      await tf.nextFrame();

      const nTr = this.train.xs.shape[0], nTe = this.test.xs.shape[0];
      this._setDataStatus(`Train samples: <b>${nTr}</b><br/>Test samples: <b>${nTe}</b>`);
      this._log('Data loaded. You can Train or Load a model.');
      this._enableBtns({ afterLoad: true, hasModel: !!this.model });
    } catch (err) {
      console.error(err);
      this._log(`Load error: ${err.message}`);
      alert(`Load error: ${err.message}`);
    }
  }

  // ---------- Model ----------
  _buildModel() {
    const m = tf.sequential();
    m.add(tf.layers.conv2d({
      filters: 32, kernelSize: 3, activation: 'relu', padding: 'same',
      inputShape: [28, 28, 1]
    }));
    m.add(tf.layers.conv2d({ filters: 64, kernelSize: 3, activation: 'relu', padding: 'same' }));
    m.add(tf.layers.maxPooling2d({ poolSize: 2 }));
    m.add(tf.layers.dropout({ rate: 0.25 }));
    m.add(tf.layers.flatten());
    m.add(tf.layers.dense({ units: 128, activation: 'relu' }));
    m.add(tf.layers.dropout({ rate: 0.5 }));
    m.add(tf.layers.dense({ units: 10, activation: 'softmax' }));
    m.compile({ optimizer: 'adam', loss: 'categoricalCrossentropy', metrics: ['accuracy'] });

    // Display model info
    this._setModelInfo(`Layers: <b>${m.layers.length}</b><br/>Total parameters: <b>${m.countParams().toLocaleString()}</b>`);
    try { console.log(m.summary()); } catch {}
    return m;
  }

  async onTrain() {
    try {
      if (!this.train) { alert('Load data first.'); return; }

      // Replace old model if any
      if (this.model) { this.model.dispose(); this.model = null; }
      this.model = this._buildModel();
      this.bestValAcc = 0;
      this._enableBtns({ afterLoad: true, hasModel: true });

      // Train/val split (uses DataLoader.splitTrainVal)
      const { trainXs, trainYs, valXs, valYs } = this.loader.splitTrainVal(this.train.xs, this.train.ys, 0.1);

      const surface = { name: 'Fit (loss / accuracy)', tab: 'Training' };
      const visCallbacks = tfvis.show.fitCallbacks(
        surface,
        ['loss', 'val_loss', 'acc', 'val_acc'],
        { callbacks: ['onEpochEnd', 'onBatchEnd'] }
      );

      const epochs = 8, batchSize = 128;
      this._log(`Training… epochs=${epochs}, batchSize=${batchSize}`);
      const t0 = performance.now();

      await this.model.fit(trainXs, trainYs, {
        epochs, batchSize, shuffle: true, validationData: [valXs, valYs],
        callbacks: {
          onEpochEnd: async (epoch, logs) => {
            const valAcc = logs?.val_acc ?? logs?.val_accuracy ?? 0;
            if (valAcc > this.bestValAcc) this.bestValAcc = valAcc;
            this._setMetrics(`Best Val Accuracy: <b>${(this.bestValAcc * 100).toFixed(2)}%</b>`);
            await visCallbacks.onEpochEnd?.(epoch, logs);
            await tf.nextFrame();
          },
          onBatchEnd: async (b, logs) => { await visCallbacks.onBatchEnd?.(b, logs); }
        }
      });

      const dur = ((performance.now() - t0) / 1000).toFixed(2);
      this._log(`Training finished in ${dur}s. Best Val Acc ${(this.bestValAcc * 100).toFixed(2)}%.`);

      // Clean split tensors
      trainXs.dispose(); trainYs.dispose(); valXs.dispose(); valYs.dispose();
    } catch (err) {
      console.error(err);
      this._log(`Train error: ${err.message}`);
      alert(`Train error: ${err.message}`);
    }
  }

  // ---------- Evaluation (FIXED: use arrays, not tensors, for tfvis.metrics.*) ----------
  async onEvaluate() {
    try {
      if (!this.model || !this.test) { alert('Need a model and test data.'); return; }
      this._log('Evaluating on test set…');

      const [lossT, accT] = this.model.evaluate(this.test.xs, this.test.ys);
      const loss = (await lossT.data())[0];
      const acc  = (await accT.data())[0];
      lossT.dispose(); accT.dispose();
      this._setMetrics(`Test Accuracy: <b>${(acc * 100).toFixed(2)}%</b> &nbsp; | &nbsp; Test Loss: <b>${loss.toFixed(4)}</b>`);
      await tf.nextFrame();

      // Predictions and labels as PLAIN ARRAYS (not tensors!) for tfvis.metrics.*
      const predTensor = this.model.predict(this.test.xs);
      const yPredArr   = Array.from(await predTensor.argMax(-1).data()); // [N]
      predTensor.dispose();

      const yTrueArr   = Array.from(await this.test.ys.argMax(-1).data()); // [N]

      // Confusion matrix
      const cm = await tfvis.metrics.confusionMatrix(yTrueArr, yPredArr, 10);
      const evalSurf = { name: 'Evaluation', tab: 'Evaluation' };
      await tfvis.render.confusionMatrix(
        evalSurf,
        { values: cm, tickLabels: [...Array(10)].map((_, i) => String(i)) },
        { shadeDiagonal: true }
      );

      // Per-class accuracy from CM
      const perClass = cm.map((row, i) => {
        const total = row.reduce((a, b) => a + b, 0);
        const correct = row[i] || 0;
        return { x: String(i), y: total ? correct / total : 0 };
      });
      await tfvis.render.barchart(
        { name: 'Per-class Accuracy', tab: 'Evaluation' },
        perClass,
        { xLabel: 'Class', yLabel: 'Accuracy', yAxisDomain: [0, 1], height: 300 }
      );

      this._log('Evaluation complete. See Visor for charts.');
    } catch (err) {
      console.error(err);
      this._log(`Eval error: ${err.message}`);
      alert(`Eval error: ${err.message}`);
    }
  }

  // ---------- Preview 5 ----------
  async onTestFive() {
    try {
      if (!this.model || !this.test) { alert('Need a model and test data.'); return; }

      const batch = this.loader.getRandomTestBatch(this.test.xs, this.test.ys, 5);
      const pred = this.model.predict(batch.xs);
      const predLabels = Array.from(await pred.argMax(-1).data());
      const trueLabels = Array.from(await batch.ys.argMax(-1).data());

      const row = this._$('preview'); row.innerHTML = '';
      for (let i = 0; i < 5; i++) {
        const wrap = document.createElement('div'); wrap.className = 'pitem';
        const c = document.createElement('canvas'); c.className = 'preview';
        const img = tf.tidy(() => batch.xs.slice([i, 0, 0, 0], [1, 28, 28, 1]));
        this.loader.draw28x28ToCanvas(img, c, 4); img.dispose();

        const ok = predLabels[i] === trueLabels[i];
        const lab = document.createElement('div');
        lab.innerHTML = `Pred: <b>${predLabels[i]}</b> &nbsp;|&nbsp; True: <b>${trueLabels[i]}</b>`;
        lab.className = ok ? 'ok' : 'bad';

        wrap.appendChild(c); wrap.appendChild(lab); row.appendChild(wrap);
      }

      pred.dispose(); batch.xs.dispose(); batch.ys.dispose();
      await tf.nextFrame();
      this._log('Rendered 5 random test predictions.');
    } catch (err) {
      console.error(err);
      this._log(`Preview error: ${err.message}`);
      alert(`Preview error: ${err.message}`);
    }
  }

  // ---------- Save / Load ----------
  async onSaveDownload() {
    try {
      if (!this.model) { alert('No model to save.'); return; }
      await this.model.save('downloads://mnist-cnn');
      this._log('Model saved (model.json + weights.bin).');
    } catch (err) {
      console.error(err);
      this._log(`Save error: ${err.message}`);
      alert(`Save error: ${err.message}`);
    }
  }

  async onLoadFromFiles() {
    try {
      const json = this._$('modelJson').files[0];
      const bin  = this._$('modelWeights').files[0];
      if (!json || !bin) { alert('Pick BOTH model.json and weights.bin.'); return; }

      this._log('Loading model from files…');
      const loaded = await tf.loadLayersModel(tf.io.browserFiles([json, bin]));
      if (this.model) this.model.dispose();
      this.model = loaded;

      const total = this.model.countParams();
      this._setModelInfo(`Loaded model ✓<br/>Layers: <b>${this.model.layers.length}</b><br/>Total parameters: <b>${total.toLocaleString()}</b>`);
      this._enableBtns({ afterLoad: !!this.test, hasModel: true });
      this._log('Model loaded. You can Evaluate or Test 5 Random.');
    } catch (err) {
      console.error(err);
      this._log(`Load model error: ${err.message}`);
      alert(`Load model error: ${err.message}`);
    }
  }

  // ---------- Reset ----------
  onReset() {
    try {
      if (this.model) { this.model.dispose(); this.model = null; }
      if (this.train) { this.train.xs.dispose(); this.train.ys.dispose(); this.train = null; }
      if (this.test)  { this.test.xs.dispose();  this.test.ys.dispose();  this.test  = null; }
      this.loader.dispose();

      this._$('preview').innerHTML = '';
      this._setDataStatus('No data loaded');
      this._setModelInfo('No model');
      this._setMetrics('No metrics yet.');
      this._enableBtns({ afterLoad: false, hasModel: false });
      this._log('Reset complete.');
    } catch (err) {
      console.error(err);
      this._log(`Reset error: ${err.message}`);
    }
  }
}

// Boot
window.addEventListener('DOMContentLoaded', () => { window.mnistApp = new MNISTApp(); });
