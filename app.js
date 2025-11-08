// app.js — FINAL
/**
 * Browser-only MNIST trainer/evaluator using TensorFlow.js + tfjs-vis.
 * - Loads CSV data from file inputs (handled by data-loader.js).
 * - Trains a small CNN with live charts.
 * - Evaluates on test set with confusion matrix + per-class accuracy.
 * - Shows 5 random test predictions.
 * - File-based save/load (downloads:// + browserFiles).
 *
 * NOTE (fixes for your earlier errors):
 * 1) Train split uses Tensor indices (tf.tensor1d(..., 'int32')) when calling tf.gather
 *    to avoid "indices must be a Tensor" errors from Uint32Array.
 * 2) Evaluation builds 1D JS arrays for tfjs-vis.confusionMatrix to avoid
 *    "labels must be a 1D tensor" issues.
 */

class MNISTApp {
  constructor() {
    this.dataLoader = new DataLoader();
    this.train = null; // { xs, ys }
    this.test  = null; // { xs, ys }
    this.model = null;
    this.bestValAcc = 0;

    this.$ = (id) => document.getElementById(id);
    this._bindUI();
    this.log('Ready. Upload mnist_train.csv & mnist_test.csv, then click "Load Data".');
  }

  _bindUI() {
    this.$('loadData').addEventListener('click', () => this.onLoadData());
    this.$('train').addEventListener('click', () => this.onTrain());
    this.$('evaluate').addEventListener('click', () => this.onEvaluate());
    this.$('testFive').addEventListener('click', () => this.onTestFive());
    this.$('saveModel').addEventListener('click', () => this.onSaveDownload());
    this.$('loadModel').addEventListener('click', () => this.onLoadFromFiles());
    this.$('reset').addEventListener('click', () => this.onReset());
    this.$('toggleVisor').addEventListener('click', () => tfvis.visor().toggle());
  }

  log(message) {
    const el = this.$('logs');
    const t = new Date().toLocaleTimeString();
    const line = document.createElement('div');
    line.textContent = `[${t}] ${message}`;
    el.appendChild(line);
    el.scrollTop = el.scrollHeight;
  }

  _enable(afterLoad = false, afterTrain = false) {
    this.$('train').disabled = !afterLoad;
    this.$('evaluate').disabled = !(afterLoad || afterTrain) || !this.model;
    this.$('testFive').disabled = !(afterLoad || afterTrain) || !this.model;
    this.$('saveModel').disabled = !this.model;
  }

  async onLoadData() {
    try {
      const trainFile = this.$('trainFile').files[0];
      const testFile  = this.$('testFile').files[0];
      if (!trainFile || !testFile) {
        alert('Please choose BOTH mnist_train.csv and mnist_test.csv files.');
        return;
      }

      this.log('Loading training CSV…');
      this.train = await this.dataLoader.loadTrainFromFiles(trainFile);
      await tf.nextFrame();

      this.log('Loading test CSV…');
      this.test  = await this.dataLoader.loadTestFromFiles(testFile);
      await tf.nextFrame();

      const nTr = this.train.xs.shape[0];
      const nTe = this.test.xs.shape[0];
      this.$('dataStatus').innerHTML = `Train samples: <b>${nTr}</b><br/>Test samples: <b>${nTe}</b>`;
      this._enable(true, false);
      this.log('Data loaded. You can Train, or Load a model from files.');
    } catch (err) {
      console.error(err);
      this.log(`Load error: ${err.message}`);
      alert(`Load error: ${err.message}`);
    }
  }

  _buildModel() {
    const model = tf.sequential();
    model.add(tf.layers.conv2d({
      filters: 32, kernelSize: 3, activation: 'relu', padding: 'same', inputShape: [28, 28, 1]
    }));
    model.add(tf.layers.conv2d({ filters: 64, kernelSize: 3, activation: 'relu', padding: 'same' }));
    model.add(tf.layers.maxPooling2d({ poolSize: 2 }));
    model.add(tf.layers.dropout({ rate: 0.25 }));
    model.add(tf.layers.flatten());
    model.add(tf.layers.dense({ units: 128, activation: 'relu' }));
    model.add(tf.layers.dropout({ rate: 0.5 }));
    model.add(tf.layers.dense({ units: 10, activation: 'softmax' }));

    model.compile({ optimizer: 'adam', loss: 'categoricalCrossentropy', metrics: ['accuracy'] });

    // Update Model Info panel
    const total = model.countParams();
    this.$('modelInfo').innerHTML = `Layers: <b>${model.layers.length}</b><br/>Total parameters: <b>${total.toLocaleString()}</b>`;
    console.log(model.summary());
    return model;
  }

  async onTrain() {
    try {
      if (!this.train) { alert('Please load data first.'); return; }

      // Replace any existing model
      if (this.model) { this.model.dispose(); this.model = null; }
      this.model = this._buildModel();
      this._enable(true, false);
      this.bestValAcc = 0;

      // === IMPORTANT: make Tensor indices for tf.gather (fixes Uint32Array issue) ===
      const N = this.train.xs.shape[0];
      const allIdx = tf.util.createShuffledIndices(N);
      const valCount = Math.max(1, Math.floor(N * 0.1));
      const valIdxA = Array.from(allIdx.slice(0, valCount));
      const trnIdxA = Array.from(allIdx.slice(valCount));
      const valIdxT = tf.tensor1d(valIdxA, 'int32');
      const trnIdxT = tf.tensor1d(trnIdxA, 'int32');

      const trainXs = tf.gather(this.train.xs, trnIdxT);
      const trainYs = tf.gather(this.train.ys, trnIdxT);
      const valXs   = tf.gather(this.train.xs, valIdxT);
      const valYs   = tf.gather(this.train.ys, valIdxT);
      valIdxT.dispose(); trnIdxT.dispose();

      const surface = { name: 'Fit (loss / accuracy)', tab: 'Training' };
      const visCallbacks = tfvis.show.fitCallbacks(surface, ['loss', 'val_loss', 'acc', 'val_acc'], {
        callbacks: ['onEpochEnd', 'onBatchEnd']
      });

      const epochs = 8;
      const batchSize = 128;
      this.log(`Training… epochs=${epochs}, batchSize=${batchSize}`);
      const t0 = performance.now();

      await this.model.fit(trainXs, trainYs, {
        epochs, batchSize, shuffle: true, validationData: [valXs, valYs],
        callbacks: {
          onEpochEnd: async (epoch, logs) => {
            const valAcc = logs?.val_acc ?? logs?.val_accuracy ?? 0;
            if (valAcc > this.bestValAcc) this.bestValAcc = valAcc;
            this.$('metrics').innerHTML = `Best Val Accuracy: <b>${(this.bestValAcc * 100).toFixed(2)}%</b>`;
            await visCallbacks.onEpochEnd?.(epoch, logs);
            await tf.nextFrame();
          },
          onBatchEnd: async (batch, logs) => { await visCallbacks.onBatchEnd?.(batch, logs); }
        }
      });

      const dur = ((performance.now() - t0) / 1000).toFixed(2);
      this.log(`Training finished in ${dur}s. Best Val Acc ${(this.bestValAcc * 100).toFixed(2)}%.`);
      this._enable(true, true);

      // Cleanup split tensors
      trainXs.dispose(); trainYs.dispose(); valXs.dispose(); valYs.dispose();
    } catch (err) {
      console.error(err);
      this.log(`Train error: ${err.message}`);
      alert(`Train error: ${err.message}`);
    }
  }

  async onEvaluate() {
    try {
      if (!this.model || !this.test) { alert('Need a trained/loaded model AND test data.'); return; }
      this.log('Evaluating on test set…');

      const [lossT, accT] = this.model.evaluate(this.test.xs, this.test.ys);
      const loss = (await lossT.data())[0];
      const acc  = (await accT.data())[0];
      lossT.dispose(); accT.dispose();

      this.$('metrics').innerHTML = `Test Accuracy: <b>${(acc * 100).toFixed(2)}%</b> &nbsp; | &nbsp; Test Loss: <b>${loss.toFixed(4)}</b>`;

      // === Build 1D JS arrays for tfjs-vis metrics (avoids "labels must be a 1D tensor") ===
      const pred1D = tf.tidy(() => this.model.predict(this.test.xs).argMax(1)); // [N]
      const true1D = tf.tidy(() => this.test.ys.argMax(1));                     // [N]
      const yPred = Array.from(await pred1D.data());
      const yTrue = Array.from(await true1D.data());
      pred1D.dispose(); true1D.dispose();

      const cm = await tfvis.metrics.confusionMatrix(yTrue, yPred, 10);
      await tfvis.render.confusionMatrix(
        { name: 'Confusion Matrix', tab: 'Evaluation' },
        { values: cm, tickLabels: [...Array(10)].map((_, i) => String(i)) },
        { shadeDiagonal: true }
      );

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

      this.log('Evaluation complete. See Visor for detailed charts.');
    } catch (err) {
      console.error(err);
      this.log(`Eval error: ${err.message}`);
      alert(`Eval error: ${err.message}`);
    }
  }

  async onTestFive() {
    try {
      if (!this.model || !this.test) { alert('Need a trained/loaded model AND test data.'); return; }

      const k = 5;
      // Random indices as a Tensor to use with tf.gather
      const n = this.test.xs.shape[0];
      const idxArr = Array.from({ length: k }, () => Math.floor(Math.random() * n));
      const idxT = tf.tensor1d(idxArr, 'int32');

      const xs = tf.gather(this.test.xs, idxT);
      const ys = tf.gather(this.test.ys, idxT);
      idxT.dispose();

      const pred = this.model.predict(xs);
      const yPred = Array.from(await pred.argMax(-1).data());
      const yTrue = Array.from(await ys.argMax(-1).data());

      const row = this.$('preview');
      row.innerHTML = '';
      for (let i = 0; i < k; i++) {
        const wrap = document.createElement('div'); wrap.className = 'pitem';
        const canvas = document.createElement('canvas'); canvas.className = 'preview';

        const img = tf.tidy(() => xs.slice([i, 0, 0, 0], [1, 28, 28, 1]));
        this.dataLoader.draw28x28ToCanvas(img, canvas, 4);
        img.dispose();

        const ok = yPred[i] === yTrue[i];
        const lab = document.createElement('div');
        lab.className = ok ? 'ok' : 'bad';
        lab.innerHTML = `Pred: <b>${yPred[i]}</b> &nbsp;|&nbsp; True: <b>${yTrue[i]}</b>`;

        wrap.appendChild(canvas);
        wrap.appendChild(lab);
        row.appendChild(wrap);
      }

      pred.dispose(); xs.dispose(); ys.dispose();
      this.log('Rendered 5 random predictions.');
    } catch (err) {
      console.error(err);
      this.log(`Preview error: ${err.message}`);
      alert(`Preview error: ${err.message}`);
    }
  }

  async onSaveDownload() {
    try {
      if (!this.model) { alert('No model to save. Train or load one first.'); return; }
      await this.model.save('downloads://mnist-cnn');
      this.log('Model saved (model.json + weights.bin).');
    } catch (err) {
      console.error(err);
      this.log(`Save error: ${err.message}`);
      alert(`Save error: ${err.message}`);
    }
  }

  async onLoadFromFiles() {
    try {
      const jsonF = this.$('modelJson').files[0];
      const binF  = this.$('modelWeights').files[0];
      if (!jsonF || !binF) { alert('Pick BOTH model.json and weights.bin.'); return; }

      this.log('Loading model from files…');
      const loaded = await tf.loadLayersModel(tf.io.browserFiles([jsonF, binF]));
      if (this.model) this.model.dispose();
      this.model = loaded;

      const total = this.model.countParams();
      this.$('modelInfo').innerHTML = `Loaded model ✓<br/>Layers: <b>${this.model.layers.length}</b><br/>Total parameters: <b>${total.toLocaleString()}</b>`;
      this._enable(!!this.test, true);
      this.log('Model loaded. You can Evaluate or Test 5 Random.');
    } catch (err) {
      console.error(err);
      this.log(`Load model error: ${err.message}`);
      alert(`Load model error: ${err.message}`);
    }
  }

  onReset() {
    try {
      if (this.model) { this.model.dispose(); this.model = null; }
      if (this.train) { this.train.xs.dispose(); this.train.ys.dispose(); this.train = null; }
      if (this.test)  { this.test.xs.dispose();  this.test.ys.dispose();  this.test  = null; }
      this.dataLoader.dispose();

      this.$('preview').innerHTML = '';
      this.$('dataStatus').innerHTML = 'No data loaded';
      this.$('modelInfo').innerHTML = 'No model';
      this.$('metrics').innerHTML = 'No metrics yet.';
      this._enable(false, false);
      this.log('Reset complete.');
    } catch (err) {
      console.error(err);
      this.log(`Reset error: ${err.message}`);
    }
  }
}

// Boot the app
window.addEventListener('DOMContentLoaded', () => { window.mnistApp = new MNISTApp(); });
