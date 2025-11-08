// app.js
/**
 * Browser-only MNIST demo:
 * - File-based CSV loading (no network)
 * - CNN training with tfjs-vis live charts
 * - Evaluation (overall acc, confusion matrix, per-class accuracy)
 * - Preview 5 random test images with predicted vs true labels (colored)
 * - File-based model save/load (downloads + browserFiles)
 * - Memory-safe (dispose, tf.tidy) and responsive (await tf.nextFrame)
 */

class MNISTApp {
  constructor() {
    this.data = new DataLoader();
    this.model = null;
    this.train = null;
    this.test = null;
    this.bestValAcc = 0;
    this._initUI();
    this.log('Ready. Upload mnist_train.csv & mnist_test.csv, then click “Load Data”.');
  }

  _qs(id) { return document.getElementById(id); }

  _initUI() {
    this._qs('loadData').addEventListener('click', () => this.onLoadData());
    this._qs('train').addEventListener('click', () => this.onTrain());
    this._qs('evaluate').addEventListener('click', () => this.onEvaluate());
    this._qs('testFive').addEventListener('click', () => this.onTestFive());
    this._qs('saveModel').addEventListener('click', () => this.onSaveDownload());
    this._qs('loadModel').addEventListener('click', () => this.onLoadFromFiles());
    this._qs('reset').addEventListener('click', () => this.onReset());
    this._qs('toggleVisor').addEventListener('click', () => tfvis.visor().toggle());
  }

  log(msg) {
    const el = this._qs('logs');
    const t = new Date().toLocaleTimeString();
    const div = document.createElement('div');
    div.textContent = `[${t}] ${msg}`;
    el.appendChild(div);
    el.scrollTop = el.scrollHeight;
  }

  setStatusData(html) { this._qs('dataStatus').innerHTML = html; }
  setStatusMetrics(html){ this._qs('metrics').innerHTML = html; }
  setModelInfo(html){ this._qs('modelInfo').innerHTML = html; }

  _enableButtons(afterLoad=false, afterTrain=false) {
    this._qs('train').disabled = !afterLoad;
    this._qs('evaluate').disabled = !(afterLoad || afterTrain) || !this.model;
    this._qs('testFive').disabled = !(afterLoad || afterTrain) || !this.model;
    this._qs('saveModel').disabled = !this.model;
  }

  async onLoadData() {
    try {
      const trainFile = this._qs('trainFile').files[0];
      const testFile  = this._qs('testFile').files[0];
      if (!trainFile || !testFile) {
        alert('Select BOTH mnist_train.csv and mnist_test.csv.');
        return;
      }
      this.log('Loading training CSV…');
      this.train = await this.data.loadTrainFromFiles(trainFile);
      await tf.nextFrame();

      this.log('Loading test CSV…');
      this.test  = await this.data.loadTestFromFiles(testFile);
      await tf.nextFrame();

      const nTr = this.train.xs.shape[0];
      const nTe = this.test.xs.shape[0];
      this.setStatusData(`Train samples: <b>${nTr}</b><br/>Test samples: <b>${nTe}</b>`);
      this.log('Data loaded. You can Train or Load Model.');
      this._enableButtons(true, false);
    } catch (err) {
      console.error(err);
      this.log(`Load error: ${err.message}`);
      alert(`Load error: ${err.message}`);
    }
  }

  _buildModel() {
    const m = tf.sequential();
    // Conv block 1
    m.add(tf.layers.conv2d({ filters: 32, kernelSize: 3, activation: 'relu', padding: 'same', inputShape: [28,28,1] }));
    // Conv block 2
    m.add(tf.layers.conv2d({ filters: 64, kernelSize: 3, activation: 'relu', padding: 'same' }));
    // Pool + Dropout
    m.add(tf.layers.maxPooling2d({ poolSize: 2 }));
    m.add(tf.layers.dropout({ rate: 0.25 }));
    // Dense head
    m.add(tf.layers.flatten());
    m.add(tf.layers.dense({ units: 128, activation: 'relu' }));
    m.add(tf.layers.dropout({ rate: 0.5 }));
    m.add(tf.layers.dense({ units: 10, activation: 'softmax' }));

    m.compile({ optimizer: 'adam', loss: 'categoricalCrossentropy', metrics: ['accuracy'] });

    // Show summary & counts for “Model Info”
    const total = m.countParams();
    const layers = m.layers.length;
    this.setModelInfo(`Layers: <b>${layers}</b><br/>Total parameters: <b>${total.toLocaleString()}</b>`);
    console.log(m.summary());
    return m;
  }

  async onTrain() {
    try {
      if (!this.train) { alert('Load data first.'); return; }

      // Dispose old model if any
      if (this.model) {
        this.model.dispose();
        this.model = null;
      }
      this.model = this._buildModel();
      this._enableButtons(true, false);
      this.bestValAcc = 0;

      // Split train/val
      const { trainXs, trainYs, valXs, valYs } = this.data.splitTrainVal(this.train.xs, this.train.ys, 0.1);

      const surface = { name: 'Fit (loss / accuracy)', tab: 'Training' };
      const fitCallbacks = tfvis.show.fitCallbacks(surface, ['loss', 'val_loss', 'acc', 'val_acc'], {
        callbacks: ['onEpochEnd', 'onBatchEnd']
      });

      const epochs = 8;         // default 5–10
      const batchSize = 128;    // default 64–128

      this.log(`Training… epochs=${epochs}, batchSize=${batchSize}`);
      const t0 = performance.now();

      await this.model.fit(trainXs, trainYs, {
        epochs, batchSize, shuffle: true,
        validationData: [valXs, valYs],
        callbacks: {
          onEpochEnd: async (ep, logs) => {
            const valAcc = logs?.val_acc ?? logs?.val_accuracy ?? 0;
            if (valAcc > this.bestValAcc) this.bestValAcc = valAcc;
            this.setStatusMetrics(`Best Val Accuracy: <b>${(this.bestValAcc*100).toFixed(2)}%</b>`);
            await fitCallbacks.onEpochEnd?.(ep, logs);
            await tf.nextFrame();
          },
          onBatchEnd: async (b, logs) => { await fitCallbacks.onBatchEnd?.(b, logs); }
        }
      });

      const dur = ((performance.now() - t0)/1000).toFixed(2);
      this.log(`Training finished in ${dur}s. Best Val Acc ${(this.bestValAcc*100).toFixed(2)}%.`);
      this._enableButtons(true, true);

      // Dispose split tensors
      trainXs.dispose(); trainYs.dispose(); valXs.dispose(); valYs.dispose();
    } catch (err) {
      console.error(err);
      this.log(`Train error: ${err.message}`);
      alert(`Train error: ${err.message}`);
    }
  }

  async onEvaluate() {
    try {
      if (!this.model || !this.test) { alert('Need a trained or loaded model + test data.'); return; }

      this.log('Evaluating on test set…');
      const [lossT, accT] = this.model.evaluate(this.test.xs, this.test.ys);
      const loss = (await lossT.data())[0];
      const acc  = (await accT.data())[0];
      lossT.dispose(); accT.dispose();

      this.setStatusMetrics(`Test Accuracy: <b>${(acc*100).toFixed(2)}%</b> &nbsp; | &nbsp; Test Loss: <b>${loss.toFixed(4)}</b>`);
      await tf.nextFrame();

      // Predictions for confusion matrix and per-class accuracy
      const preds = tf.tidy(() => this.model.predict(this.test.xs).argMax(-1));
      const labels = this.test.ys.argMax(-1);

      const yPred = Array.from(await preds.data());
      const yTrue = Array.from(await labels.data());

      // Confusion matrix
      const cm = await tfvis.metrics.confusionMatrix(yTrue, yPred, 10);
      const evalSurf = { name: 'Evaluation', tab: 'Evaluation' };
      await tfvis.render.confusionMatrix(evalSurf, { values: cm, tickLabels: [...Array(10)].map((_,i)=>String(i)) }, { shadeDiagonal: true });

      // Per-class accuracy derived from CM
      const perClass = cm.map((row, i) => {
        const total = row.reduce((a,b)=>a+b,0);
        const correct = row[i] ?? 0;
        return { index: String(i), accuracy: total ? correct/total : 0 };
      });
      await tfvis.render.barchart(
        { name: 'Per-class Accuracy', tab: 'Evaluation' },
        perClass.map(d => ({ x: d.index, y: d.accuracy })),
        { yLabel: 'Accuracy', xLabel: 'Class', height: 300, yAxisDomain: [0,1] }
      );

      preds.dispose(); labels.dispose();
      this.log('Evaluation complete. See Visor for charts.');
    } catch (err) {
      console.error(err);
      this.log(`Eval error: ${err.message}`);
      alert(`Eval error: ${err.message}`);
    }
  }

  async onTestFive() {
    try {
      if (!this.model || !this.test) { alert('Need a trained or loaded model + test data.'); return; }

      const batch = this.data.getRandomTestBatch(this.test.xs, this.test.ys, 5);
      const pred = this.model.predict(batch.xs);
      const predLabels = Array.from(await pred.argMax(-1).data());
      const trueLabels = Array.from(await batch.ys.argMax(-1).data());

      const row = this._qs('preview');
      row.innerHTML = '';

      for (let i = 0; i < 5; i++) {
        const wrap = document.createElement('div');
        wrap.className = 'pitem';

        const c = document.createElement('canvas');
        c.className = 'preview';
        // slice a single image: [i,:,:,:]
        const img = tf.tidy(() => batch.xs.slice([i,0,0,0],[1,28,28,1]));
        this.data.draw28x28ToCanvas(img, c, 4);
        img.dispose();

        const ok = predLabels[i] === trueLabels[i];
        const lab = document.createElement('div');
        lab.innerHTML = `Pred: <b>${predLabels[i]}</b> &nbsp;|&nbsp; True: <b>${trueLabels[i]}</b>`;
        lab.className = ok ? 'ok' : 'bad';

        wrap.appendChild(c);
        wrap.appendChild(lab);
        row.appendChild(wrap);
      }

      pred.dispose(); batch.xs.dispose(); batch.ys.dispose();
      await tf.nextFrame();
      this.log('Rendered 5 random test predictions.');
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
      const json = this._qs('modelJson').files[0];
      const bin  = this._qs('modelWeights').files[0];
      if (!json || !bin) { alert('Pick BOTH model.json and weights.bin.'); return; }

      this.log('Loading model from files…');
      const loaded = await tf.loadLayersModel(tf.io.browserFiles([json, bin]));

      // Dispose any existing model to avoid leaks
      if (this.model) this.model.dispose();
      this.model = loaded;

      const total = this.model.countParams();
      this.setModelInfo(`Loaded model ✓<br/>Layers: <b>${this.model.layers.length}</b><br/>Total parameters: <b>${total.toLocaleString()}</b>`);
      this._enableButtons(!!this.test, true);
      this.log('Model loaded. You can Evaluate, Test 5 Random, or Save.');
    } catch (err) {
      console.error(err);
      this.log(`Load model error: ${err.message}`);
      alert(`Load model error: ${err.message}`);
    }
  }

  onReset() {
    try {
      // Dispose everything
      if (this.model) { this.model.dispose(); this.model = null; }
      if (this.train) { this.train.xs.dispose(); this.train.ys.dispose(); this.train = null; }
      if (this.test)  { this.test.xs.dispose();  this.test.ys.dispose();  this.test  = null; }
      this.data.dispose();

      // Clear UI
      this._qs('preview').innerHTML = '';
      this.setStatusData('No data loaded');
      this.setModelInfo('No model');
      this.setStatusMetrics('No metrics yet.');
      this._enableButtons(false, false);
      this.log('Reset complete.');
    } catch (err) {
      console.error(err);
      this.log(`Reset error: ${err.message}`);
    }
  }
}

// Boot
window.addEventListener('DOMContentLoaded', () => { window.mnistApp = new MNISTApp(); });
