// app.js — FINAL (fixes "labels must be a 1D tensor" and keeps evaluation robust)
/**
 * MNIST browser-only trainer with:
 * - File-based CSV loading (handled by DataLoader)
 * - CNN training with tfjs-vis
 * - Evaluation: overall accuracy + confusion matrix + per-class accuracy bar chart
 * - Random 5 preview with colored predicted labels
 * - File-based save/load only (downloads:// and browserFiles)
 *
 * Notes on fixes:
 * - Evaluation now builds 1D tensors for labels/preds explicitly (argMax(1)),
 *   converts to typed arrays, then feeds tfvis.metrics.confusionMatrix.
 * - No tfvis.show.perClassAccuracy misuse; we render a bar chart from the CM.
 */

class MNISTApp {
  constructor() {
    this.data = new DataLoader();
    this.model = null;
    this.train = null; // { xs, ys }
    this.test  = null; // { xs, ys }
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

  log(msg) {
    const t = new Date().toLocaleTimeString();
    const line = document.createElement('div');
    line.textContent = `[${t}] ${msg}`;
    const el = this.$('logs');
    el.appendChild(line);
    el.scrollTop = el.scrollHeight;
  }

  _setDataStatus(html)   { this.$('dataStatus').innerHTML = html; }
  _setModelInfo(html)    { this.$('modelInfo').innerHTML  = html; }
  _setMetrics(html)      { this.$('metrics').innerHTML    = html; }
  _enable(afterLoad=false, afterTrain=false) {
    this.$('train').disabled     = !afterLoad;
    const haveModel = !!this.model;
    const evalable  = (afterLoad || afterTrain) && haveModel;
    this.$('evaluate').disabled  = !evalable;
    this.$('testFive').disabled  = !evalable;
    this.$('saveModel').disabled = !haveModel;
  }

  async onLoadData() {
    try {
      const tr = this.$('trainFile').files[0];
      const te = this.$('testFile').files[0];
      if (!tr || !te) { alert('Select BOTH mnist_train.csv and mnist_test.csv.'); return; }

      this.log('Loading training CSV...');
      this.train = await this.data.loadTrainFromFiles(tr);
      await tf.nextFrame();

      this.log('Loading test CSV...');
      this.test  = await this.data.loadTestFromFiles(te);
      await tf.nextFrame();

      this._setDataStatus(`Train samples: <b>${this.train.xs.shape[0]}</b><br/>Test samples: <b>${this.test.xs.shape[0]}</b>`);
      this.log('Data loaded. You can Train or Load Model.');
      this._enable(true, false);
    } catch (err) {
      console.error(err);
      this.log(`Load error: ${err.message}`);
      alert(`Load error: ${err.message}`);
    }
  }

  _buildModel() {
    const m = tf.sequential();
    m.add(tf.layers.conv2d({ filters: 32, kernelSize: 3, activation: 'relu', padding: 'same', inputShape: [28,28,1] }));
    m.add(tf.layers.conv2d({ filters: 64, kernelSize: 3, activation: 'relu', padding: 'same' }));
    m.add(tf.layers.maxPooling2d({ poolSize: 2 }));
    m.add(tf.layers.dropout({ rate: 0.25 }));
    m.add(tf.layers.flatten());
    m.add(tf.layers.dense({ units: 128, activation: 'relu' }));
    m.add(tf.layers.dropout({ rate: 0.5 }));
    m.add(tf.layers.dense({ units: 10, activation: 'softmax' }));

    m.compile({ optimizer: 'adam', loss: 'categoricalCrossentropy', metrics: ['accuracy'] });

    const total = m.countParams();
    this._setModelInfo(`Layers: <b>${m.layers.length}</b><br/>Total parameters: <b>${total.toLocaleString()}</b>`);
    console.log(m.summary());
    return m;
  }

  async onTrain() {
    try {
      if (!this.train) { alert('Load data first.'); return; }

      // Dispose old model if any
      if (this.model) { this.model.dispose(); this.model = null; }
      this.model = this._buildModel();
      this._enable(true, false);
      this.bestValAcc = 0;

      // Train/val split
      const { trainXs, trainYs, valXs, valYs } = this.data.splitTrainVal(this.train.xs, this.train.ys, 0.1);

      const surface = { name: 'Fit (loss / accuracy)', tab: 'Training' };
      const fitCallbacks = tfvis.show.fitCallbacks(surface, ['loss', 'val_loss', 'acc', 'val_acc'], {
        callbacks: ['onEpochEnd', 'onBatchEnd'],
      });

      const epochs = 8;
      const batchSize = 128;
      this.log(`Training... epochs=${epochs}, batchSize=${batchSize}`);
      const t0 = performance.now();

      await this.model.fit(trainXs, trainYs, {
        epochs, batchSize, shuffle: true,
        validationData: [valXs, valYs],
        callbacks: {
          onEpochEnd: async (ep, logs) => {
            const va = logs?.val_acc ?? logs?.val_accuracy ?? 0;
            if (va > this.bestValAcc) this.bestValAcc = va;
            this._setMetrics(`Best Val Accuracy: <b>${(this.bestValAcc*100).toFixed(2)}%</b>`);
            await fitCallbacks.onEpochEnd?.(ep, logs);
            await tf.nextFrame();
          },
          onBatchEnd: async (b, logs) => { await fitCallbacks.onBatchEnd?.(b, logs); }
        }
      });

      const s = ((performance.now() - t0) / 1000).toFixed(2);
      this.log(`Training finished in ${s}s. Best Val Acc ${(this.bestValAcc*100).toFixed(2)}%.`);
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
      if (!this.model || !this.test) { alert('Need a trained or loaded model + test data.'); return; }

      this.log('Evaluating on test set...');

      // Overall metrics via model.evaluate
      const [lossT, accT] = this.model.evaluate(this.test.xs, this.test.ys);
      const [loss, acc] = [ (await lossT.data())[0], (await accT.data())[0] ];
      lossT.dispose(); accT.dispose();
      this._setMetrics(`Test Accuracy: <b>${(acc*100).toFixed(2)}%</b> &nbsp; | &nbsp; Test Loss: <b>${loss.toFixed(4)}</b>`);

      // FIXED: Create proper 1D arrays for confusion matrix
      let yTrueArr, yPredArr;
      
      // Use tf.tidy to automatically clean up intermediate tensors
      await tf.tidy(async () => {
        // Get true labels - convert from one-hot to class indices
        const trueLabelsTensor = this.test.ys.argMax(1); // Shape: [N]
        yTrueArr = await trueLabelsTensor.array(); // Convert to regular JavaScript array
        
        // Get predictions
        const predictionsTensor = this.model.predict(this.test.xs);
        const predLabelsTensor = predictionsTensor.argMax(1); // Shape: [N]
        yPredArr = await predLabelsTensor.array(); // Convert to regular JavaScript array
        
        // Explicitly dispose the tensors we created
        predictionsTensor.dispose();
      });

      console.log('True labels array:', yTrueArr.slice(0, 10)); // Debug first 10
      console.log('Pred labels array:', yPredArr.slice(0, 10)); // Debug first 10
      console.log('True labels type:', typeof yTrueArr[0]);
      console.log('Pred labels type:', typeof yPredArr[0]);

      // Verify we have proper arrays
      if (!Array.isArray(yTrueArr) || !Array.isArray(yPredArr)) {
        throw new Error('Labels are not proper arrays');
      }

      if (yTrueArr.length !== yPredArr.length) {
        throw new Error('True and prediction arrays have different lengths');
      }

      // Create confusion matrix
      this.log('Creating confusion matrix...');
      const confusionMatrix = await tfvis.metrics.confusionMatrix(yTrueArr, yPredArr);
      console.log('Confusion matrix:', confusionMatrix);

      const evalSurf = { name: 'Evaluation', tab: 'Evaluation' };
      
      // Render confusion matrix
      await tfvis.render.confusionMatrix(
        evalSurf, 
        { 
          values: confusionMatrix, 
          tickLabels: ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] 
        },
        { 
          width: 400, 
          height: 400 
        }
      );

      // Calculate and render per-class accuracy
      const perClassAccuracy = [];
      for (let i = 0; i < 10; i++) {
        const total = confusionMatrix[i].reduce((sum, val) => sum + val, 0);
        const correct = confusionMatrix[i][i];
        const accuracy = total > 0 ? correct / total : 0;
        perClassAccuracy.push({ x: i.toString(), y: accuracy });
      }

      await tfvis.render.barchart(
        { name: 'Per-class Accuracy', tab: 'Evaluation' },
        perClassAccuracy,
        { 
          xLabel: 'Digit', 
          yLabel: 'Accuracy',
          yAxisDomain: [0, 1],
          height: 300
        }
      );

      await tf.nextFrame();
      this.log('Evaluation complete. See Visor for charts.');
    } catch (err) {
      console.error('Detailed eval error:', err);
      this.log(`Eval error: ${err.message}`);
      alert(`Eval error: ${err.message}. Check console for details.`);
    }
  }

  async onTestFive() {
    try {
      if (!this.model || !this.test) { alert('Need a trained or loaded model + test data.'); return; }

      const batch = this.data.getRandomTestBatch(this.test.xs, this.test.ys, 5);
      const pred = this.model.predict(batch.xs);
      const predLabels = Array.from(await pred.argMax(-1).data());
      const trueLabels = Array.from(await batch.ys.argMax(-1).data());

      const row = this.$('preview');
      row.innerHTML = '';

      for (let i = 0; i < 5; i++) {
        const wrap = document.createElement('div');
        wrap.className = 'pitem';

        const c = document.createElement('canvas');
        c.className = 'preview';
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
      const json = this.$('modelJson').files[0];
      const bin  = this.$('modelWeights').files[0];
      if (!json || !bin) { alert('Pick BOTH model.json and weights.bin.'); return; }

      this.log('Loading model from files...');
      const loaded = await tf.loadLayersModel(tf.io.browserFiles([json, bin]));

      if (this.model) this.model.dispose();
      this.model = loaded;

      const total = this.model.countParams();
      this._setModelInfo(`Loaded model ✓<br/>Layers: <b>${this.model.layers.length}</b><br/>Total parameters: <b>${total.toLocaleString()}</b>`);
      this._enable(!!this.test, true);
      this.log('Model loaded. You can Evaluate, Test 5 Random, or Save.');
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
      this.data.dispose();

      this.$('preview').innerHTML = '';
      this._setDataStatus('No data loaded');
      this._setModelInfo('No model');
      this._setMetrics('No metrics yet.');
      this._enable(false, false);
      this.log('Reset complete.');
    } catch (err) {
      console.error(err);
      this.log(`Reset error: ${err.message}`);
    }
  }
}

// Boot
window.addEventListener('DOMContentLoaded', () => { window.mnistApp = new MNISTApp(); });
