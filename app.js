// app.js — UPDATED WITH AUTOENCODER DENOISING
/**
 * MNIST browser-only trainer with:
 * - File-based CSV loading (handled by DataLoader)
 * - CNN training with tfjs-vis
 * - Evaluation: overall accuracy + confusion matrix + per-class accuracy bar chart
 * - Random 5 preview with colored predicted labels
 * - File-based save/load only (downloads:// and browserFiles)
 * - AUTOENCODER DENOISING: Train denoiser with autoencoder
 */

class MNISTApp {
  constructor() {
    this.data = new DataLoader();
    this.model = null;
    this.autoencoder = null; // NEW: Autoencoder model for denoising
    this.train = null; // { xs, ys }
    this.test  = null; // { xs, ys }
    this.bestValAcc = 0;
    this.currentMode = 'classifier'; // 'classifier' or 'autoencoder'

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
    
    // NEW: Mode switcher between classifier and autoencoder
    this.$('switchMode').addEventListener('click', () => this.switchMode());
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

  // NEW: Switch between classifier and autoencoder modes
  switchMode() {
    this.currentMode = this.currentMode === 'classifier' ? 'autoencoder' : 'classifier';
    const modeDisplay = this.$('currentMode');
    const trainButton = this.$('train');
    const testButton = this.$('testFive');
    
    modeDisplay.textContent = this.currentMode === 'classifier' ? 'Classifier Mode' : 'Autoencoder Denoising Mode';
    
    if (this.currentMode === 'autoencoder') {
      trainButton.textContent = 'Train Autoencoder';
      testButton.textContent = 'Test Denoising (5 Random)';
      this.log('Switched to Autoencoder Denoising Mode');
    } else {
      trainButton.textContent = 'Train';
      testButton.textContent = 'Test 5 Random';
      this.log('Switched to Classifier Mode');
    }
    
    // Reset model when switching modes
    if (this.model) {
      this.model.dispose();
      this.model = null;
    }
    if (this.autoencoder) {
      this.autoencoder.dispose();
      this.autoencoder = null;
    }
    this._setModelInfo('No model - select mode and train');
    this._setMetrics('Switch between classifier and denoising modes');
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

  // NEW: Build autoencoder model for denoising
  _buildAutoencoder() {
    const model = tf.sequential();
    
    // Encoder
    model.add(tf.layers.conv2d({
      filters: 32, 
      kernelSize: 3, 
      activation: 'relu', 
      padding: 'same', 
      inputShape: [28, 28, 1]
    }));
    model.add(tf.layers.maxPooling2d({ poolSize: 2, padding: 'same' }));
    model.add(tf.layers.conv2d({
      filters: 16, 
      kernelSize: 3, 
      activation: 'relu', 
      padding: 'same'
    }));
    
    // Decoder
    model.add(tf.layers.conv2d({
      filters: 16, 
      kernelSize: 3, 
      activation: 'relu', 
      padding: 'same'
    }));
    model.add(tf.layers.upSampling2d({ size: 2 }));
    model.add(tf.layers.conv2d({
      filters: 32, 
      kernelSize: 3, 
      activation: 'relu', 
      padding: 'same'
    }));
    model.add(tf.layers.conv2d({
      filters: 1, 
      kernelSize: 3, 
      activation: 'sigmoid', 
      padding: 'same'
    }));
    
    model.compile({
      optimizer: 'adam',
      loss: 'meanSquaredError',
      metrics: ['mse']
    });

    const total = model.countParams();
    this._setModelInfo(`Autoencoder Layers: <b>${model.layers.length}</b><br/>Total parameters: <b>${total.toLocaleString()}</b>`);
    console.log('Autoencoder Summary:');
    console.log(model.summary());
    return model;
  }

  async onTrain() {
    try {
      if (!this.train) { alert('Load data first.'); return; }

      if (this.currentMode === 'classifier') {
        await this._trainClassifier();
      } else {
        await this._trainAutoencoder();
      }
    } catch (err) {
      console.error(err);
      this.log(`Train error: ${err.message}`);
      alert(`Train error: ${err.message}`);
    }
  }

  async _trainClassifier() {
    // Dispose old model if any
    if (this.model) { this.model.dispose(); this.model = null; }
    this.model = this._buildModel();
    this._enable(true, false);
    this.bestValAcc = 0;

    // Train/val split
    const { trainXs, trainYs, valXs, valYs } = this.data.splitTrainVal(this.train.xs, this.train.ys, 0.1);

    const surface = { name: 'Classifier Training', tab: 'Training' };
    const fitCallbacks = tfvis.show.fitCallbacks(surface, ['loss', 'val_loss', 'acc', 'val_acc'], {
      callbacks: ['onEpochEnd', 'onBatchEnd'],
    });

    const epochs = 8;
    const batchSize = 128;
    this.log(`Training Classifier... epochs=${epochs}, batchSize=${batchSize}`);
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
    this.log(`Classifier training finished in ${s}s. Best Val Acc ${(this.bestValAcc*100).toFixed(2)}%.`);
    this._enable(true, true);

    // Cleanup split tensors
    trainXs.dispose(); trainYs.dispose(); valXs.dispose(); valYs.dispose();
  }

  async _trainAutoencoder() {
    // Dispose old autoencoder if any
    if (this.autoencoder) { this.autoencoder.dispose(); this.autoencoder = null; }
    this.autoencoder = this._buildAutoencoder();
    this._enable(true, false);

    // For autoencoder, we use images as both input and target
    // Create noisy inputs and clean targets
    const noisyTrain = this.data.addNoise(this.train.xs, 0.5);
    const noisyTest = this.data.addNoise(this.test.xs, 0.5);

    const surface = { name: 'Autoencoder Training', tab: 'Training' };
    const fitCallbacks = tfvis.show.fitCallbacks(surface, ['loss', 'val_loss'], {
      callbacks: ['onEpochEnd', 'onBatchEnd'],
    });

    const epochs = 10;
    const batchSize = 128;
    this.log(`Training Autoencoder... epochs=${epochs}, batchSize=${batchSize}`);
    const t0 = performance.now();

    await this.autoencoder.fit(noisyTrain, this.train.xs, {
      epochs, batchSize, shuffle: true,
      validationData: [noisyTest, this.test.xs],
      callbacks: {
        onEpochEnd: async (ep, logs) => {
          this._setMetrics(`Autoencoder Loss: <b>${logs?.loss?.toFixed(4)}</b>, Val Loss: <b>${logs?.val_loss?.toFixed(4)}</b>`);
          await fitCallbacks.onEpochEnd?.(ep, logs);
          await tf.nextFrame();
        },
        onBatchEnd: async (b, logs) => { await fitCallbacks.onBatchEnd?.(b, logs); }
      }
    });

    const s = ((performance.now() - t0) / 1000).toFixed(2);
    this.log(`Autoencoder training finished in ${s}s.`);
    this._enable(true, true);

    // Cleanup
    noisyTrain.dispose();
    noisyTest.dispose();
  }

  async onEvaluate() {
    try {
      if (this.currentMode === 'classifier') {
        await this._evaluateClassifier();
      } else {
        await this._evaluateAutoencoder();
      }
    } catch (err) {
      console.error('Detailed eval error:', err);
      this.log(`Eval error: ${err.message}`);
      alert(`Eval error: ${err.message}. Check console for details.`);
    }
  }

  async _evaluateClassifier() {
    if (!this.model || !this.test) { alert('Need a trained classifier + test data.'); return; }

    this.log('Evaluating classifier on test set...');

    // Overall metrics via model.evaluate
    const [lossT, accT] = this.model.evaluate(this.test.xs, this.test.ys);
    const [loss, acc] = [ (await lossT.data())[0], (await accT.data())[0] ];
    lossT.dispose(); accT.dispose();
    this._setMetrics(`Test Accuracy: <b>${(acc*100).toFixed(2)}%</b> &nbsp; | &nbsp; Test Loss: <b>${loss.toFixed(4)}</b>`);

    // Create proper 1D arrays for confusion matrix
    let yTrueArr, yPredArr;
    
    await tf.tidy(async () => {
      const trueLabelsTensor = this.test.ys.argMax(1);
      yTrueArr = await trueLabelsTensor.array();
      
      const predictionsTensor = this.model.predict(this.test.xs);
      const predLabelsTensor = predictionsTensor.argMax(1);
      yPredArr = await predLabelsTensor.array();
      
      predictionsTensor.dispose();
    });

    // Create confusion matrix
    const confusionMatrix = await tfvis.metrics.confusionMatrix(yTrueArr, yPredArr);

    const evalSurf = { name: 'Classifier Evaluation', tab: 'Evaluation' };
    
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
    this.log('Classifier evaluation complete. See Visor for charts.');
  }

  async _evaluateAutoencoder() {
    if (!this.autoencoder || !this.test) { alert('Need a trained autoencoder + test data.'); return; }

    this.log('Evaluating autoencoder denoising...');

    // Create noisy test data
    const noisyTest = this.data.addNoise(this.test.xs, 0.5);
    
    // Get denoised predictions
    const denoised = this.autoencoder.predict(noisyTest);
    
    // Calculate MSE between original and denoised
    const mse = tf.metrics.meanSquaredError(this.test.xs, denoised);
    const mseValue = (await mse.data())[0];
    
    this._setMetrics(`Denoising MSE: <b>${mseValue.toFixed(6)}</b>`);
    
    // Show sample comparisons
    this.log(`Autoencoder evaluation complete. MSE: ${mseValue.toFixed(6)}`);
    
    // Cleanup
    noisyTest.dispose();
    denoised.dispose();
    mse.dispose();
  }

  async onTestFive() {
    try {
      if (this.currentMode === 'classifier') {
        await this._testFiveClassifier();
      } else {
        await this._testFiveDenoising();
      }
    } catch (err) {
      console.error(err);
      this.log(`Preview error: ${err.message}`);
      alert(`Preview error: ${err.message}`);
    }
  }

  async _testFiveClassifier() {
    if (!this.model || !this.test) { alert('Need a trained classifier + test data.'); return; }

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
  }

  async _testFiveDenoising() {
    if (!this.autoencoder || !this.test) { alert('Need a trained autoencoder + test data.'); return; }

    // Get batch with original, noisy, and will generate denoised
    const batch = this.data.getRandomDenoisingBatch(this.test.xs, this.test.ys, 5, 0.5);
    const denoised = this.autoencoder.predict(batch.noisy);
    const trueLabels = Array.from(await batch.ys.argMax(-1).data());

    const row = this.$('preview');
    row.innerHTML = '<div style="grid-column: 1 / -1; text-align: center; margin-bottom: 10px; color: var(--muted);">Original | Noisy Input | Denoised Output</div>';

    for (let i = 0; i < 5; i++) {
      const wrap = document.createElement('div');
      wrap.className = 'pitem';

      // Original image
      const origCanvas = document.createElement('canvas');
      origCanvas.className = 'preview';
      const origImg = tf.tidy(() => batch.original.slice([i,0,0,0],[1,28,28,1]));
      this.data.draw28x28ToCanvas(origImg, origCanvas, 4);
      origImg.dispose();

      // Noisy image
      const noisyCanvas = document.createElement('canvas');
      noisyCanvas.className = 'preview';
      const noisyImg = tf.tidy(() => batch.noisy.slice([i,0,0,0],[1,28,28,1]));
      this.data.draw28x28ToCanvas(noisyImg, noisyCanvas, 4);
      noisyImg.dispose();

      // Denoised image
      const denoisedCanvas = document.createElement('canvas');
      denoisedCanvas.className = 'preview';
      const denoisedImg = tf.tidy(() => denoised.slice([i,0,0,0],[1,28,28,1]));
      this.data.draw28x28ToCanvas(denoisedImg, denoisedCanvas, 4);
      denoisedImg.dispose();

      const lab = document.createElement('div');
      lab.innerHTML = `True: <b>${trueLabels[i]}</b>`;
      lab.className = 'ok';

      wrap.appendChild(origCanvas);
      wrap.appendChild(noisyCanvas);
      wrap.appendChild(denoisedCanvas);
      wrap.appendChild(lab);
      row.appendChild(wrap);
    }

    denoised.dispose(); batch.original.dispose(); batch.noisy.dispose(); batch.ys.dispose();
    await tf.nextFrame();
    this.log('Rendered 5 random denoising examples.');
  }

  async onSaveDownload() {
    try {
      if (this.currentMode === 'classifier' && !this.model) { 
        alert('No classifier model to save. Train or load one first.'); 
        return; 
      }
      if (this.currentMode === 'autoencoder' && !this.autoencoder) { 
        alert('No autoencoder model to save. Train or load one first.'); 
        return; 
      }

      const modelToSave = this.currentMode === 'classifier' ? this.model : this.autoencoder;
      const prefix = this.currentMode === 'classifier' ? 'mnist-cnn' : 'mnist-autoencoder';
      
      await modelToSave.save(`downloads://${prefix}`);
      this.log(`Model saved (${prefix}.json + ${prefix}.weights.bin).`);
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

      if (this.currentMode === 'classifier') {
        if (this.model) this.model.dispose();
        this.model = loaded;
      } else {
        if (this.autoencoder) this.autoencoder.dispose();
        this.autoencoder = loaded;
      }

      const total = loaded.countParams();
      const modelType = this.currentMode === 'classifier' ? 'Classifier' : 'Autoencoder';
      this._setModelInfo(`Loaded ${modelType} model ✓<br/>Layers: <b>${loaded.layers.length}</b><br/>Total parameters: <b>${total.toLocaleString()}</b>`);
      this._enable(!!this.test, true);
      this.log(`${modelType} model loaded. You can Evaluate, Test 5 Random, or Save.`);
    } catch (err) {
      console.error(err);
      this.log(`Load model error: ${err.message}`);
      alert(`Load model error: ${err.message}`);
    }
  }

  onReset() {
    try {
      if (this.model) { this.model.dispose(); this.model = null; }
      if (this.autoencoder) { this.autoencoder.dispose(); this.autoencoder = null; }
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
window.addEventListener('DOMContentLoaded', () => { window.mnistApp = new
