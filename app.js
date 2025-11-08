// app.js - SIMPLIFIED WORKING VERSION
class MNISTApp {
    constructor() {
        this.data = new DataLoader();
        this.model = null;
        this.autoencoder = null;
        this.trainData = null;
        this.testData = null;
        this.currentMode = 'classifier';
        
        this.init();
    }

    init() {
        this.bindEvents();
        this.log('App loaded. Upload CSV files and click Load Data.');
    }

    bindEvents() {
        document.getElementById('loadData').addEventListener('click', () => this.loadData());
        document.getElementById('train').addEventListener('click', () => this.train());
        document.getElementById('evaluate').addEventListener('click', () => this.evaluate());
        document.getElementById('testFive').addEventListener('click', () => this.testFive());
        document.getElementById('saveModel').addEventListener('click', () => this.saveModel());
        document.getElementById('loadModel').addEventListener('click', () => this.loadModel());
        document.getElementById('reset').addEventListener('click', () => this.reset());
        document.getElementById('toggleVisor').addEventListener('click', () => tfvis.visor().toggle());
        document.getElementById('switchMode').addEventListener('click', () => this.switchMode());
    }

    log(message) {
        const logs = document.getElementById('logs');
        const line = document.createElement('div');
        line.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
        logs.appendChild(line);
        logs.scrollTop = logs.scrollHeight;
    }

    async loadData() {
        try {
            const trainFile = document.getElementById('trainFile').files[0];
            const testFile = document.getElementById('testFile').files[0];
            
            if (!trainFile || !testFile) {
                alert('Please select both train and test CSV files');
                return;
            }

            this.log('Loading training data...');
            this.trainData = await this.data.loadTrainFromFiles(trainFile);
            
            this.log('Loading test data...');
            this.testData = await this.data.loadTestFromFiles(testFile);
            
            document.getElementById('dataStatus').innerHTML = 
                `Train: ${this.trainData.xs.shape[0]} samples<br>Test: ${this.testData.xs.shape[0]} samples`;
            
            document.getElementById('train').disabled = false;
            this.log('Data loaded successfully!');
            
        } catch (error) {
            this.log(`Error: ${error.message}`);
            alert(`Load error: ${error.message}`);
        }
    }

    switchMode() {
        this.currentMode = this.currentMode === 'classifier' ? 'autoencoder' : 'classifier';
        const modeElement = document.getElementById('currentMode');
        const trainBtn = document.getElementById('train');
        const testBtn = document.getElementById('testFive');
        
        modeElement.textContent = this.currentMode === 'classifier' ? 'Classifier Mode' : 'Autoencoder Denoising Mode';
        modeElement.className = `mode-indicator ${this.currentMode}`;
        
        if (this.currentMode === 'autoencoder') {
            trainBtn.textContent = 'Train Autoencoder';
            testBtn.textContent = 'Test Denoising';
            this.log('Switched to Autoencoder Denoising Mode');
        } else {
            trainBtn.textContent = 'Train Classifier';
            testBtn.textContent = 'Test 5 Random';
            this.log('Switched to Classifier Mode');
        }
    }

    createClassifier() {
        const model = tf.sequential();
        model.add(tf.layers.conv2d({
            filters: 32, kernelSize: 3, activation: 'relu', padding: 'same', inputShape: [28, 28, 1]
        }));
        model.add(tf.layers.maxPooling2d({ poolSize: 2 }));
        model.add(tf.layers.conv2d({ filters: 64, kernelSize: 3, activation: 'relu', padding: 'same' }));
        model.add(tf.layers.maxPooling2d({ poolSize: 2 }));
        model.add(tf.layers.flatten());
        model.add(tf.layers.dense({ units: 128, activation: 'relu' }));
        model.add(tf.layers.dropout({ rate: 0.5 }));
        model.add(tf.layers.dense({ units: 10, activation: 'softmax' }));

        model.compile({
            optimizer: 'adam',
            loss: 'categoricalCrossentropy',
            metrics: ['accuracy']
        });

        return model;
    }

    createAutoencoder() {
        const model = tf.sequential();
        
        // Encoder
        model.add(tf.layers.conv2d({
            filters: 32, kernelSize: 3, activation: 'relu', padding: 'same', inputShape: [28, 28, 1]
        }));
        model.add(tf.layers.maxPooling2d({ poolSize: 2, padding: 'same' }));
        model.add(tf.layers.conv2d({
            filters: 16, kernelSize: 3, activation: 'relu', padding: 'same'
        }));
        
        // Decoder
        model.add(tf.layers.conv2d({
            filters: 16, kernelSize: 3, activation: 'relu', padding: 'same'
        }));
        model.add(tf.layers.upSampling2d({ size: 2 }));
        model.add(tf.layers.conv2d({
            filters: 32, kernelSize: 3, activation: 'relu', padding: 'same'
        }));
        model.add(tf.layers.conv2d({
            filters: 1, kernelSize: 3, activation: 'sigmoid', padding: 'same'
        }));

        model.compile({
            optimizer: 'adam',
            loss: 'meanSquaredError'
        });

        return model;
    }

    async train() {
        if (!this.trainData) {
            alert('Please load data first');
            return;
        }

        try {
            if (this.currentMode === 'classifier') {
                await this.trainClassifier();
            } else {
                await this.trainAutoencoder();
            }
        } catch (error) {
            this.log(`Training error: ${error.message}`);
        }
    }

    async trainClassifier() {
        if (this.model) this.model.dispose();
        this.model = this.createClassifier();

        const { trainXs, trainYs, valXs, valYs } = this.data.splitTrainVal(
            this.trainData.xs, this.trainData.ys, 0.1
        );

        this.log('Training classifier...');
        
        await this.model.fit(trainXs, trainYs, {
            epochs: 5,
            batchSize: 128,
            validationData: [valXs, valYs],
            callbacks: tfvis.show.fitCallbacks(
                { name: 'Classifier Training', tab: 'Training' },
                ['loss', 'val_loss', 'acc', 'val_acc']
            )
        });

        document.getElementById('evaluate').disabled = false;
        document.getElementById('testFive').disabled = false;
        document.getElementById('saveModel').disabled = false;
        this.log('Classifier training completed!');
    }

    async trainAutoencoder() {
        if (this.autoencoder) this.autoencoder.dispose();
        this.autoencoder = this.createAutoencoder();

        // Create noisy data for training
        const noisyTrain = this.data.addNoise(this.trainData.xs);
        const noisyTest = this.data.addNoise(this.testData.xs);

        this.log('Training autoencoder for denoising...');
        
        await this.autoencoder.fit(noisyTrain, this.trainData.xs, {
            epochs: 10,
            batchSize: 128,
            validationData: [noisyTest, this.testData.xs],
            callbacks: tfvis.show.fitCallbacks(
                { name: 'Autoencoder Training', tab: 'Training' },
                ['loss', 'val_loss']
            )
        });

        document.getElementById('evaluate').disabled = false;
        document.getElementById('testFive').disabled = false;
        document.getElementById('saveModel').disabled = false;
        this.log('Autoencoder training completed!');
        
        noisyTrain.dispose();
        noisyTest.dispose();
    }

    async evaluate() {
        if (!this.testData) {
            alert('Please load test data first');
            return;
        }

        try {
            if (this.currentMode === 'classifier') {
                await this.evaluateClassifier();
            } else {
                await this.evaluateAutoencoder();
            }
        } catch (error) {
            this.log(`Evaluation error: ${error.message}`);
        }
    }

    async evaluateClassifier() {
        if (!this.model) {
            alert('Please train a classifier first');
            return;
        }

        const [testLoss, testAcc] = this.model.evaluate(this.testData.xs, this.testData.ys);
        const loss = (await testLoss.data())[0];
        const acc = (await testAcc.data())[0];
        
        document.getElementById('metrics').innerHTML = 
            `Test Accuracy: <b>${(acc * 100).toFixed(2)}%</b><br>Test Loss: <b>${loss.toFixed(4)}</b>`;
        
        this.log(`Classifier evaluation: ${(acc * 100).toFixed(2)}% accuracy`);
        
        testLoss.dispose();
        testAcc.dispose();
    }

    async evaluateAutoencoder() {
        if (!this.autoencoder) {
            alert('Please train an autoencoder first');
            return;
        }

        const noisyTest = this.data.addNoise(this.testData.xs);
        const denoised = this.autoencoder.predict(noisyTest);
        const mse = tf.losses.meanSquaredError(this.testData.xs, denoised);
        const mseValue = (await mse.data())[0];
        
        document.getElementById('metrics').innerHTML = 
            `Denoising MSE: <b>${mseValue.toFixed(6)}</b>`;
        
        this.log(`Autoencoder evaluation: MSE = ${mseValue.toFixed(6)}`);
        
        noisyTest.dispose();
        denoised.dispose();
        mse.dispose();
    }

    async testFive() {
        if (!this.testData) {
            alert('Please load test data first');
            return;
        }

        try {
            if (this.currentMode === 'classifier') {
                await this.testFiveClassifier();
            } else {
                await this.testFiveDenoising();
            }
        } catch (error) {
            this.log(`Test error: ${error.message}`);
        }
    }

    async testFiveClassifier() {
        if (!this.model) {
            alert('Please train a classifier first');
            return;
        }

        const batch = this.data.getRandomTestBatch(this.testData.xs, this.testData.ys, 5);
        const predictions = this.model.predict(batch.xs);
        const predLabels = Array.from(await predictions.argMax(1).data());
        const trueLabels = Array.from(await batch.ys.argMax(1).data());

        const preview = document.getElementById('preview');
        preview.innerHTML = '';

        for (let i = 0; i < 5; i++) {
            const container = document.createElement('div');
            container.className = 'pitem';
            
            const canvas = document.createElement('canvas');
            const image = tf.tidy(() => batch.xs.slice([i, 0, 0, 0], [1, 28, 28, 1]));
            this.data.draw28x28ToCanvas(image, canvas);
            image.dispose();
            
            const label = document.createElement('div');
            const correct = predLabels[i] === trueLabels[i];
            label.innerHTML = `Pred: <b>${predLabels[i]}</b> | True: <b>${trueLabels[i]}</b>`;
            label.className = correct ? 'ok' : 'bad';
            
            container.appendChild(canvas);
            container.appendChild(label);
            preview.appendChild(container);
        }

        predictions.dispose();
        batch.xs.dispose();
        batch.ys.dispose();
        
        this.log('Displayed 5 random classifier predictions');
    }

    async testFiveDenoising() {
        if (!this.autoencoder) {
            alert('Please train an autoencoder first');
            return;
        }

        const batch = this.data.getRandomDenoisingBatch(this.testData.xs, 5);
        const denoised = this.autoencoder.predict(batch.noisy);

        const preview = document.getElementById('preview');
        preview.innerHTML = '<div style="grid-column: 1 / -1; text-align: center; margin-bottom: 10px; color: #94a3b8;">Original | Noisy | Denoised</div>';

        for (let i = 0; i < 5; i++) {
            const container = document.createElement('div');
            container.className = 'pitem';
            
            // Original
            const origCanvas = document.createElement('canvas');
            const origImage = tf.tidy(() => batch.original.slice([i, 0, 0, 0], [1, 28, 28, 1]));
            this.data.draw28x28ToCanvas(origImage, origCanvas);
            origImage.dispose();
            
            // Noisy
            const noisyCanvas = document.createElement('canvas');
            const noisyImage = tf.tidy(() => batch.noisy.slice([i, 0, 0, 0], [1, 28, 28, 1]));
            this.data.draw28x28ToCanvas(noisyImage, noisyCanvas);
            noisyImage.dispose();
            
            // Denoised
            const denoisedCanvas = document.createElement('canvas');
            const denoisedImage = tf.tidy(() => denoised.slice([i, 0, 0, 0], [1, 28, 28, 1]));
            this.data.draw28x28ToCanvas(denoisedImage, denoisedCanvas);
            denoisedImage.dispose();
            
            container.appendChild(origCanvas);
            container.appendChild(noisyCanvas);
            container.appendChild(denoisedCanvas);
            preview.appendChild(container);
        }

        batch.original.dispose();
        batch.noisy.dispose();
        denoised.dispose();
        
        this.log('Displayed 5 random denoising examples');
    }

    async saveModel() {
        try {
            if (this.currentMode === 'classifier' && this.model) {
                await this.model.save('downloads://mnist-classifier');
                this.log('Classifier model saved');
            } else if (this.currentMode === 'autoencoder' && this.autoencoder) {
                await this.autoencoder.save('downloads://mnist-autoencoder');
                this.log('Autoencoder model saved');
            } else {
                alert('No model to save');
            }
        } catch (error) {
            this.log(`Save error: ${error.message}`);
        }
    }

    async loadModel() {
        try {
            const jsonFile = document.getElementById('modelJson').files[0];
            const weightsFile = document.getElementById('modelWeights').files[0];
            
            if (!jsonFile || !weightsFile) {
                alert('Please select both model.json and weights.bin files');
                return;
            }

            const model = await tf.loadLayersModel(tf.io.browserFiles([jsonFile, weightsFile]));
            
            if (this.currentMode === 'classifier') {
                if (this.model) this.model.dispose();
                this.model = model;
                this.log('Classifier model loaded');
            } else {
                if (this.autoencoder) this.autoencoder.dispose();
                this.autoencoder = model;
                this.log('Autoencoder model loaded');
            }
            
            document.getElementById('evaluate').disabled = false;
            document.getElementById('testFive').disabled = false;
            document.getElementById('saveModel').disabled = false;
            
        } catch (error) {
            this.log(`Load error: ${error.message}`);
        }
    }

    reset() {
        if (this.model) this.model.dispose();
        if (this.autoencoder) this.autoencoder.dispose();
        if (this.trainData) this.data.dispose();
        if (this.testData) this.data.dispose();
        
        this.model = null;
        this.autoencoder = null;
        this.trainData = null;
        this.testData = null;
        
        document.getElementById('dataStatus').innerHTML = 'No data loaded';
        document.getElementById('metrics').innerHTML = 'No metrics yet';
        document.getElementById('preview').innerHTML = '';
        document.getElementById('train').disabled = true;
        document.getElementById('evaluate').disabled = true;
        document.getElementById('testFive').disabled = true;
        document.getElementById('saveModel').disabled = true;
        
        this.log('Reset complete');
    }
}

// Initialize app when page loads
window.addEventListener('DOMContentLoaded', () => {
    window.app = new MNISTApp();
});
