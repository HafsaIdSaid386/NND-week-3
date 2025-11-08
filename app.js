// Set CPU backend for reliability
tf.setBackend('cpu').then(() => {
    console.log('TensorFlow.js ready - Using CPU backend');
});

/**
 * Main MNIST Denoising Autoencoder Application
 */
class MNISTApp {
    constructor() {
        this.dataLoader = new DataLoader();
        this.model = null;
        this.isTraining = false;
        this.initializeUI();
        this.bindEvents();
    }

    initializeUI() {
        this.elements = {
            trainFile: document.getElementById('trainFile'),
            testFile: document.getElementById('testFile'),
            modelJson: document.getElementById('modelJson'),
            modelWeights: document.getElementById('modelWeights'),
            loadData: document.getElementById('loadData'),
            useTestData: document.getElementById('useTestData'),
            train: document.getElementById('train'),
            evaluate: document.getElementById('evaluate'),
            testFive: document.getElementById('testFive'),
            saveModel: document.getElementById('saveModel'),
            loadModel: document.getElementById('loadModel'),
            reset: document.getElementById('reset'),
            toggleVisor: document.getElementById('toggleVisor'),
            dataStatus: document.getElementById('dataStatus'),
            modelInfo: document.getElementById('modelInfo'),
            trainingLogs: document.getElementById('trainingLogs'),
            imagePreview: document.getElementById('imagePreview'),
            predictionResults: document.getElementById('predictionResults'),
            metrics: document.getElementById('metrics')
        };
        
        this.updateUIState();
    }

    bindEvents() {
        this.elements.loadData.addEventListener('click', () => this.onLoadData());
        this.elements.useTestData.addEventListener('click', () => this.onUseTestData());
        this.elements.train.addEventListener('click', () => this.onTrain());
        this.elements.evaluate.addEventListener('click', () => this.onEvaluate());
        this.elements.testFive.addEventListener('click', () => this.onTestFive());
        this.elements.saveModel.addEventListener('click', () => this.onSaveDownload());
        this.elements.loadModel.addEventListener('click', () => this.onLoadFromFiles());
        this.elements.reset.addEventListener('click', () => this.onReset());
        this.elements.toggleVisor.addEventListener('click', () => this.toggleVisor());
        
        this.elements.modelJson.addEventListener('change', () => this.checkModelFiles());
        this.elements.modelWeights.addEventListener('change', () => this.checkModelFiles());
    }

    checkModelFiles() {
        const hasJson = this.elements.modelJson.files.length > 0;
        const hasWeights = this.elements.modelWeights.files.length > 0;
        this.elements.loadModel.disabled = !(hasJson && hasWeights);
    }

    updateUIState() {
        const hasData = this.dataLoader.trainData && this.dataLoader.testData;
        const hasModel = this.model !== null;
        
        this.elements.train.disabled = !hasData || this.isTraining;
        this.elements.evaluate.disabled = !hasModel || !hasData;
        this.elements.testFive.disabled = !hasModel || !hasData;
        this.elements.saveModel.disabled = !hasModel;
    }

    log(message) {
        const timestamp = new Date().toLocaleTimeString();
        this.elements.trainingLogs.innerHTML += `[${timestamp}] ${message}<br>`;
        this.elements.trainingLogs.scrollTop = this.elements.trainingLogs.scrollHeight;
        console.log(message);
    }

    async onLoadData() {
        try {
            const trainFile = this.elements.trainFile.files[0];
            const testFile = this.elements.testFile.files[0];
            
            if (!trainFile || !testFile) {
                alert('Please select both train and test CSV files');
                return;
            }
            
            this.log('Loading training data...');
            await this.dataLoader.loadTrainFromFiles(trainFile);
            
            this.log('Loading test data...');
            await this.dataLoader.loadTestFromFiles(testFile);
            
            const trainSamples = this.dataLoader.trainData.xs.shape[0];
            const testSamples = this.dataLoader.testData.xs.shape[0];
            
            this.elements.dataStatus.innerHTML = `
                <span class="success">✓ Data Loaded Successfully!</span><br>
                Train samples: ${trainSamples}<br>
                Test samples: ${testSamples}<br>
                <em>Click "Train Model" to start training</em>
            `;
            
            this.log(`SUCCESS: Loaded ${trainSamples} training and ${testSamples} test samples`);
            this.updateUIState();
            
        } catch (error) {
            this.log(`ERROR: ${error.message}`);
            this.elements.dataStatus.innerHTML = `
                <span class="error">✗ Failed to load data</span><br>
                Error: ${error.message}<br>
                <button onclick="app.onUseTestData()" class="test-btn">Use Test Data Instead</button>
            `;
        }
    }

    async onUseTestData() {
        try {
            this.log('Creating synthetic test data...');
            
            this.dataLoader.trainData = this.dataLoader.createTestData();
            this.dataLoader.testData = this.dataLoader.createTestData();
            
            const trainSamples = this.dataLoader.trainData.xs.shape[0];
            const testSamples = this.dataLoader.testData.xs.shape[0];
            
            this.elements.dataStatus.innerHTML = `
                <span class="success">✓ Using Synthetic Test Data</span><br>
                Train samples: ${trainSamples}<br>
                Test samples: ${testSamples}<br>
                <em>You can now test the application</em>
            `;
            
            this.log(`SUCCESS: Created ${trainSamples} synthetic samples`);
            this.updateUIState();
            
        } catch (error) {
            this.log(`ERROR creating test data: ${error.message}`);
        }
    }

    addNoise(images, noiseFactor = 0.5) {
        return tf.tidy(() => {
            const noise = tf.randomNormal(images.shape, 0, noiseFactor);
            return images.add(noise).clipByValue(0, 1);
        });
    }

    createDenoisingAutoencoder() {
        const model = tf.sequential({
            layers: [
                // Encoder
                tf.layers.conv2d({
                    inputShape: [28, 28, 1],
                    filters: 32,
                    kernelSize: 3,
                    activation: 'relu',
                    padding: 'same'
                }),
                tf.layers.conv2d({
                    filters: 16,
                    kernelSize: 3,
                    activation: 'relu',
                    padding: 'same'
                }),
                
                // Decoder
                tf.layers.conv2d({
                    filters: 16,
                    kernelSize: 3,
                    activation: 'relu',
                    padding: 'same'
                }),
                tf.layers.conv2d({
                    filters: 32,
                    kernelSize: 3,
                    activation: 'relu',
                    padding: 'same'
                }),
                
                // Output
                tf.layers.conv2d({
                    filters: 1,
                    kernelSize: 3,
                    activation: 'sigmoid',
                    padding: 'same'
                })
            ]
        });
        
        model.compile({
            optimizer: 'adam',
            loss: 'meanSquaredError',
            metrics: ['mse']
        });
        
        return model;
    }

    async onTrain() {
        if (this.isTraining) return;
        
        try {
            this.isTraining = true;
            this.updateUIState();
            this.log('Starting denoising autoencoder training...');
            
            // Create model
            this.model = this.createDenoisingAutoencoder();
            this.updateModelInfo();
            
            // Use reasonable sample size
            const numSamples = Math.min(500, this.dataLoader.trainData.xs.shape[0]);
            this.log(`Training on ${numSamples} samples...`);
            
            const trainingData = tf.tidy(() => {
                const indices = Array.from({length: numSamples}, (_, i) => i);
                const cleanImages = tf.gather(this.dataLoader.trainData.xs, indices);
                const noisyImages = this.addNoise(cleanImages, 0.5);
                return { noisyImages, cleanImages };
            });
            
            // Train the model
            await this.model.fit(
                trainingData.noisyImages,
                trainingData.cleanImages,
                {
                    epochs: 8,
                    batchSize: 32,
                    validationSplit: 0.2,
                    shuffle: true,
                    callbacks: {
                        onEpochEnd: (epoch, logs) => {
                            const progress = ((epoch + 1) / 8 * 100).toFixed(0);
                            this.log(`Epoch ${epoch + 1}/8 (${progress}%) - Loss: ${logs.loss.toFixed(4)}`);
                        }
                    }
                }
            );
            
            trainingData.noisyImages.dispose();
            trainingData.cleanImages.dispose();
            
            this.log('🎉 Training completed successfully!');
            this.log('Model is ready for denoising tasks.');
            
        } catch (error) {
            this.log(`❌ Training error: ${error.message}`);
            console.error('Training error:', error);
        } finally {
            this.isTraining = false;
            this.updateUIState();
        }
    }

    async onEvaluate() {
        if (!this.model || !this.dataLoader.testData) return;
        
        try {
            this.log('Evaluating denoising performance...');
            
            const testSize = Math.min(100, this.dataLoader.testData.xs.shape[0]);
            const testSubset = tf.tidy(() => {
                const indices = Array.from({length: testSize}, (_, i) => i);
                return tf.gather(this.dataLoader.testData.xs, indices);
            });
            
            const noisyTest = this.addNoise(testSubset, 0.5);
            const denoised = this.model.predict(noisyTest);
            
            const mse = tf.losses.meanSquaredError(testSubset, denoised);
            const mseValue = (await mse.data())[0];
            
            this.elements.metrics.innerHTML = `
                <strong>Denoising Performance</strong><br>
                Mean Squared Error: <strong>${mseValue.toFixed(6)}</strong><br>
                <small>Lower values indicate better reconstruction</small>
            `;
            
            this.log(`Evaluation completed - MSE: ${mseValue.toFixed(6)}`);
            
            testSubset.dispose();
            noisyTest.dispose();
            denoised.dispose();
            mse.dispose();
            
        } catch (error) {
            this.log(`Evaluation error: ${error.message}`);
        }
    }

    async onTestFive() {
        if (!this.model || !this.dataLoader.testData) return;
        
        try {
            this.log('Testing denoising on 5 random images...');
            
            const batch = this.dataLoader.getRandomTestBatch(
                this.dataLoader.testData.xs,
                this.dataLoader.testData.ys,
                5
            );
            
            const noisyImages = this.addNoise(batch.xs, 0.5);
            const denoisedImages = this.model.predict(noisyImages);
            
            this.elements.imagePreview.innerHTML = '';
            this.elements.predictionResults.innerHTML = '<h4>Denoising Results</h4>';
            
            for (let i = 0; i < 5; i++) {
                const row = document.createElement('div');
                row.style.display = 'flex';
                row.style.gap = '15px';
                row.style.marginBottom: '20px';
                row.style.alignItems: 'center';
                row.style.justifyContent: 'center';
                
                const originalImg = this.createImageWithLabel(
                    tf.slice(batch.xs, [i, 0, 0, 0], [1, 28, 28, 1]),
                    'Original'
                );
                
                const noisyImg = this.createImageWithLabel(
                    tf.slice(noisyImages, [i, 0, 0, 0], [1, 28, 28, 1]),
                    'Noisy'
                );
                
                const denoisedImg = this.createImageWithLabel(
                    tf.slice(denoisedImages, [i, 0, 0, 0], [1, 28, 28, 1]),
                    'Denoised'
                );
                
                row.appendChild(originalImg);
                row.appendChild(noisyImg);
                row.appendChild(denoisedImg);
                this.elements.imagePreview.appendChild(row);
            }
            
            this.log('Displayed denoising comparison for 5 images');
            
            tf.dispose([batch.xs, batch.ys, noisyImages, denoisedImages]);
            
        } catch (error) {
            this.log(`Test error: ${error.message}`);
        }
    }

    createImageWithLabel(tensor, label) {
        const container = document.createElement('div');
        container.style.display = 'flex';
        container.style.flexDirection = 'column';
        container.style.alignItems = 'center';
        
        const canvas = document.createElement('canvas');
        canvas.style.border = '2px solid #333';
        this.dataLoader.draw28x28ToCanvas(tensor, canvas, 4);
        
        const labelDiv = document.createElement('div');
        labelDiv.innerHTML = `<strong>${label}</strong>`;
        labelDiv.style.marginTop = '8px';
        labelDiv.style.fontWeight = 'bold';
        
        container.appendChild(canvas);
        container.appendChild(labelDiv);
        
        return container;
    }

    async onSaveDownload() {
        if (!this.model) return;
        
        try {
            this.log('Saving model...');
            await this.model.save('downloads://mnist-denoising-autoencoder');
            this.log('✅ Model saved successfully! Check your downloads folder.');
        } catch (error) {
            this.log(`❌ Save error: ${error.message}`);
        }
    }

    async onLoadFromFiles() {
        try {
            const jsonFile = this.elements.modelJson.files[0];
            const weightsFile = this.elements.modelWeights.files[0];
            
            if (!jsonFile || !weightsFile) {
                alert('Please select both model files');
                return;
            }
            
            this.log('Loading model...');
            this.model = await tf.loadLayersModel(tf.io.browserFiles([jsonFile, weightsFile]));
            
            this.model.compile({
                optimizer: 'adam',
                loss: 'meanSquaredError',
                metrics: ['mse']
            });
            
            this.updateModelInfo();
            this.updateUIState();
            this.log('✅ Model loaded successfully!');
            
        } catch (error) {
            this.log(`❌ Load error: ${error.message}`);
        }
    }

    updateModelInfo() {
        if (!this.model) return;
        
        let totalParams = 0;
        const layersInfo = this.model.layers.map(layer => {
            const params = layer.countParams();
            totalParams += params;
            return `${layer.name}: ${params} params (${layer.getClassName()})`;
        }).join('<br>');
        
        this.elements.modelInfo.innerHTML = `
            <strong>Denoising Autoencoder</strong><br>
            Layers: ${this.model.layers.length}<br>
            Total parameters: ${totalParams.toLocaleString()}<br>
            <details><summary>Layer Details</summary><div style="font-size: 12px;">${layersInfo}</div></details>
        `;
    }

    onReset() {
        if (this.model) {
            this.model.dispose();
            this.model = null;
        }
        this.dataLoader.dispose();
        
        this.elements.dataStatus.innerHTML = 'No data loaded';
        this.elements.modelInfo.innerHTML = 'No model loaded';
        this.elements.trainingLogs.innerHTML = '';
        this.elements.imagePreview.innerHTML = '';
        this.elements.predictionResults.innerHTML = '';
        this.elements.metrics.innerHTML = 'No evaluation performed';
        
        this.elements.trainFile.value = '';
        this.elements.testFile.value = '';
        this.elements.modelJson.value = '';
        this.elements.modelWeights.value = '';
        
        this.updateUIState();
        this.log('Application reset');
    }

    toggleVisor() {
        tfvis.visor().toggle();
    }
}

// Global reference for button callbacks
let app;

// Initialize application
document.addEventListener('DOMContentLoaded', () => {
    app = new MNISTApp();
});
