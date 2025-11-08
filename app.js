// Set CPU backend for consistent performance
tf.setBackend('cpu').then(() => {
    console.log('TensorFlow.js using CPU backend');
});

/**
 * Main application class for MNIST Denoising Autoencoder
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
        const logEntry = document.createElement('div');
        logEntry.className = 'log-entry';
        logEntry.innerHTML = `[${timestamp}] ${message}`;
        this.elements.trainingLogs.appendChild(logEntry);
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
                <strong>Data Loaded Successfully!</strong><br>
                Train samples: ${trainSamples}<br>
                Test samples: ${testSamples}<br>
                Ready for denoising autoencoder training.
            `;
            
            this.log(`Data loaded! Train: ${trainSamples}, Test: ${testSamples} samples`);
            this.updateUIState();
            
        } catch (error) {
            this.log(`ERROR: ${error.message}`);
            alert(`Failed to load data: ${error.message}\n\nPlease ensure:\n- Files are proper MNIST CSV format\n- Each row: label (0-9) + 784 pixel values (0-255)\n- No headers in CSV\n- Files are not corrupted`);
        }
    }

    /**
     * Add random Gaussian noise to images
     */
    addNoise(images, noiseFactor = 0.5) {
        return tf.tidy(() => {
            const noise = tf.randomNormal(images.shape, 0, noiseFactor);
            const noisyImages = images.add(noise);
            return noisyImages.clipByValue(0, 1);
        });
    }

    /**
     * Create CNN Autoencoder for denoising
     */
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
                
                // Output - reconstruct original image
                tf.layers.conv2d({
                    filters: 1,
                    kernelSize: 3,
                    activation: 'sigmoid',
                    padding: 'same'
                })
            ]
        });
        
        model.compile({
            optimizer: tf.train.adam(0.001),
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
            
            // Use smaller subset for faster training
            const numSamples = Math.min(1000, this.dataLoader.trainData.xs.shape[0]);
            this.log(`Using ${numSamples} samples for training`);
            
            // Prepare training data
            const trainingData = tf.tidy(() => {
                const indices = tf.util.createShuffledIndices(numSamples);
                const cleanImages = tf.gather(this.dataLoader.trainData.xs, indices);
                const noisyImages = this.addNoise(cleanImages, 0.5);
                
                return { noisyImages, cleanImages };
            });
            
            this.log('Training model (noisy → clean reconstruction)...');
            
            // Train the model
            const history = await this.model.fit(
                trainingData.noisyImages,
                trainingData.cleanImages,
                {
                    epochs: 10,
                    batchSize: 32,
                    validationSplit: 0.2,
                    shuffle: true,
                    callbacks: {
                        onEpochEnd: (epoch, logs) => {
                            this.log(`Epoch ${epoch + 1}/10 - Loss: ${logs.loss.toFixed(4)}, Val Loss: ${logs.val_loss ? logs.val_loss.toFixed(4) : 'N/A'}`);
                        }
                    }
                }
            );
            
            // Clean up
            trainingData.noisyImages.dispose();
            trainingData.cleanImages.dispose();
            
            this.log('Training completed successfully!');
            this.log('Model is ready for denoising tasks.');
            
        } catch (error) {
            this.log(`Training error: ${error.message}`);
            console.error('Training error details:', error);
        } finally {
            this.isTraining = false;
            this.updateUIState();
        }
    }

    async onEvaluate() {
        if (!this.model || !this.dataLoader.testData) return;
        
        try {
            this.log('Evaluating denoising performance...');
            
            const testSize = Math.min(200, this.dataLoader.testData.xs.shape[0]);
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
                Mean Squared Error: ${mseValue.toFixed(6)}<br>
                <small>Lower is better - measures reconstruction quality</small>
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
            this.elements.predictionResults.innerHTML = '<h4>Denoising Results Comparison</h4>';
            
            for (let i = 0; i < 5; i++) {
                const row = document.createElement('div');
                row.style.display = 'flex';
                row.style.gap = '15px';
                row.style.marginBottom = '20px';
                row.style.alignItems = 'center';
                row.style.justifyContent = 'center';
                
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
            
            this.log('Displayed denoising comparison');
            
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
            this.log('Model saved successfully! Check your downloads folder.');
        } catch (error) {
            this.log(`Save error: ${error.message}`);
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
                optimizer: tf.train.adam(0.001),
                loss: 'meanSquaredError',
                metrics: ['mse']
            });
            
            this.updateModelInfo();
            this.updateUIState();
            this.log('Model loaded successfully!');
            
        } catch (error) {
            this.log(`Load error: ${error.message}`);
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

// Initialize application
document.addEventListener('DOMContentLoaded', () => {
    new MNISTApp();
});
