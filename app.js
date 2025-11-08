// Force TensorFlow.js to use CPU backend to avoid WebGL issues
tf.setBackend('cpu').then(() => {
    console.log('Using CPU backend');
});

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
                Train samples: ${trainSamples}<br>
                Test samples: ${testSamples}<br>
                <strong>Task: Denoising Autoencoder</strong>
            `;
            
            this.log('Data loaded successfully! Ready for denoising autoencoder training.');
            this.updateUIState();
            
        } catch (error) {
            this.log(`Error loading data: ${error.message}`);
            alert(`Failed to load data: ${error.message}`);
        }
    }

    /**
     * Add random noise to images (Step 1 of homework)
     */
    addNoise(images, noiseFactor = 0.5) {
        return tf.tidy(() => {
            // Add Gaussian noise
            const noise = tf.randomNormal(images.shape, 0, noiseFactor);
            const noisyImages = images.add(noise);
            // Clip values to [0, 1] range
            return noisyImages.clipByValue(0, 1);
        });
    }

    /**
     * Create CNN Autoencoder for denoising (Step 2 of homework)
     */
    createDenoisingAutoencoder() {
        const model = tf.sequential();
        
        // Encoder
        model.add(tf.layers.conv2d({
            inputShape: [28, 28, 1],
            filters: 32,
            kernelSize: 3,
            activation: 'relu',
            padding: 'same'
        }));
        model.add(tf.layers.maxPooling2d({poolSize: 2, padding: 'same'}));
        
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
        model.add(tf.layers.upSampling2d({size: 2}));
        
        model.add(tf.layers.conv2d({
            filters: 32,
            kernelSize: 3,
            activation: 'relu',
            padding: 'same'
        }));
        
        // Output layer - reconstruct original image
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
        
        return model;
    }

    async onTrain() {
        if (this.isTraining) return;
        
        try {
            this.isTraining = true;
            this.updateUIState();
            this.log('Starting DENOISING AUTOENCODER training...');
            this.log('Step 1: Adding noise to training data');
            
            // Create denoising autoencoder
            this.model = this.createDenoisingAutoencoder();
            this.updateModelInfo();
            
            // Use smaller subset for faster training
            const numSamples = Math.min(2000, this.dataLoader.trainData.xs.shape[0]);
            const trainSubset = tf.tidy(() => {
                const indices = Array.from({length: numSamples}, (_, i) => i);
                const cleanImages = tf.gather(this.dataLoader.trainData.xs, indices);
                return cleanImages;
            });
            
            this.log(`Step 2: Training autoencoder on ${numSamples} samples`);
            
            let currentEpoch = 0;
            
            // Train autoencoder
            const history = await this.model.fit(
                trainSubset,  // Input: clean images
                trainSubset,  // Target: same clean images (autoencoder)
                {
                    epochs: 5,
                    batchSize: 32,
                    validationSplit: 0.1,
                    shuffle: true,
                    callbacks: {
                        onEpochBegin: (epoch) => {
                            currentEpoch = epoch;
                            this.log(`Epoch ${epoch + 1}: Training denoiser...`);
                        },
                        onEpochEnd: (epoch, logs) => {
                            this.log(`Epoch ${epoch + 1} completed - Loss: ${logs.loss.toFixed(4)}, Val Loss: ${logs.val_loss ? logs.val_loss.toFixed(4) : 'N/A'}`);
                        }
                    }
                }
            );
            
            // Clean up
            trainSubset.dispose();
            
            this.log('Denoising autoencoder training completed!');
            this.log('Ready to test denoising on noisy images.');
            
        } catch (error) {
            this.log(`Training error: ${error.message}`);
            console.error('Detailed training error:', error);
            alert(`Training failed: ${error.message}`);
        } finally {
            this.isTraining = false;
            this.updateUIState();
        }
    }

    async onEvaluate() {
        if (!this.model || !this.dataLoader.testData) return;
        
        try {
            this.log('Evaluating denoising performance...');
            
            // Use smaller test subset
            const testSize = Math.min(100, this.dataLoader.testData.xs.shape[0]);
            const testSubset = tf.tidy(() => {
                const indices = Array.from({length: testSize}, (_, i) => i);
                return tf.gather(this.dataLoader.testData.xs, indices);
            });
            
            // Add noise to test images
            const noisyTest = this.addNoise(testSubset, 0.5);
            
            // Denoise them
            const denoised = this.model.predict(noisyTest);
            
            // Calculate reconstruction error
            const mse = tf.losses.meanSquaredError(testSubset, denoised);
            const mseValue = (await mse.data())[0];
            
            this.elements.metrics.innerHTML = `
                <strong>Denoising Performance:</strong><br>
                Mean Squared Error: ${mseValue.toFixed(4)}<br>
                Lower is better - measures reconstruction quality
            `;
            
            this.log(`Denoising evaluation completed! MSE: ${mseValue.toFixed(4)}`);
            
            // Clean up
            testSubset.dispose();
            noisyTest.dispose();
            denoised.dispose();
            mse.dispose();
            
        } catch (error) {
            this.log(`Evaluation error: ${error.message}`);
            alert(`Evaluation failed: ${error.message}`);
        }
    }

    /**
     * Modified Test 5 Random to show denoising results (Step 3 of homework)
     */
    async onTestFive() {
        if (!this.model || !this.dataLoader.testData) return;
        
        try {
            this.log('Testing denoising on 5 random images...');
            
            // Get random batch
            const batch = this.dataLoader.getRandomTestBatch(
                this.dataLoader.testData.xs,
                this.dataLoader.testData.ys,
                5
            );
            
            // Add noise to these images
            const noisyImages = this.addNoise(batch.xs, 0.6);
            
            // Denoise them
            const denoisedImages = this.model.predict(noisyImages);
            
            this.elements.imagePreview.innerHTML = '';
            this.elements.predictionResults.innerHTML = '<h3>Denoising Results (Original → Noisy → Denoised)</h3>';
            
            // Create comparison for each image
            for (let i = 0; i < 5; i++) {
                const comparisonContainer = document.createElement('div');
                comparisonContainer.style.display = 'flex';
                comparisonContainer.style.gap = '10px';
                comparisonContainer.style.marginBottom = '20px';
                comparisonContainer.style.alignItems = 'center';
                
                // Original image
                const originalContainer = this.createImageWithLabel(
                    tf.slice(batch.xs, [i, 0, 0, 0], [1, 28, 28, 1]),
                    'Original'
                );
                
                // Noisy image
                const noisyContainer = this.createImageWithLabel(
                    tf.slice(noisyImages, [i, 0, 0, 0], [1, 28, 28, 1]),
                    'Noisy'
                );
                
                // Denoised image
                const denoisedContainer = this.createImageWithLabel(
                    tf.slice(denoisedImages, [i, 0, 0, 0], [1, 28, 28, 1]),
                    'Denoised'
                );
                
                comparisonContainer.appendChild(originalContainer);
                comparisonContainer.appendChild(noisyContainer);
                comparisonContainer.appendChild(denoisedContainer);
                
                this.elements.imagePreview.appendChild(comparisonContainer);
            }
            
            this.log('Displayed denoising comparison for 5 images');
            
            // Clean up
            tf.dispose([batch.xs, batch.ys, noisyImages, denoisedImages]);
            
        } catch (error) {
            this.log(`Test error: ${error.message}`);
            alert(`Testing failed: ${error.message}`);
        }
    }

    createImageWithLabel(tensor, label) {
        const container = document.createElement('div');
        container.style.display = 'flex';
        container.style.flexDirection = 'column';
        container.style.alignItems = 'center';
        
        const canvas = document.createElement('canvas');
        this.dataLoader.draw28x28ToCanvas(tensor, canvas, 3);
        
        const labelDiv = document.createElement('div');
        labelDiv.innerHTML = `<strong>${label}</strong>`;
        labelDiv.style.marginTop = '5px';
        
        container.appendChild(canvas);
        container.appendChild(labelDiv);
        
        return container;
    }

    async onSaveDownload() {
        if (!this.model) return;
        
        try {
            this.log('Saving denoising autoencoder model...');
            await this.model.save('downloads://mnist-denoising-autoencoder');
            this.log('Model saved successfully! (Step 4 completed)');
        } catch (error) {
            this.log(`Save error: ${error.message}`);
            alert(`Failed to save model: ${error.message}`);
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
            
            this.log('Loading denoising autoencoder model...');
            
            this.model = await tf.loadLayersModel(tf.io.browserFiles([jsonFile, weightsFile]));
            
            this.model.compile({
                optimizer: 'adam',
                loss: 'meanSquaredError',
                metrics: ['mse']
            });
            
            this.updateModelInfo();
            this.updateUIState();
            this.log('Denoising autoencoder loaded successfully! (Step 4 completed)');
            
        } catch (error) {
            this.log(`Load error: ${error.message}`);
            alert(`Failed to load model: ${error.message}`);
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
            <details><summary>Layer Details</summary>${layersInfo}</details>
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

// Initialize application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new MNISTApp();
});
