// Force TensorFlow.js to use CPU backend to avoid WebGL issues
tf.setBackend('cpu').then(() => {
    console.log('Using CPU backend');
});

/**
 * Main application class for MNIST Denoising Autoencoder
 * Handles UI, model training, and denoising operations
 */
class MNISTApp {
    constructor() {
        this.dataLoader = new DataLoader();
        this.model = null;
        this.isTraining = false;
        this.initializeUI();
        this.bindEvents();
    }

    /**
     * Initialize UI elements reference
     */
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

    /**
     * Bind event listeners to UI elements
     */
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

    /**
     * Check if both model files are selected for loading
     */
    checkModelFiles() {
        const hasJson = this.elements.modelJson.files.length > 0;
        const hasWeights = this.elements.modelWeights.files.length > 0;
        this.elements.loadModel.disabled = !(hasJson && hasWeights);
    }

    /**
     * Update UI state based on current application state
     */
    updateUIState() {
        const hasData = this.dataLoader.trainData && this.dataLoader.testData;
        const hasModel = this.model !== null;
        
        this.elements.train.disabled = !hasData || this.isTraining;
        this.elements.evaluate.disabled = !hasModel || !hasData;
        this.elements.testFive.disabled = !hasModel || !hasData;
        this.elements.saveModel.disabled = !hasModel;
    }

    /**
     * Log messages to training logs and console
     * @param {string} message - Message to log
     */
    log(message) {
        const timestamp = new Date().toLocaleTimeString();
        this.elements.trainingLogs.innerHTML += `[${timestamp}] ${message}<br>`;
        this.elements.trainingLogs.scrollTop = this.elements.trainingLogs.scrollHeight;
        console.log(message);
    }

    /**
     * Load and process MNIST data from CSV files
     */
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
     * Add random Gaussian noise to images (Step 1 of homework)
     * @param {tf.Tensor} images - Clean images tensor
     * @param {number} noiseFactor - Noise intensity (0-1)
     * @returns {tf.Tensor} Noisy images
     */
    addNoise(images, noiseFactor = 0.5) {
        return tf.tidy(() => {
            // Add Gaussian noise with mean 0 and specified standard deviation
            const noise = tf.randomNormal(images.shape, 0, noiseFactor);
            const noisyImages = images.add(noise);
            // Clip values to maintain valid pixel range [0, 1]
            return noisyImages.clipByValue(0, 1);
        });
    }

    /**
     * Create CNN Autoencoder for denoising (Step 2 of homework)
     * Uses same padding throughout to maintain 28x28 dimensions
     */
    createDenoisingAutoencoder() {
        const model = tf.sequential();
        
        // Encoder - maintain spatial dimensions with 'same' padding
        model.add(tf.layers.conv2d({
            inputShape: [28, 28, 1],
            filters: 16,
            kernelSize: 3,
            activation: 'relu',
            padding: 'same'  // Maintain 28x28 dimensions
        }));
        
        model.add(tf.layers.conv2d({
            filters: 8,
            kernelSize: 3,
            activation: 'relu',
            padding: 'same'  // Maintain 28x28 dimensions
        }));
        
        // Decoder - maintain spatial dimensions with 'same' padding
        model.add(tf.layers.conv2d({
            filters: 8,
            kernelSize: 3,
            activation: 'relu',
            padding: 'same'  // Maintain 28x28 dimensions
        }));
        
        model.add(tf.layers.conv2d({
            filters: 16,
            kernelSize: 3,
            activation: 'relu',
            padding: 'same'  // Maintain 28x28 dimensions
        }));
        
        // Output layer - reconstruct original image with same shape
        model.add(tf.layers.conv2d({
            filters: 1,
            kernelSize: 3,
            activation: 'sigmoid',  // Sigmoid for pixel values in [0,1] range
            padding: 'same'  // Maintain 28x28 dimensions
        }));
        
        model.compile({
            optimizer: 'adam',
            loss: 'meanSquaredError',  // MSE loss for image reconstruction
            metrics: ['mse']
        });
        
        return model;
    }

    /**
     * Train the denoising autoencoder model
     */
    async onTrain() {
        if (this.isTraining) return;
        
        try {
            this.isTraining = true;
            this.updateUIState();
            this.log('Starting DENOISING AUTOENCODER training...');
            this.log('Step 1: Adding noise to training data');
            
            // Create denoising autoencoder model
            this.model = this.createDenoisingAutoencoder();
            this.updateModelInfo();
            
            // Use smaller subset for faster training
            const numSamples = Math.min(1000, this.dataLoader.trainData.xs.shape[0]);
            
            this.log(`Using ${numSamples} samples for training`);
            
            // Create training data: noisy inputs, clean targets
            const trainingData = tf.tidy(() => {
                const indices = Array.from({length: numSamples}, (_, i) => i);
                const cleanImages = tf.gather(this.dataLoader.trainData.xs, indices);
                const noisyImages = this.addNoise(cleanImages, 0.5);
                
                return { noisyImages, cleanImages };
            });
            
            // Log shapes for debugging
            this.log(`Input shape: ${JSON.stringify(trainingData.noisyImages.shape)}`);
            this.log(`Target shape: ${JSON.stringify(trainingData.cleanImages.shape)}`);
            
            this.log('Step 2: Training autoencoder (noisy → clean)');
            
            let currentEpoch = 0;
            
            // Train the autoencoder to reconstruct clean images from noisy ones
            const history = await this.model.fit(
                trainingData.noisyImages,  // Input: noisy images
                trainingData.cleanImages,  // Target: clean images
                {
                    epochs: 5,  // Reasonable number of epochs
                    batchSize: 32,
                    validationSplit: 0.1,
                    shuffle: true,
                    callbacks: {
                        onEpochBegin: (epoch) => {
                            currentEpoch = epoch;
                            this.log(`Epoch ${epoch + 1}: Training denoiser...`);
                        },
                        onEpochEnd: (epoch, logs) => {
                            this.log(`Epoch ${epoch + 1} completed - Loss: ${logs.loss.toFixed(4)}${logs.val_loss ? `, Val Loss: ${logs.val_loss.toFixed(4)}` : ''}`);
                        }
                    }
                }
            );
            
            // Clean up training tensors
            trainingData.noisyImages.dispose();
            trainingData.cleanImages.dispose();
            
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

    /**
     * Evaluate denoising performance on test data
     */
    async onEvaluate() {
        if (!this.model || !this.dataLoader.testData) return;
        
        try {
            this.log('Evaluating denoising performance...');
            
            // Use smaller test subset for evaluation
            const testSize = Math.min(100, this.dataLoader.testData.xs.shape[0]);
            const testSubset = tf.tidy(() => {
                const indices = Array.from({length: testSize}, (_, i) => i);
                return tf.gather(this.dataLoader.testData.xs, indices);
            });
            
            // Create noisy test images
            const noisyTest = this.addNoise(testSubset, 0.5);
            
            // Denoise using trained model
            const denoised = this.model.predict(noisyTest);
            
            // Calculate reconstruction error (Mean Squared Error)
            const mse = tf.losses.meanSquaredError(testSubset, denoised);
            const mseValue = (await mse.data())[0];
            
            this.elements.metrics.innerHTML = `
                <strong>Denoising Performance:</strong><br>
                Mean Squared Error: ${mseValue.toFixed(4)}<br>
                Lower is better - measures reconstruction quality
            `;
            
            this.log(`Denoising evaluation completed! MSE: ${mseValue.toFixed(4)}`);
            
            // Clean up tensors
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
     * Test denoising on 5 random images (Step 3 of homework)
     * Shows original, noisy, and denoised images for comparison
     */
    async onTestFive() {
        if (!this.model || !this.dataLoader.testData) return;
        
        try {
            this.log('Testing denoising on 5 random images...');
            
            // Get random batch of clean test images
            const batch = this.dataLoader.getRandomTestBatch(
                this.dataLoader.testData.xs,
                this.dataLoader.testData.ys,
                5
            );
            
            // Add noise to create corrupted inputs
            const noisyImages = this.addNoise(batch.xs, 0.6);
            
            // Denoise using trained autoencoder
            const denoisedImages = this.model.predict(noisyImages);
            
            // Clear previous results
            this.elements.imagePreview.innerHTML = '';
            this.elements.predictionResults.innerHTML = '<h3>Denoising Results (Original → Noisy → Denoised)</h3>';
            
            // Create visual comparison for each image
            for (let i = 0; i < 5; i++) {
                const comparisonContainer = document.createElement('div');
                comparisonContainer.style.display = 'flex';
                comparisonContainer.style.gap = '10px';
                comparisonContainer.style.marginBottom = '20px';
                comparisonContainer.style.alignItems = 'center';
                
                // Original clean image
                const originalContainer = this.createImageWithLabel(
                    tf.slice(batch.xs, [i, 0, 0, 0], [1, 28, 28, 1]),
                    'Original'
                );
                
                // Noisy input image
                const noisyContainer = this.createImageWithLabel(
                    tf.slice(noisyImages, [i, 0, 0, 0], [1, 28, 28, 1]),
                    'Noisy'
                );
                
                // Denoised output image
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
            
            // Clean up tensors
            tf.dispose([batch.xs, batch.ys, noisyImages, denoisedImages]);
            
        } catch (error) {
            this.log(`Test error: ${error.message}`);
            alert(`Testing failed: ${error.message}`);
        }
    }

    /**
     * Create image container with label for display
     * @param {tf.Tensor} tensor - Image tensor
     * @param {string} label - Display label
     * @returns {HTMLElement} Container with image and label
     */
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

    /**
     * Save trained model to files (Step 4 of homework)
     */
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

    /**
     * Load pre-trained model from files (Step 4 of homework)
     */
    async onLoadFromFiles() {
        try {
            const jsonFile = this.elements.modelJson.files[0];
            const weightsFile = this.elements.modelWeights.files[0];
            
            if (!jsonFile || !weightsFile) {
                alert('Please select both model files');
                return;
            }
            
            this.log('Loading denoising autoencoder model...');
            
            // Load model architecture and weights
            this.model = await tf.loadLayersModel(tf.io.browserFiles([jsonFile, weightsFile]));
            
            // Recompile the loaded model
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

    /**
     * Update model information display
     */
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

    /**
     * Reset application state and clear all data
     */
    onReset() {
        if (this.model) {
            this.model.dispose();
            this.model = null;
        }
        this.dataLoader.dispose();
        
        // Clear all UI elements
        this.elements.dataStatus.innerHTML = 'No data loaded';
        this.elements.modelInfo.innerHTML = 'No model loaded';
        this.elements.trainingLogs.innerHTML = '';
        this.elements.imagePreview.innerHTML = '';
        this.elements.predictionResults.innerHTML = '';
        this.elements.metrics.innerHTML = 'No evaluation performed';
        
        // Clear file inputs
        this.elements.trainFile.value = '';
        this.elements.testFile.value = '';
        this.elements.modelJson.value = '';
        this.elements.modelWeights.value = '';
        
        this.updateUIState();
        this.log('Application reset');
    }

    /**
     * Toggle TensorFlow.js visor for charts and metrics
     */
    toggleVisor() {
        tfvis.visor().toggle();
    }
}

// Initialize application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new MNISTApp();
});
