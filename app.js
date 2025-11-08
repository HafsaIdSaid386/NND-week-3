// app.js
/**
 * Main application for MNIST CNN training and evaluation
 * Handles UI interactions, model management, and training workflow
 */

class MNISTApp {
    constructor() {
        this.dataLoader = new DataLoader();
        this.model = null;
        this.trainData = null;
        this.testData = null;
        this.isTraining = false;
        this.currentEpoch = 0;
        
        this.initializeUI();
        this.log('Application initialized. Please load MNIST CSV files.');
    }

    /**
     * Initialize UI event listeners
     */
    initializeUI() {
        // File inputs
        document.getElementById('loadData').addEventListener('click', () => this.onLoadData());
        document.getElementById('train').addEventListener('click', () => this.onTrain());
        document.getElementById('evaluate').addEventListener('click', () => this.onEvaluate());
        document.getElementById('testFive').addEventListener('click', () => this.onTestFive());
        document.getElementById('saveModel').addEventListener('click', () => this.onSaveDownload());
        document.getElementById('loadModel').addEventListener('click', () => this.onLoadFromFiles());
        document.getElementById('reset').addEventListener('click', () => this.onReset());
        document.getElementById('toggleVisor').addEventListener('click', () => this.toggleVisor());
    }

    /**
     * Load data from uploaded CSV files
     */
    async onLoadData() {
        try {
            const trainFile = document.getElementById('trainFile').files[0];
            const testFile = document.getElementById('testFile').files[0];
            
            if (!trainFile || !testFile) {
                alert('Please select both train and test CSV files.');
                return;
            }

            this.log('Loading training data...');
            this.trainData = await this.dataLoader.loadTrainFromFiles(trainFile);
            
            this.log('Loading test data...');
            this.testData = await this.dataLoader.loadTestFromFiles(testFile);
            
            // Update UI
            document.getElementById('train').disabled = false;
            document.getElementById('evaluate').disabled = false;
            document.getElementById('testFive').disabled = false;
            
            const trainSamples = this.trainData.xs.shape[0];
            const testSamples = this.testData.xs.shape[0];
            
            document.getElementById('dataStatus').innerHTML = `
                Train samples: ${trainSamples}<br>
                Test samples: ${testSamples}
            `;
            
            this.log('Data loaded successfully!');
            
        } catch (error) {
            this.log(`Error loading data: ${error.message}`);
            console.error(error);
        }
    }

    /**
     * Create and train CNN model
     */
    async onTrain() {
        if (!this.trainData) {
            alert('Please load data first.');
            return;
        }

        try {
            this.isTraining = true;
            this.currentEpoch = 0;
            
            // Create model
            this.model = this.createModel();
            this.log('Model created successfully.');
            
            // Split training data
            const { trainXs, trainYs, valXs, valYs } = this.dataLoader.splitTrainVal(
                this.trainData.xs, this.trainData.ys, 0.1
            );
            
            // Compile model
            this.model.compile({
                optimizer: 'adam',
                loss: 'categoricalCrossentropy',
                metrics: ['accuracy']
            });
            
            // Setup callbacks for visualization
            const surface = { name: 'Model Training', tab: 'Training' };
            const fitCallbacks = tfvis.show.fitCallbacks(surface, ['loss', 'val_loss', 'acc', 'val_acc'], {
                callbacks: ['onEpochEnd', 'onBatchEnd']
            });
            
            // Train model
            this.log('Starting training...');
            const startTime = Date.now();
            
            await this.model.fit(trainXs, trainYs, {
                epochs: 10,
                batchSize: 128,
                validationData: [valXs, valYs],
                shuffle: true,
                callbacks: fitCallbacks
            });
            
            const trainingTime = ((Date.now() - startTime) / 1000).toFixed(2);
            this.log(`Training completed in ${trainingTime} seconds.`);
            
            // Enable save button
            document.getElementById('saveModel').disabled = false;
            
            // Cleanup
            trainXs.dispose();
            trainYs.dispose();
            valXs.dispose();
            valYs.dispose();
            
        } catch (error) {
            this.log(`Training error: ${error.message}`);
            console.error(error);
        } finally {
            this.isTraining = false;
        }
    }

    /**
     * Create CNN model architecture
     * @returns {tf.LayersModel} Compiled CNN model
     */
    createModel() {
        const model = tf.sequential();
        
        // First convolutional block
        model.add(tf.layers.conv2d({
            filters: 32,
            kernelSize: 3,
            activation: 'relu',
            padding: 'same',
            inputShape: [28, 28, 1]
        }));
        
        // Second convolutional block
        model.add(tf.layers.conv2d({
            filters: 64,
            kernelSize: 3,
            activation: 'relu',
            padding: 'same'
        }));
        
        // Pooling and dropout
        model.add(tf.layers.maxPooling2d({ poolSize: 2 }));
        model.add(tf.layers.dropout({ rate: 0.25 }));
        
        // Flatten for dense layers
        model.add(tf.layers.flatten());
        
        // Dense layers
        model.add(tf.layers.dense({ units: 128, activation: 'relu' }));
        model.add(tf.layers.dropout({ rate: 0.5 }));
        model.add(tf.layers.dense({ units: 10, activation: 'softmax' }));
        
        // Display model summary
        model.summary();
        
        // Update model info
        const totalParams = model.countParams();
        const layers = model.layers.length;
        document.getElementById('modelInfo').innerHTML = `
            Layers: ${layers}<br>
            Total parameters: ${totalParams.toLocaleString()}
        `;
        
        return model;
    }

    /**
     * Evaluate model on test data
     */
    async onEvaluate() {
        if (!this.model || !this.testData) {
            alert('Please train or load a model and ensure test data is available.');
            return;
        }

        try {
            this.log('Evaluating model...');
            
            // Calculate test accuracy
            const evaluation = this.model.evaluate(this.testData.xs, this.testData.ys);
            const testLoss = evaluation[0].dataSync()[0];
            const testAcc = evaluation[1].dataSync()[0];
            
            // Update metrics
            document.getElementById('metrics').innerHTML = `
                Test Accuracy: ${(testAcc * 100).toFixed(2)}%<br>
                Test Loss: ${testLoss.toFixed(4)}
            `;
            
            this.log(`Test accuracy: ${(testAcc * 100).toFixed(2)}%`);
            
            // Generate predictions for confusion matrix
            const predictions = this.model.predict(this.testData.xs);
            const predLabels = predictions.argMax(-1);
            const trueLabels = this.testData.ys.argMax(-1);
            
            // Create confusion matrix
            const confusionMatrix = await tfvis.metrics.confusionMatrix(trueLabels, predLabels);
            
            // Visualize results
            const metricsSurface = { name: 'Model Evaluation', tab: 'Evaluation' };
            
            tfvis.show.perClassAccuracy(metricsSurface, confusionMatrix, ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']);
            tfvis.render.confusionMatrix(metricsSurface, { values: confusionMatrix }, {
                width: 400,
                height: 400
            });
            
            // Cleanup
            predictions.dispose();
            predLabels.dispose();
            trueLabels.dispose();
            evaluation[0].dispose();
            evaluation[1].dispose();
            
        } catch (error) {
            this.log(`Evaluation error: ${error.message}`);
            console.error(error);
        }
    }

    /**
     * Test 5 random samples and display predictions
     */
    async onTestFive() {
        if (!this.model || !this.testData) {
            alert('Please train or load a model and ensure test data is available.');
            return;
        }

        try {
            const batch = this.dataLoader.getRandomTestBatch(this.testData.xs, this.testData.ys, 5);
            const predictions = this.model.predict(batch.xs);
            const predLabels = predictions.argMax(-1).dataSync();
            const trueLabels = batch.ys.argMax(-1).dataSync();
            
            // Display preview
            const previewContainer = document.getElementById('preview');
            previewContainer.innerHTML = '';
            
            for (let i = 0; i < 5; i++) {
                const previewItem = document.createElement('div');
                previewItem.className = 'preview-item';
                
                const canvas = document.createElement('canvas');
                this.dataLoader.draw28x28ToCanvas(batch.xs.slice([i, 0, 0, 0], [1, 28, 28, 1]), canvas);
                
                const predLabel = document.createElement('div');
                predLabel.innerHTML = `<strong>Pred:</strong> ${predLabels[i]} | <strong>True:</strong> ${trueLabels[i]}`;
                predLabel.className = predLabels[i] === trueLabels[i] ? 'correct' : 'incorrect';
                
                previewItem.appendChild(canvas);
                previewItem.appendChild(predLabel);
                previewContainer.appendChild(previewItem);
            }
            
            // Cleanup
            predictions.dispose();
            batch.xs.dispose();
            batch.ys.dispose();
            
        } catch (error) {
            this.log(`Test preview error: ${error.message}`);
            console.error(error);
        }
    }

    /**
     * Save model to downloadable files
     */
    async onSaveDownload() {
        if (!this.model) {
            alert('No model to save. Please train a model first.');
            return;
        }

        try {
            await this.model.save('downloads://mnist-cnn');
            this.log('Model saved successfully. Check your downloads folder.');
        } catch (error) {
            this.log(`Model save error: ${error.message}`);
            console.error(error);
        }
    }

    /**
     * Load model from uploaded files
     */
    async onLoadFromFiles() {
        try {
            const jsonFile = document.getElementById('modelJson').files[0];
            const weightsFile = document.getElementById('modelWeights').files[0];
            
            if (!jsonFile || !weightsFile) {
                alert('Please select both model JSON and weights files.');
                return;
            }

            this.log('Loading model...');
            
            this.model = await tf.loadLayersModel(tf.io.browserFiles([jsonFile, weightsFile]));
            
            // Update UI
            document.getElementById('evaluate').disabled = false;
            document.getElementById('testFive').disabled = false;
            document.getElementById('saveModel').disabled = false;
            
            // Display model info
            this.model.summary();
            const totalParams = this.model.countParams();
            const layers = this.model.layers.length;
            
            document.getElementById('modelInfo').innerHTML = `
                Layers: ${layers}<br>
                Total parameters: ${totalParams.toLocaleString()}
            `;
            
            this.log('Model loaded successfully!');
            
        } catch (error) {
            this.log(`Model load error: ${error.message}`);
            console.error(error);
        }
    }

    /**
     * Reset application state
     */
    onReset() {
        // Cleanup tensors and model
        if (this.model) {
            this.model.dispose();
            this.model = null;
        }
        
        this.dataLoader.dispose();
        this.trainData = null;
        this.testData = null;
        
        // Reset UI
        document.getElementById('train').disabled = true;
        document.getElementById('evaluate').disabled = true;
        document.getElementById('testFive').disabled = true;
        document.getElementById('saveModel').disabled = true;
        
        document.getElementById('dataStatus').innerHTML = 'No data loaded';
        document.getElementById('modelInfo').innerHTML = 'No model loaded';
        document.getElementById('metrics').innerHTML = 'No metrics available';
        document.getElementById('preview').innerHTML = '';
        
        this.log('Application reset.');
    }

    /**
     * Toggle tfjs-vis visor
     */
    toggleVisor() {
        tfvis.visor().toggle();
    }

    /**
     * Log message to UI
     * @param {string} message - Message to log
     */
    log(message) {
        const logsElement = document.getElementById('logs');
        const timestamp = new Date().toLocaleTimeString();
        const logEntry = document.createElement('div');
        logEntry.textContent = `[${timestamp}] ${message}`;
        logsElement.appendChild(logEntry);
        logsElement.scrollTop = logsElement.scrollHeight;
    }
}

// Initialize application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.mnistApp = new MNISTApp();
});
