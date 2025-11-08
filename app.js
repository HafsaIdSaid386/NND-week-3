// app.js
/**
 * Main application for MNIST CNN classifier
 * Handles UI interactions, model training, and evaluation
 */

class MNISTApp {
    constructor() {
        this.dataLoader = new DataLoader();
        this.model = null;
        this.isTraining = false;
        this.currentEpoch = 0;
        
        this.initializeUI();
        this.log('Application initialized. Please load MNIST data files.');
    }

    /**
     * Initialize UI event listeners
     */
    initializeUI() {
        // File input handlers
        document.getElementById('loadData').addEventListener('click', () => this.onLoadData());
        document.getElementById('train').addEventListener('click', () => this.onTrain());
        document.getElementById('evaluate').addEventListener('click', () => this.onEvaluate());
        document.getElementById('testFive').addEventListener('click', () => this.onTestFive());
        document.getElementById('saveModel').addEventListener('click', () => this.onSaveDownload());
        document.getElementById('loadModel').addEventListener('click', () => this.onLoadFromFiles());
        document.getElementById('reset').addEventListener('click', () => this.onReset());
        document.getElementById('toggleVisor').addEventListener('click', () => this.toggleVisor());

        // Enable load model button when files are selected
        document.getElementById('modelJsonFile').addEventListener('change', () => this.checkModelFiles());
        document.getElementById('modelWeightsFile').addEventListener('change', () => this.checkModelFiles());
    }

    /**
     * Check if both model files are selected and enable load button
     */
    checkModelFiles() {
        const jsonFile = document.getElementById('modelJsonFile').files[0];
        const weightsFile = document.getElementById('modelWeightsFile').files[0];
        document.getElementById('loadModel').disabled = !(jsonFile && weightsFile);
    }

    /**
     * Load data from CSV files
     */
    async onLoadData() {
        try {
            const trainFile = document.getElementById('trainFile').files[0];
            const testFile = document.getElementById('testFile').files[0];
            
            if (!trainFile || !testFile) {
                alert('Please select both train and test CSV files');
                return;
            }

            this.log('Loading training data...');
            const trainData = await this.dataLoader.loadTrainFromFiles(trainFile);
            
            this.log('Loading test data...');
            const testData = await this.dataLoader.loadTestFromFiles(testFile);

            // Update data status
            document.getElementById('dataStatus').innerHTML = `
                Train samples: ${trainData.xs.shape[0]}<br>
                Test samples: ${testData.xs.shape[0]}
            `;

            // Enable buttons
            document.getElementById('train').disabled = false;
            document.getElementById('evaluate').disabled = false;
            document.getElementById('testFive').disabled = false;

            this.log('Data loaded successfully!');
            
        } catch (error) {
            this.log(`Error loading data: ${error.message}`);
            console.error('Data loading error:', error);
        }
    }

    /**
     * Create and train CNN model
     */
    async onTrain() {
        if (this.isTraining) {
            alert('Training already in progress');
            return;
        }

        try {
            this.isTraining = true;
            this.log('Starting model training...');

            // Create model
            this.model = this.createModel();
            this.updateModelInfo();

            // Split training data
            const { trainXs, trainYs, valXs, valYs } = this.dataLoader.splitTrainVal(
                this.dataLoader.trainData.xs, 
                this.dataLoader.trainData.ys, 
                0.1
            );

            // Set up training callbacks
            const fitCallbacks = tfvis.show.fitCallbacks(
                { name: 'Training Metrics' },
                ['loss', 'val_loss', 'acc', 'val_acc']
            );

            // Train model
            const history = await this.model.fit(trainXs, trainYs, {
                epochs: 10,
                batchSize: 128,
                validationData: [valXs, valYs],
                shuffle: true,
                callbacks: fitCallbacks
            });

            // Clean up tensors
            trainXs.dispose();
            trainYs.dispose();
            valXs.dispose();
            valYs.dispose();

            this.log('Training completed!');
            document.getElementById('saveModel').disabled = false;

        } catch (error) {
            this.log(`Training error: ${error.message}`);
            console.error('Training error:', error);
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
        
        // Flatten and dense layers
        model.add(tf.layers.flatten());
        model.add(tf.layers.dense({ units: 128, activation: 'relu' }));
        model.add(tf.layers.dropout({ rate: 0.5 }));
        model.add(tf.layers.dense({ units: 10, activation: 'softmax' }));

        // Compile model
        model.compile({
            optimizer: 'adam',
            loss: 'categoricalCrossentropy',
            metrics: ['accuracy']
        });

        return model;
    }

    /**
     * Evaluate model on test data
     */
    async onEvaluate() {
        if (!this.model) {
            alert('Please train or load a model first');
            return;
        }

        try {
            this.log('Evaluating model...');
            
            const testData = this.dataLoader.testData;
            const evaluation = await this.model.evaluate(testData.xs, testData.ys);
            const accuracy = evaluation[1].dataSync()[0];
            
            // Update metrics
            document.getElementById('metrics').innerHTML = `
                Overall Accuracy: <strong>${(accuracy * 100).toFixed(2)}%</strong>
            `;

            // Generate predictions for confusion matrix
            const predictions = this.model.predict(testData.xs);
            const predLabels = predictions.argMax(-1);
            const trueLabels = testData.ys.argMax(-1);

            // Create confusion matrix
            const confusionMatrix = await tfvis.metrics.confusionMatrix(trueLabels, predLabels);
            
            // Show charts in visor
            const confusionMatrixContainer = { name: 'Confusion Matrix', tab: 'Evaluation' };
            tfvis.render.confusionMatrix(confusionMatrixContainer, confusionMatrix, 10);
            
            // Calculate per-class accuracy
            const classAccuracy = this.calculatePerClassAccuracy(confusionMatrix);
            const accuracyContainer = { name: 'Per-Class Accuracy', tab: 'Evaluation' };
            tfvis.render.barchart(accuracyContainer, classAccuracy);
            
            // Clean up
            predictions.dispose();
            predLabels.dispose();
            trueLabels.dispose();
            
            this.log(`Evaluation completed. Accuracy: ${(accuracy * 100).toFixed(2)}%`);
            
        } catch (error) {
            this.log(`Evaluation error: ${error.message}`);
            console.error('Evaluation error:', error);
        }
    }

    /**
     * Calculate per-class accuracy from confusion matrix
     * @param {Array} confusionMatrix - Confusion matrix data
     * @returns {Array} Per-class accuracy data
     */
    calculatePerClassAccuracy(confusionMatrix) {
        return confusionMatrix.map((row, i) => {
            const correct = row[i];
            const total = row.reduce((sum, val) => sum + val, 0);
            const accuracy = total > 0 ? correct / total : 0;
            return {
                index: i,
                accuracy: accuracy,
                label: `Class ${i}`
            };
        });
    }

    /**
     * Test 5 random samples and display predictions
     */
    async onTestFive() {
        if (!this.model || !this.dataLoader.testData) {
            alert('Please load data and train a model first');
            return;
        }

        try {
            const testData = this.dataLoader.testData;
            const batch = this.dataLoader.getRandomTestBatch(testData.xs, testData.ys, 5);
            
            const predictions = this.model.predict(batch.xs);
            const predLabels = predictions.argMax(-1).dataSync();
            const trueLabels = batch.ys.argMax(-1).dataSync();
            
            // Display preview
            this.displayPreview(batch.xs, predLabels, trueLabels);
            
            // Clean up
            predictions.dispose();
            batch.xs.dispose();
            batch.ys.dispose();
            
        } catch (error) {
            this.log(`Test error: ${error.message}`);
            console.error('Test error:', error);
        }
    }

    /**
     * Display preview of test samples with predictions
     * @param {tf.Tensor} images - Image tensors
     * @param {Array} predLabels - Predicted labels
     * @param {Array} trueLabels - True labels
     */
    displayPreview(images, predLabels, trueLabels) {
        const container = document.getElementById('previewContainer');
        container.innerHTML = '';
        
        // Create preview items
        for (let i = 0; i < 5; i++) {
            const previewItem = document.createElement('div');
            previewItem.className = 'preview-item';
            
            const canvas = document.createElement('canvas');
            const labelDiv = document.createElement('div');
            
            // Draw image
            const imageSlice = images.slice([i, 0, 0, 0], [1, 28, 28, 1]);
            this.dataLoader.draw28x28ToCanvas(imageSlice, canvas, 4);
            imageSlice.dispose();
            
            // Display prediction and true label
            const isCorrect = predLabels[i] === trueLabels[i];
            labelDiv.innerHTML = `
                <strong>Pred:</strong> <span class="${isCorrect ? 'correct' : 'incorrect'}">${predLabels[i]}</span><br>
                <strong>True:</strong> ${trueLabels[i]}
            `;
            
            previewItem.appendChild(canvas);
            previewItem.appendChild(labelDiv);
            container.appendChild(previewItem);
        }
    }

    /**
     * Save model to files
     */
    async onSaveDownload() {
        if (!this.model) {
            alert('No model to save');
            return;
        }

        try {
            await this.model.save('downloads://mnist-cnn');
            this.log('Model saved successfully!');
        } catch (error) {
            this.log(`Save error: ${error.message}`);
            console.error('Save error:', error);
        }
    }

    /**
     * Load model from files
     */
    async onLoadFromFiles() {
        try {
            const jsonFile = document.getElementById('modelJsonFile').files[0];
            const weightsFile = document.getElementById('modelWeightsFile').files[0];
            
            if (!jsonFile || !weightsFile) {
                alert('Please select both model files');
                return;
            }

            this.log('Loading model...');
            this.model = await tf.loadLayersModel(
                tf.io.browserFiles([jsonFile, weightsFile])
            );
            
            this.updateModelInfo();
            document.getElementById('saveModel').disabled = false;
            document.getElementById('evaluate').disabled = false;
            document.getElementById('testFive').disabled = false;
            
            this.log('Model loaded successfully!');
            
        } catch (error) {
            this.log(`Model load error: ${error.message}`);
            console.error('Model load error:', error);
        }
    }

    /**
     * Update model information display
     */
    updateModelInfo() {
        if (!this.model) return;
        
        let totalParams = 0;
        this.model.summary(null, null, (line) => {
            const match = line.match(/params:\s+([\d,]+)/);
            if (match) {
                totalParams += parseInt(match[1].replace(/,/g, ''));
            }
        });
        
        document.getElementById('modelInfo').innerHTML = `
            Layers: ${this.model.layers.length}<br>
            Total parameters: ${totalParams.toLocaleString()}
        `;
    }

    /**
     * Reset application state
     */
    onReset() {
        if (this.isTraining) {
            alert('Cannot reset during training');
            return;
        }

        // Dispose model and data
        if (this.model) {
            this.model.dispose();
            this.model = null;
        }
        
        this.dataLoader.dispose();
        
        // Reset UI
        document.getElementById('dataStatus').textContent = 'No data loaded';
        document.getElementById('modelInfo').textContent = 'No model loaded';
        document.getElementById('metrics').textContent = 'No evaluation performed';
        document.getElementById('previewContainer').innerHTML = '';
        
        // Reset buttons
        document.getElementById('train').disabled = true;
        document.getElementById('evaluate').disabled = true;
        document.getElementById('testFive').disabled = true;
        document.getElementById('saveModel').disabled = true;
        document.getElementById('loadModel').disabled = true;
        
        // Clear file inputs
        document.getElementById('trainFile').value = '';
        document.getElementById('testFile').value = '';
        document.getElementById('modelJsonFile').value = '';
        document.getElementById('modelWeightsFile').value = '';
        
        this.log('Application reset');
    }

    /**
     * Toggle tfjs-vis visor
     */
    toggleVisor() {
        tfvis.visor().toggle();
    }

    /**
     * Add message to training logs
     * @param {string} message - Log message
     */
    log(message) {
        const logs = document.getElementById('trainingLogs');
        const timestamp = new Date().toLocaleTimeString();
        const logEntry = document.createElement('div');
        logEntry.textContent = `[${timestamp}] ${message}`;
        logs.appendChild(logEntry);
        logs.scrollTop = logs.scrollHeight;
    }
}

// Initialize application when page loads
let app;
document.addEventListener('DOMContentLoaded', () => {
    app = new MNISTApp();
});
