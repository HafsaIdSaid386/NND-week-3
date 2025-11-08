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
        console.log(message); // Also log to console for debugging
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
            const trainData = await this.dataLoader.loadTrainFromFiles(trainFile);
            this.log(`Training data loaded: ${trainData.xs.shape[0]} samples`);
            
            this.log('Loading test data...');
            const testData = await this.dataLoader.loadTestFromFiles(testFile);
            this.log(`Test data loaded: ${testData.xs.shape[0]} samples`);
            
            const trainSamples = trainData.xs.shape[0];
            const testSamples = testData.xs.shape[0];
            
            this.elements.dataStatus.innerHTML = `
                Train samples: ${trainSamples}<br>
                Test samples: ${testSamples}
            `;
            
            this.log('Data loaded successfully!');
            this.updateUIState();
            
        } catch (error) {
            this.log(`Error loading data: ${error.message}`);
            alert(`Failed to load data: ${error.message}`);
        }
    }

    createModel() {
        // Very simple model for maximum compatibility
        const model = tf.sequential({
            layers: [
                tf.layers.flatten({inputShape: [28, 28, 1]}),
                tf.layers.dense({units: 128, activation: 'relu'}),
                tf.layers.dense({units: 10, activation: 'softmax'})
            ]
        });
        
        model.compile({
            optimizer: 'adam',
            loss: 'categoricalCrossentropy',
            metrics: ['accuracy']
        });
        
        return model;
    }

    async onTrain() {
        if (this.isTraining) return;
        
        try {
            this.isTraining = true;
            this.updateUIState();
            this.log('Starting model training...');
            this.log('Using simple dense network for maximum compatibility');
            
            // Create new model
            this.model = this.createModel();
            this.updateModelInfo();
            
            // Use a very small subset for testing first
            const numTrainSamples = this.dataLoader.trainData.xs.shape[0];
            const numTestSamples = Math.min(1000, numTrainSamples); // Use smaller subset
            
            this.log(`Using ${numTestSamples} samples for training`);
            
            // Take a subset of data
            const trainSubset = tf.tidy(() => {
                const indices = Array.from({length: numTestSamples}, (_, i) => i);
                const trainXs = tf.gather(this.dataLoader.trainData.xs, indices);
                const trainYs = tf.gather(this.dataLoader.trainData.ys, indices);
                return {trainXs, trainYs};
            });
            
            let currentEpoch = 0;
            
            // Train with very simple parameters
            const history = await this.model.fit(trainSubset.trainXs, trainSubset.trainYs, {
                epochs: 2,  // Just 2 epochs for testing
                batchSize: 32,
                validationSplit: 0.1,
                shuffle: true,
                callbacks: {
                    onEpochBegin: (epoch) => {
                        currentEpoch = epoch;
                        this.log(`Starting epoch ${epoch + 1}...`);
                    },
                    onBatchEnd: (batch, logs) => {
                        // Show progress every 10 batches
                        if (batch % 10 === 0) {
                            this.log(`Epoch ${currentEpoch + 1} - Batch ${batch} - loss: ${logs.loss.toFixed(4)}`);
                        }
                    },
                    onEpochEnd: (epoch, logs) => {
                        this.log(`Epoch ${epoch + 1} completed - loss: ${logs.loss.toFixed(4)}, acc: ${logs.acc.toFixed(4)}, val_loss: ${logs.val_loss ? logs.val_loss.toFixed(4) : 'N/A'}, val_acc: ${logs.val_acc ? logs.val_acc.toFixed(4) : 'N/A'}`);
                    }
                }
            });
            
            // Clean up
            trainSubset.trainXs.dispose();
            trainSubset.trainYs.dispose();
            
            this.log('Training completed successfully!');
            this.log('You can now test the model with "Test 5 Random" or evaluate full performance');
            
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
            this.log('Evaluating model on test data...');
            
            // Use smaller subset for evaluation
            const testSize = Math.min(500, this.dataLoader.testData.xs.shape[0]);
            const testSubset = tf.tidy(() => {
                const indices = Array.from({length: testSize}, (_, i) => i);
                const testXs = tf.gather(this.dataLoader.testData.xs, indices);
                const testYs = tf.gather(this.dataLoader.testData.ys, indices);
                return {testXs, testYs};
            });
            
            const predictions = this.model.predict(testSubset.testXs);
            const predictedLabels = tf.argMax(predictions, 1);
            const trueLabels = tf.argMax(testSubset.testYs, 1);
            
            const accuracy = await this.calculateAccuracy(predictedLabels, trueLabels);
            
            this.elements.metrics.innerHTML = `Overall Accuracy: ${(accuracy * 100).toFixed(2)}% (on ${testSize} samples)`;
            
            this.log(`Evaluation completed! Accuracy: ${(accuracy * 100).toFixed(2)}% on ${testSize} samples`);
            
            // Clean up
            testSubset.testXs.dispose();
            testSubset.testYs.dispose();
            tf.dispose([predictions, predictedLabels, trueLabels]);
            
        } catch (error) {
            this.log(`Evaluation error: ${error.message}`);
            alert(`Evaluation failed: ${error.message}`);
        }
    }

    async calculateAccuracy(predicted, trueLabels) {
        const equals = tf.equal(predicted, trueLabels);
        const accuracy = equals.mean();
        const result = await accuracy.data();
        equals.dispose();
        accuracy.dispose();
        return result[0];
    }

    async onTestFive() {
        if (!this.model || !this.dataLoader.testData) return;
        
        try {
            const batch = this.dataLoader.getRandomTestBatch(
                this.dataLoader.testData.xs,
                this.dataLoader.testData.ys,
                5
            );
            
            const predictions = this.model.predict(batch.xs);
            const predictedLabels = tf.argMax(predictions, 1).arraySync();
            
            this.elements.imagePreview.innerHTML = '';
            this.elements.predictionResults.innerHTML = '';
            
            for (let i = 0; i < 5; i++) {
                const previewItem = document.createElement('div');
                previewItem.className = 'preview-item';
                
                const canvas = document.createElement('canvas');
                this.dataLoader.draw28x28ToCanvas(tf.slice(batch.xs, [i, 0, 0, 0], [1, 28, 28, 1]), canvas);
                
                const labelDiv = document.createElement('div');
                const isCorrect = predictedLabels[i] === batch.trueLabels[i];
                labelDiv.innerHTML = `
                    <span class="${isCorrect ? 'correct' : 'wrong'}">
                        Pred: ${predictedLabels[i]} | True: ${batch.trueLabels[i]}
                    </span>
                `;
                
                previewItem.appendChild(canvas);
                previewItem.appendChild(labelDiv);
                this.elements.imagePreview.appendChild(previewItem);
            }
            
            this.log('Displayed 5 random test samples with predictions');
            
            tf.dispose([batch.xs, batch.ys, predictions, predictedLabels]);
            
        } catch (error) {
            this.log(`Test error: ${error.message}`);
            alert(`Testing failed: ${error.message}`);
        }
    }

    async onSaveDownload() {
        if (!this.model) return;
        
        try {
            this.log('Saving model...');
            await this.model.save('downloads://mnist-model');
            this.log('Model saved successfully!');
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
            
            this.log('Loading model...');
            
            this.model = await tf.loadLayersModel(tf.io.browserFiles([jsonFile, weightsFile]));
            
            this.model.compile({
                optimizer: 'adam',
                loss: 'categoricalCrossentropy',
                metrics: ['accuracy']
            });
            
            this.updateModelInfo();
            this.updateUIState();
            this.log('Model loaded successfully!');
            
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
            return `${layer.name}: ${params} params`;
        }).join('<br>');
        
        this.elements.modelInfo.innerHTML = `
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
