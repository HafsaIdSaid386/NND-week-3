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
        const model = tf.sequential({
            layers: [
                tf.layers.conv2d({
                    inputShape: [28, 28, 1],
                    filters: 32,
                    kernelSize: 3,
                    activation: 'relu',
                    padding: 'same'
                }),
                tf.layers.conv2d({
                    filters: 64,
                    kernelSize: 3,
                    activation: 'relu',
                    padding: 'same'
                }),
                tf.layers.maxPooling2d({ poolSize: 2 }),
                tf.layers.dropout({ rate: 0.25 }),
                tf.layers.flatten(),
                tf.layers.dense({ units: 128, activation: 'relu' }),
                tf.layers.dropout({ rate: 0.5 }),
                tf.layers.dense({ units: 10, activation: 'softmax' })
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
            
            if (!this.model) {
                this.model = this.createModel();
            }
            
            const { trainXs, trainYs, valXs, valYs } = this.dataLoader.splitTrainVal(
                this.dataLoader.trainData.xs, 
                this.dataLoader.trainData.ys, 
                0.1
            );
            
            this.updateModelInfo();
            
            const history = await this.model.fit(trainXs, trainYs, {
                epochs: 5,
                batchSize: 128,
                validationData: [valXs, valYs],
                shuffle: true,
                callbacks: tfvis.show.fitCallbacks(
                    { name: 'Training Performance' },
                    ['loss', 'acc', 'val_loss', 'val_acc'],
                    { callbacks: ['onEpochEnd'] }
                )
            });
            
            const finalAcc = history.history.acc[history.history.acc.length - 1];
            const finalValAcc = history.history.val_acc[history.history.val_acc.length - 1];
            this.log(`Training completed! Final accuracy: ${finalAcc.toFixed(4)}, Validation: ${finalValAcc.toFixed(4)}`);
            
            tf.dispose([trainXs, trainYs, valXs, valYs]);
            
        } catch (error) {
            this.log(`Training error: ${error.message}`);
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
            
            const testXs = this.dataLoader.testData.xs;
            const testYs = this.dataLoader.testData.ys;
            
            const predictions = this.model.predict(testXs);
            const predictedLabels = tf.argMax(predictions, 1);
            const trueLabels = tf.argMax(testYs, 1);
            
            const accuracy = await this.calculateAccuracy(predictedLabels, trueLabels);
            
            this.elements.metrics.innerHTML = `Overall Accuracy: ${(accuracy * 100).toFixed(2)}%`;
            
            await this.createEvaluationCharts(predictions, testYs);
            
            tf.dispose([predictions, predictedLabels, trueLabels]);
            
            this.log(`Evaluation completed! Accuracy: ${(accuracy * 100).toFixed(2)}%`);
            
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

    async createEvaluationCharts(predictions, trueLabels) {
        const predictedLabels = await tf.argMax(predictions, 1).array();
        const actualLabels = await tf.argMax(trueLabels, 1).array();
        
        const confusionMatrix = await tfvis.metrics.confusionMatrix(actualLabels, predictedLabels);
        tfvis.render.confusionMatrix(
            { name: 'Confusion Matrix', tab: 'Evaluation' },
            { values: confusionMatrix },
            { shadeDiagonal: true }
        );
        
        const classAccuracy = [];
        const numClasses = 10;
        
        for (let i = 0; i < numClasses; i++) {
            const classIndices = actualLabels.map((label, idx) => label === i);
            const correct = classIndices.filter((isClass, idx) => isClass && predictedLabels[idx] === i).length;
            const total = classIndices.filter(isClass => isClass).length;
            classAccuracy.push(total > 0 ? correct / total : 0);
        }
        
        tfvis.render.barchart(
            { name: 'Per-Class Accuracy', tab: 'Evaluation' },
            { values: classAccuracy.map((acc, i) => ({ index: i, value: acc })) },
            { yAxisDomain: [0, 1] }
        );
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
            await this.model.save('downloads://mnist-cnn');
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

document.addEventListener('DOMContentLoaded', () => {
    new MNISTApp();
});
