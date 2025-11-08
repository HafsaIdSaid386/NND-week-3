// data-loader.js
/**
 * Data loader utility for MNIST CSV files
 * Handles file parsing, normalization, and tensor management
 */

class DataLoader {
    constructor() {
        this.trainData = null;
        this.testData = null;
    }

    /**
     * Load training data from CSV file
     * @param {File} file - CSV file containing training data
     * @returns {Promise<{xs: tf.Tensor, ys: tf.Tensor}>} Normalized images and one-hot labels
     */
    async loadTrainFromFiles(file) {
        return tf.tidy(() => {
            const data = this.parseCSV(file);
            return this.processData(data);
        });
    }

    /**
     * Load test data from CSV file
     * @param {File} file - CSV file containing test data
     * @returns {Promise<{xs: tf.Tensor, ys: tf.Tensor}>} Normalized images and one-hot labels
     */
    async loadTestFromFiles(file) {
        return tf.tidy(() => {
            const data = this.parseCSV(file);
            return this.processData(data);
        });
    }

    /**
     * Parse CSV file content
     * @param {File} file - CSV file to parse
     * @returns {Array<{label: number, pixels: number[]}>} Parsed data
     */
    parseCSV(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            
            reader.onload = (e) => {
                try {
                    const content = e.target.result;
                    const lines = content.split('\n').filter(line => line.trim() !== '');
                    const data = [];
                    
                    for (const line of lines) {
                        const values = line.split(',').map(val => parseInt(val.trim()));
                        
                        if (values.length !== 785) {
                            console.warn('Skipping invalid line:', line);
                            continue;
                        }
                        
                        const label = values[0];
                        const pixels = values.slice(1, 785);
                        
                        data.push({ label, pixels });
                    }
                    
                    resolve(data);
                } catch (error) {
                    reject(new Error(`CSV parsing error: ${error.message}`));
                }
            };
            
            reader.onerror = () => reject(new Error('File reading error'));
            reader.readAsText(file);
        });
    }

    /**
     * Process parsed data into tensors
     * @param {Array<{label: number, pixels: number[]}>} data - Parsed CSV data
     * @returns {{xs: tf.Tensor, ys: tf.Tensor}} Processed tensors
     */
    processData(data) {
        return tf.tidy(() => {
            const labels = [];
            const pixels = [];
            
            for (const item of data) {
                labels.push(item.label);
                pixels.push(...item.pixels);
            }
            
            // Create tensors
            const xs = tf.tensor4d(pixels, [data.length, 28, 28, 1]);
            const ys = tf.oneHot(tf.tensor1d(labels, 'int32'), 10);
            
            // Normalize pixel values to [0, 1]
            const normalizedXs = xs.div(255);
            
            return { xs: normalizedXs, ys: ys };
        });
    }

    /**
     * Split training data into training and validation sets
     * @param {tf.Tensor} xs - Input features
     * @param {tf.Tensor} ys - Labels
     * @param {number} valRatio - Validation ratio (default: 0.1)
     * @returns {{trainXs: tf.Tensor, trainYs: tf.Tensor, valXs: tf.Tensor, valYs: tf.Tensor}} Split data
     */
    splitTrainVal(xs, ys, valRatio = 0.1) {
        return tf.tidy(() => {
            const numSamples = xs.shape[0];
            const numVal = Math.floor(numSamples * valRatio);
            const numTrain = numSamples - numVal;
            
            // Split indices
            const indices = tf.util.createShuffledIndices(numSamples);
            const trainIndices = indices.slice(0, numTrain);
            const valIndices = indices.slice(numTrain);
            
            // Split tensors
            const trainXs = tf.gather(xs, trainIndices);
            const trainYs = tf.gather(ys, trainIndices);
            const valXs = tf.gather(xs, valIndices);
            const valYs = tf.gather(ys, valIndices);
            
            return { trainXs, trainYs, valXs, valYs };
        });
    }

    /**
     * Get random batch of test samples for preview
     * @param {tf.Tensor} xs - Test features
     * @param {tf.Tensor} ys - Test labels
     * @param {number} k - Number of samples (default: 5)
     * @returns {{xs: tf.Tensor, ys: tf.Tensor, indices: number[]}} Random batch
     */
    getRandomTestBatch(xs, ys, k = 5) {
        return tf.tidy(() => {
            const numSamples = xs.shape[0];
            const indices = [];
            
            // Generate random indices
            for (let i = 0; i < k; i++) {
                indices.push(Math.floor(Math.random() * numSamples));
            }
            
            const batchXs = tf.gather(xs, indices);
            const batchYs = tf.gather(ys, indices);
            
            return { xs: batchXs, ys: batchYs, indices };
        });
    }

    /**
     * Draw 28x28 image to canvas
     * @param {tf.Tensor} tensor - Image tensor
     * @param {HTMLCanvasElement} canvas - Target canvas
     * @param {number} scale - Scale factor (default: 4)
     */
    draw28x28ToCanvas(tensor, canvas, scale = 4) {
        return tf.tidy(() => {
            // Ensure tensor is 2D and denormalized
            const image = tensor.squeeze().mul(255).cast('int32');
            
            const ctx = canvas.getContext('2d');
            canvas.width = 28 * scale;
            canvas.height = 28 * scale;
            
            // Create image data
            const imageData = new ImageData(28, 28);
            const data = image.dataSync();
            
            for (let i = 0; i < 784; i++) {
                const val = data[i];
                imageData.data[i * 4] = val;     // R
                imageData.data[i * 4 + 1] = val; // G
                imageData.data[i * 4 + 2] = val; // B
                imageData.data[i * 4 + 3] = 255; // A
            }
            
            // Draw and scale
            const tempCanvas = document.createElement('canvas');
            tempCanvas.width = 28;
            tempCanvas.height = 28;
            const tempCtx = tempCanvas.getContext('2d');
            tempCtx.putImageData(imageData, 0, 0);
            
            ctx.imageSmoothingEnabled = false;
            ctx.drawImage(tempCanvas, 0, 0, 28 * scale, 28 * scale);
        });
    }

    /**
     * Clean up tensors to prevent memory leaks
     */
    dispose() {
        if (this.trainData) {
            this.trainData.xs.dispose();
            this.trainData.ys.dispose();
            this.trainData = null;
        }
        if (this.testData) {
            this.testData.xs.dispose();
            this.testData.ys.dispose();
            this.testData = null;
        }
    }
}
