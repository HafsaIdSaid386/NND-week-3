/**
 * DataLoader class - Handles loading and processing of MNIST CSV data
 * Responsible for file parsing, tensor creation, and data management
 */
class DataLoader {
    constructor() {
        this.trainData = null;
        this.testData = null;
    }

    /**
     * Load and process data from a CSV file
     * @param {File} file - The CSV file to load
     * @returns {Promise} Promise that resolves to {xs, ys} tensors
     */
    async loadFromFile(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            
            reader.onload = (event) => {
                try {
                    const text = event.target.result;
                    const lines = text.split('\n').filter(line => line.trim() !== '');
                    
                    const labels = [];
                    const pixels = [];
                    
                    for (const line of lines) {
                        const values = line.split(',').map(val => val.trim());
                        if (values.length !== 785) continue;
                        
                        const label = parseInt(values[0]);
                        const pixelValues = values.slice(1, 785).map(Number);
                        
                        labels.push(label);
                        pixels.push(pixelValues);
                    }
                    
                    if (labels.length === 0) {
                        reject(new Error('No valid data found in file'));
                        return;
                    }
                    
                    const tensors = tf.tidy(() => {
                        // Create tensor from pixel data, normalize to [0,1], reshape to image format
                        const xs = tf.tensor2d(pixels)
                            .div(255)
                            .reshape([labels.length, 28, 28, 1]);
                        
                        // Convert labels to one-hot encoding
                        const ys = tf.oneHot(tf.tensor1d(labels, 'int32'), 10);
                        
                        return { xs, ys };
                    });
                    
                    resolve(tensors);
                    
                } catch (error) {
                    reject(error);
                }
            };
            
            reader.onerror = () => reject(new Error('Failed to read file'));
            reader.readAsText(file);
        });
    }

    /**
     * Load training data from file
     * @param {File} file - Training CSV file
     */
    async loadTrainFromFiles(file) {
        this.trainData = await this.loadFromFile(file);
        return this.trainData;
    }

    /**
     * Load test data from file
     * @param {File} file - Test CSV file
     */
    async loadTestFromFiles(file) {
        this.testData = await this.loadFromFile(file);
        return this.testData;
    }

    /**
     * Split data into training and validation sets
     * @param {tf.Tensor} xs - Input features
     * @param {tf.Tensor} ys - Labels
     * @param {number} valRatio - Validation ratio (default: 0.1)
     * @returns {Object} Split datasets
     */
    splitTrainVal(xs, ys, valRatio = 0.1) {
        return tf.tidy(() => {
            const numSamples = xs.shape[0];
            const numVal = Math.floor(numSamples * valRatio);
            const numTrain = numSamples - numVal;
            
            // Create random indices for splitting
            const indices = tf.randomUniform([numSamples]).arraySync()
                .map((val, idx) => ({ val, idx }))
                .sort((a, b) => a.val - b.val)
                .map(item => item.idx);
            
            const trainIndices = indices.slice(0, numTrain);
            const valIndices = indices.slice(numTrain);
            
            const trainXs = tf.gather(xs, trainIndices);
            const trainYs = tf.gather(ys, trainIndices);
            const valXs = tf.gather(xs, valIndices);
            const valYs = tf.gather(ys, valIndices);
            
            return { trainXs, trainYs, valXs, valYs };
        });
    }

    /**
     * Get a random batch of test samples
     * @param {tf.Tensor} xs - Test features
     * @param {tf.Tensor} ys - Test labels
     * @param {number} k - Number of samples (default: 5)
     * @returns {Object} Batch of samples with true labels
     */
    getRandomTestBatch(xs, ys, k = 5) {
        return tf.tidy(() => {
            const numSamples = xs.shape[0];
            const indices = [];
            
            // Select k unique random indices
            while (indices.length < k) {
                const idx = Math.floor(Math.random() * numSamples);
                if (!indices.includes(idx)) indices.push(idx);
            }
            
            const batchXs = tf.gather(xs, indices);
            const batchYs = tf.gather(ys, indices);
            const trueLabels = tf.argMax(batchYs, 1).arraySync();
            
            return {
                xs: batchXs,
                ys: batchYs,
                indices: indices,
                trueLabels: trueLabels
            };
        });
    }

    /**
     * Draw a 28x28 image tensor to canvas
     * @param {tf.Tensor} tensor - Image tensor [1,28,28,1]
     * @param {HTMLCanvasElement} canvas - Target canvas element
     * @param {number} scale - Scaling factor for display
     */
    draw28x28ToCanvas(tensor, canvas, scale = 4) {
        const [height, width] = [28, 28];
        canvas.width = width * scale;
        canvas.height = height * scale;
        
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        const imageData = tensor.squeeze().arraySync();
        
        // Draw each pixel to canvas
        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
                const pixel = imageData[y][x];
                const gray = Math.floor(pixel * 255);
                ctx.fillStyle = `rgb(${gray}, ${gray}, ${gray})`;
                ctx.fillRect(x * scale, y * scale, scale, scale);
            }
        }
    }

    /**
     * Clean up tensors and free memory
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
