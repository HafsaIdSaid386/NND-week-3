// data-loader.js (FIXED VERSION)
/**
 * Data loader utility for MNIST CSV files
 * Handles file parsing, normalization, and tensor creation
 */

class DataLoader {
    constructor() {
        this.trainData = null;
        this.testData = null;
    }

    /**
     * Parse CSV file and convert to tensors
     * @param {File} file - CSV file object
     * @returns {Promise<{xs: tf.Tensor, ys: tf.Tensor}>} Normalized image tensors and one-hot labels
     */
    async loadFromFile(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            
            reader.onload = (event) => {
                // Use tf.tidy to automatically clean up intermediate tensors
                const result = tf.tidy(() => {
                    try {
                        const csvText = event.target.result;
                        console.log('File loaded, first 500 chars:', csvText.substring(0, 500));
                        
                        // More robust CSV parsing
                        const lines = csvText.split('\n')
                            .map(line => line.trim())
                            .filter(line => line.length > 0);
                        
                        console.log(`Found ${lines.length} lines in CSV`);
                        
                        if (lines.length === 0) {
                            throw new Error('CSV file is empty');
                        }

                        const labels = [];
                        const pixels = [];
                        let validLines = 0;

                        // Parse each line: first value is label, next 784 are pixels
                        for (let i = 0; i < lines.length; i++) {
                            const line = lines[i];
                            // Split by comma and remove empty values
                            const values = line.split(',')
                                .map(val => val.trim())
                                .filter(val => val.length > 0)
                                .map(Number);
                            
                            // Check if we have exactly 785 values (label + 784 pixels)
                            if (values.length === 785) {
                                labels.push(values[0]);
                                pixels.push(values.slice(1, 785)); // Take exactly 784 pixels
                                validLines++;
                            } else if (values.length > 0) {
                                console.warn(`Line ${i + 1} has ${values.length} values, expected 785. Skipping.`);
                            }
                        }

                        console.log(`Successfully parsed ${validLines} valid lines`);
                        
                        if (validLines === 0) {
                            throw new Error('No valid data rows found in CSV file');
                        }

                        // Convert to tensors
                        const xs = tf.tensor4d(
                            pixels,
                            [pixels.length, 28, 28, 1]
                        ).div(255.0); // Normalize to [0, 1]

                        const ys = tf.oneHot(
                            tf.tensor1d(labels, 'int32'),
                            10
                        );

                        return { xs, ys };
                    } catch (error) {
                        reject(error);
                        return null;
                    }
                });
                
                if (result) {
                    resolve(result);
                }
            };
            
            reader.onerror = (error) => {
                console.error('FileReader error:', error);
                reject(new Error(`Failed to read file: ${error}`));
            };
            
            reader.readAsText(file);
        });
    }

    /**
     * Load training data from file
     * @param {File} file - Training CSV file
     * @returns {Promise<{xs: tf.Tensor, ys: tf.Tensor}>}
     */
    async loadTrainFromFiles(file) {
        console.log('Loading training file:', file.name);
        this.trainData = await this.loadFromFile(file);
        console.log('Training data loaded:', this.trainData.xs.shape);
        return this.trainData;
    }

    /**
     * Load test data from file
     * @param {File} file - Test CSV file
     * @returns {Promise<{xs: tf.Tensor, ys: tf.Tensor}>}
     */
    async loadTestFromFiles(file) {
        console.log('Loading test file:', file.name);
        this.testData = await this.loadFromFile(file);
        console.log('Test data loaded:', this.testData.xs.shape);
        return this.testData;
    }

    /**
     * Split training data into training and validation sets
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
            
            console.log(`Splitting data: ${numTrain} training, ${numVal} validation`);
            
            // Create indices and shuffle
            const indices = tf.util.createShuffledIndices(numSamples);
            
            // Split indices
            const trainIndices = indices.slice(0, numTrain);
            const valIndices = indices.slice(numTrain);
            
            // Create subsets
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
     * @returns {Object} Batch of samples
     */
    getRandomTestBatch(xs, ys, k = 5) {
        return tf.tidy(() => {
            const numSamples = xs.shape[0];
            const indices = [];
            
            // Generate k unique random indices
            while (indices.length < k && indices.length < numSamples) {
                const idx = Math.floor(Math.random() * numSamples);
                if (!indices.includes(idx)) {
                    indices.push(idx);
                }
            }
            
            console.log(`Selected random indices: ${indices}`);
            
            const batchXs = tf.gather(xs, indices);
            const batchYs = tf.gather(ys, indices);
            
            return {
                xs: batchXs,
                ys: batchYs,
                indices: indices
            };
        });
    }

    /**
     * Draw 28x28 image tensor to canvas
     * @param {tf.Tensor} tensor - Image tensor (shape: [28, 28, 1] or [1, 28, 28, 1])
     * @param {HTMLCanvasElement} canvas - Target canvas element
     * @param {number} scale - Scaling factor (default: 4)
     */
    draw28x28ToCanvas(tensor, canvas, scale = 4) {
        tf.tidy(() => {
            // Ensure tensor is 2D and normalized
            let imageTensor = tensor.squeeze();
            if (imageTensor.rank === 3) {
                imageTensor = imageTensor.squeeze();
            }
            
            // Denormalize to 0-255
            imageTensor = imageTensor.mul(255).cast('int32');
            
            // Get image data
            const data = imageTensor.dataSync();
            
            // Set canvas dimensions
            canvas.width = 28 * scale;
            canvas.height = 28 * scale;
            
            const ctx = canvas.getContext('2d');
            const imageData = ctx.createImageData(28 * scale, 28 * scale);
            
            // Draw scaled pixels
            for (let y = 0; y < 28; y++) {
                for (let x = 0; x < 28; x++) {
                    const pixelValue = data[y * 28 + x];
                    
                    for (let sy = 0; sy < scale; sy++) {
                        for (let sx = 0; sx < scale; sx++) {
                            const idx = ((y * scale + sy) * 28 * scale + (x * scale + sx)) * 4;
                            imageData.data[idx] = pixelValue;     // R
                            imageData.data[idx + 1] = pixelValue; // G
                            imageData.data[idx + 2] = pixelValue; // B
                            imageData.data[idx + 3] = 255;        // A
                        }
                    }
                }
            }
            
            ctx.putImageData(imageData, 0, 0);
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
