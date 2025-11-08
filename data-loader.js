/**
 * DataLoader class - Handles loading and processing of MNIST CSV data
 */
class DataLoader {
    constructor() {
        this.trainData = null;
        this.testData = null;
    }

    /**
     * Load and process data from a CSV file
     */
    async loadFromFile(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            
            reader.onload = (event) => {
                try {
                    const text = event.target.result;
                    const lines = text.split('\n').filter(line => line.trim() !== '');
                    
                    console.log(`Found ${lines.length} lines in file`);
                    
                    const labels = [];
                    const pixels = [];
                    let validLines = 0;
                    let skippedLines = 0;
                    
                    for (let i = 0; i < lines.length; i++) {
                        const line = lines[i].trim();
                        if (!line) continue;
                        
                        // Simple comma splitting for MNIST format
                        const values = line.split(',');
                        
                        // Check if we have enough values (1 label + 784 pixels)
                        if (values.length < 785) {
                            skippedLines++;
                            continue;
                        }
                        
                        // Parse label (first value)
                        const label = parseInt(values[0]);
                        if (isNaN(label) || label < 0 || label > 9) {
                            skippedLines++;
                            continue;
                        }
                        
                        // Parse pixels (next 784 values)
                        const pixelValues = [];
                        let validPixels = 0;
                        
                        for (let j = 1; j <= 784; j++) {
                            const pixelVal = parseInt(values[j]);
                            if (!isNaN(pixelVal)) {
                                pixelValues.push(pixelVal);
                                validPixels++;
                            } else {
                                pixelValues.push(0); // Default to 0 for invalid pixels
                            }
                        }
                        
                        // Only add if we have valid data
                        if (validPixels > 0) {
                            labels.push(label);
                            pixels.push(pixelValues);
                            validLines++;
                        } else {
                            skippedLines++;
                        }
                    }
                    
                    console.log(`Valid lines: ${validLines}, Skipped: ${skippedLines}`);
                    
                    if (validLines === 0) {
                        reject(new Error('No valid MNIST data found. File should have 785 numbers per line.'));
                        return;
                    }
                    
                    // Create tensors
                    const tensors = tf.tidy(() => {
                        // Create tensor from pixel data and normalize to [0,1]
                        const xs = tf.tensor2d(pixels, [validLines, 784])
                            .div(255)
                            .reshape([validLines, 28, 28, 1]);
                        
                        // Convert labels to one-hot encoding
                        const ys = tf.oneHot(tf.tensor1d(labels, 'int32'), 10);
                        
                        return { xs, ys };
                    });
                    
                    resolve(tensors);
                    
                } catch (error) {
                    reject(new Error(`File processing error: ${error.message}`));
                }
            };
            
            reader.onerror = () => reject(new Error('Failed to read file'));
            reader.readAsText(file);
        });
    }

    async loadTrainFromFiles(file) {
        this.trainData = await this.loadFromFile(file);
        return this.trainData;
    }

    async loadTestFromFiles(file) {
        this.testData = await this.loadFromFile(file);
        return this.testData;
    }

    splitTrainVal(xs, ys, valRatio = 0.1) {
        return tf.tidy(() => {
            const numSamples = xs.shape[0];
            const numVal = Math.floor(numSamples * valRatio);
            const numTrain = numSamples - numVal;
            
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

    getRandomTestBatch(xs, ys, k = 5) {
        return tf.tidy(() => {
            const numSamples = xs.shape[0];
            const indices = [];
            
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

    draw28x28ToCanvas(tensor, canvas, scale = 4) {
        const [height, width] = [28, 28];
        canvas.width = width * scale;
        canvas.height = height * scale;
        
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        const imageData = tensor.squeeze().arraySync();
        
        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
                const pixel = imageData[y][x];
                const gray = Math.floor(pixel * 255);
                ctx.fillStyle = `rgb(${gray}, ${gray}, ${gray})`;
                ctx.fillRect(x * scale, y * scale, scale, scale);
            }
        }
    }

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
