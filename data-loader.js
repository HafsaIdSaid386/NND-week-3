/**
 * DataLoader - Working version for MNIST CSV files
 */
class DataLoader {
    constructor() {
        this.trainData = null;
        this.testData = null;
    }

    async loadFromFile(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            
            reader.onload = (event) => {
                try {
                    const text = event.target.result;
                    const lines = text.split('\n');
                    
                    const labels = [];
                    const pixels = [];
                    let validLines = 0;

                    for (let i = 0; i < lines.length; i++) {
                        const line = lines[i].trim();
                        if (!line) continue;

                        const values = line.split(',');
                        
                        // Skip if not enough values
                        if (values.length < 10) continue;
                        
                        // Parse label (first value)
                        const label = parseInt(values[0]);
                        if (isNaN(label) || label < 0 || label > 9) continue;
                        
                        // Parse pixels - take up to 784 values
                        const pixelValues = [];
                        for (let j = 1; j < values.length && pixelValues.length < 784; j++) {
                            const val = parseInt(values[j]);
                            pixelValues.push(isNaN(val) ? 0 : Math.max(0, Math.min(255, val)));
                        }
                        
                        // Pad with zeros if needed
                        while (pixelValues.length < 784) {
                            pixelValues.push(0);
                        }
                        
                        labels.push(label);
                        pixels.push(pixelValues);
                        validLines++;
                    }

                    if (validLines === 0) {
                        reject(new Error('No valid data found. File format might be incorrect.'));
                        return;
                    }

                    console.log(`Successfully loaded ${validLines} samples`);

                    // Create tensors
                    const tensors = tf.tidy(() => {
                        const xs = tf.tensor2d(pixels, [validLines, 784])
                            .div(255)
                            .reshape([validLines, 28, 28, 1]);
                        
                        const ys = tf.oneHot(tf.tensor1d(labels, 'int32'), 10);
                        
                        return { xs, ys };
                    });

                    resolve(tensors);
                    
                } catch (error) {
                    reject(new Error(`File parsing error: ${error.message}`));
                }
            };
            
            reader.onerror = () => reject(new Error('Failed to read file'));
            reader.readAsText(file);
        });
    }

    async loadTrainFromFiles(file) {
        console.log('Loading training data from:', file.name);
        this.trainData = await this.loadFromFile(file);
        console.log('Training data loaded:', this.trainData.xs.shape);
        return this.trainData;
    }

    async loadTestFromFiles(file) {
        console.log('Loading test data from:', file.name);
        this.testData = await this.loadFromFile(file);
        console.log('Test data loaded:', this.testData.xs.shape);
        return this.testData;
    }

    createTestData() {
        console.log('Creating synthetic test data...');
        return tf.tidy(() => {
            const numSamples = 200;
            // Create simple digit-like patterns
            const images = tf.randomUniform([numSamples, 28, 28, 1], 0, 0.3);
            
            // Add some structure to make it look like digits
            const structured = images.add(
                tf.randomUniform([numSamples, 28, 28, 1], 0, 0.7)
                    .mul(tf.randomUniform([numSamples, 1, 1, 1], 0, 1))
            ).clipByValue(0, 1);
            
            const labels = tf.oneHot(
                tf.randomUniform([numSamples], 0, 10, 'int32'), 
                10
            );
            
            return { xs: structured, ys: labels };
        });
    }

    getRandomTestBatch(xs, ys, k = 5) {
        return tf.tidy(() => {
            const numSamples = xs.shape[0];
            const indices = [];
            
            for (let i = 0; i < k; i++) {
                indices.push(Math.floor(Math.random() * numSamples));
            }
            
            const batchXs = tf.gather(xs, indices);
            const batchYs = tf.gather(ys, indices);
            
            return { 
                xs: batchXs, 
                ys: batchYs, 
                indices: indices 
            };
        });
    }

    draw28x28ToCanvas(tensor, canvas, scale = 4) {
        const ctx = canvas.getContext('2d');
        canvas.width = 28 * scale;
        canvas.height = 28 * scale;
        
        const imageData = tensor.squeeze().arraySync();
        
        for (let y = 0; y < 28; y++) {
            for (let x = 0; x < 28; x++) {
                const pixel = imageData[y] ? (imageData[y][x] || 0) : 0;
                const gray = Math.floor(pixel * 255);
                ctx.fillStyle = `rgb(${gray},${gray},${gray})`;
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
