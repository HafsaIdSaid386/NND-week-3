// data-loader.js - FIXED VERSION
class DataLoader {
    constructor() {
        this.trainData = null;
        this.testData = null;
    }

    async loadTrainFromFiles(file) {
        const data = await this.parseCSV(file);
        return this.processData(data);
    }

    async loadTestFromFiles(file) {
        const data = await this.parseCSV(file);
        return this.processData(data);
    }

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
                        if (values.length >= 785) {
                            const label = values[0];
                            const pixels = values.slice(1, 785);
                            data.push({ label, pixels });
                        }
                    }
                    resolve(data);
                } catch (error) {
                    reject(error);
                }
            };
            reader.onerror = () => reject(new Error('File reading failed'));
            reader.readAsText(file);
        });
    }

    processData(data) {
        return tf.tidy(() => {
            const labels = [];
            const pixels = [];
            
            for (const item of data) {
                labels.push(item.label);
                pixels.push(...item.pixels);
            }
            
            const xs = tf.tensor4d(pixels, [data.length, 28, 28, 1]).div(255);
            const ys = tf.oneHot(tf.tensor1d(labels, 'int32'), 10);
            
            return { xs, ys };
        });
    }

    addNoise(tensor, noiseLevel = 0.5) {
        return tf.tidy(() => {
            const noise = tf.randomNormal(tensor.shape, 0, noiseLevel);
            return tensor.add(noise).clipByValue(0, 1);
        });
    }

    splitTrainVal(xs, ys, valRatio = 0.1) {
        return tf.tidy(() => {
            const numSamples = xs.shape[0];
            const numVal = Math.floor(numSamples * valRatio);
            const indices = Array.from(tf.util.createShuffledIndices(numSamples)); // FIX: Convert to Array
            
            const trainIndices = indices.slice(numVal);
            const valIndices = indices.slice(0, numVal);
            
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
            for (let i = 0; i < k; i++) {
                indices.push(Math.floor(Math.random() * numSamples));
            }
            return {
                xs: tf.gather(xs, indices),
                ys: tf.gather(ys, indices),
                indices
            };
        });
    }

    getRandomDenoisingBatch(xs, k = 5) {
        return tf.tidy(() => {
            const numSamples = xs.shape[0];
            const indices = [];
            for (let i = 0; i < k; i++) {
                indices.push(Math.floor(Math.random() * numSamples));
            }
            const original = tf.gather(xs, indices);
            const noisy = this.addNoise(original, 0.5);
            return { original, noisy, indices };
        });
    }

    draw28x28ToCanvas(tensor, canvas, scale = 4) {
        tf.tidy(() => {
            const img = tensor.squeeze();
            const data = img.mul(255).clipByValue(0, 255).dataSync();
            
            const ctx = canvas.getContext('2d');
            canvas.width = 28 * scale;
            canvas.height = 28 * scale;
            
            const imageData = ctx.createImageData(28, 28);
            for (let i = 0; i < 784; i++) {
                const val = data[i];
                imageData.data[i * 4] = val;
                imageData.data[i * 4 + 1] = val;
                imageData.data[i * 4 + 2] = val;
                imageData.data[i * 4 + 3] = 255;
            }
            
            const tempCanvas = document.createElement('canvas');
            tempCanvas.width = 28;
            tempCanvas.height = 28;
            const tempCtx = tempCanvas.getContext('2d');
            tempCtx.putImageData(imageData, 0, 0);
            
            ctx.imageSmoothingEnabled = false;
            ctx.drawImage(tempCanvas, 0, 0, canvas.width, canvas.height);
        });
    }

    dispose() {
        if (this.trainData) {
            this.trainData.xs.dispose();
            this.trainData.ys.dispose();
        }
        if (this.testData) {
            this.testData.xs.dispose();
            this.testData.ys.dispose();
        }
    }
}
