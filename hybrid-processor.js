// === Minimal FFT Implementation ===
class SimpleFFT {
    constructor(size) {
        if ((size & (size - 1)) !== 0) {
            throw new Error("FFT size must be power of 2");
        }
        this.size = size;
        this.cosTable = new Float32Array(size / 2);
        this.sinTable = new Float32Array(size / 2);
        for (let i = 0; i < size / 2; i++) {
            this.cosTable[i] = Math.cos(-2 * Math.PI * i / size);
            this.sinTable[i] = Math.sin(-2 * Math.PI * i / size);
        }
    }

    // In-place Cooley-Tukey FFT
    forward(re, im) {
        const n = this.size;
        if (re.length !== n || im.length !== n)
            throw new Error("Mismatched lengths");

        // Bit-reversal permutation
        let j = 0;
        for (let i = 0; i < n; i++) {
            if (i < j) {
                [re[i], re[j]] = [re[j], re[i]];
                [im[i], im[j]] = [im[j], im[i]];
            }
            let m = n >> 1;
            while (j >= m && m >= 2) {
                j -= m;
                m >>= 1;
            }
            j += m;
        }

        // FFT
        for (let size = 2; size <= n; size <<= 1) {
            const halfSize = size >> 1;
            const tableStep = n / size;
            for (let i = 0; i < n; i += size) {
                let k = 0;
                for (let j = i; j < i + halfSize; j++) {
                    const l = j + halfSize;
                    const tRe = re[l] * this.cosTable[k] - im[l] * this.sinTable[k];
                    const tIm = re[l] * this.sinTable[k] + im[l] * this.cosTable[k];
                    re[l] = re[j] - tRe;
                    im[l] = im[j] - tIm;
                    re[j] += tRe;
                    im[j] += tIm;
                    k += tableStep;
                }
            }
        }
    }

    // Inverse FFT
    inverse(re, im) {
        for (let i = 0; i < this.size; i++) {
            im[i] = -im[i];
        }
        this.forward(re, im);
        for (let i = 0; i < this.size; i++) {
            re[i] /= this.size;
            im[i] = -im[i] / this.size;
        }
    }
}


class HybridNoiseSuppressionProcessor extends AudioWorkletProcessor {
    constructor(options) {
        super();

        this.frameSize = 480;
        this.fftSize = 512;
        this.sampleRate = 48000;

        this.fft = new SimpleFFT(this.fftSize);
        this.re = new Float32Array(this.fftSize);
        this.im = new Float32Array(this.fftSize);

        this.inputBuffer = [];
        this.outputBuffer = [];

        this.initDSPComponents();

        this.noiseProfile = new Float32Array(this.frameSize / 2 + 1);
        this.noiseUpdateCount = 0;
        this.isNoiseEstimated = false;

        this.vadThreshold = 0.01;
        this.vadSmoothingFactor = 0.9;
        this.currentEnergy = 0;
        this.silenceCounter = 0;

        this.targetLevel = 0.1;
        this.agcGain = 1.0;
        this.agcSmoothingFactor = 0.99;

        this.port.onmessage = (e) => {
            const { type } = e.data;
            if (type === "init") {
                this.frameSize = e.data.frameSize || 480;
                this.fftSize = 512;
                this.fft = new SimpleFFT(this.fftSize);
                this.re = new Float32Array(this.fftSize);
                this.im = new Float32Array(this.fftSize);
                this.window = this.createHanningWindow(this.fftSize);
                this.overlapBuffer = new Float32Array(this.fftSize / 2);
            } else if (type === "ai_processed") {
                const aiProcessedFrame = new Float32Array(e.data.output);
                this.receiveProcessedFrame(aiProcessedFrame);
            }
        };
    }

    initDSPComponents() {
        this.hpf = {
            b0: 0.9968, b1: -1.9937, b2: 0.9968,
            a1: -1.9937, a2: 0.9937,
            x1: 0, x2: 0, y1: 0, y2: 0
        };

        this.window = this.createHanningWindow(this.fftSize);
        this.overlapBuffer = new Float32Array(this.fftSize / 2);
    }

    createHanningWindow(size) {
        const window = new Float32Array(size);
        for (let i = 0; i < size; i++) {
            window[i] = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / (size - 1));
        }
        return window;
    }

    highPassFilter(sample) {
        const output = this.hpf.b0 * sample + this.hpf.b1 * this.hpf.x1 + this.hpf.b2 * this.hpf.x2
                     - this.hpf.a1 * this.hpf.y1 - this.hpf.a2 * this.hpf.y2;
        this.hpf.x2 = this.hpf.x1;
        this.hpf.x1 = sample;
        this.hpf.y2 = this.hpf.y1;
        this.hpf.y1 = output;
        return output;
    }
        /**
     * Detects if a frame contains speech (voice activity) or not.
     *
     * @param {Float32Array} frame - Audio frame samples.
     * @param {number} sampleRate - Sampling rate (default 48000).
     * @returns {boolean} - True if voice is detected, else false.
     */
    detectVoiceActivity(frame, sampleRate = 48000) {
        if (!frame || frame.length === 0) return false;

        // --- Step 1: Compute RMS energy ---
        let energy = 0;
        for (let i = 0; i < frame.length; i++) {
            energy += frame[i] * frame[i];
        }
        energy = Math.sqrt(energy / frame.length);

        // --- Step 2: Compute Zero Crossing Rate (ZCR) ---
        let zcr = 0;
        for (let i = 1; i < frame.length; i++) {
            if ((frame[i - 1] >= 0 && frame[i] < 0) ||
                (frame[i - 1] < 0 && frame[i] >= 0)) {
                zcr++;
            }
        }
        zcr = zcr / frame.length;

        // --- Step 3: Apply thresholds ---
        // Tunable thresholds (you can tweak for sensitivity)
        const energyThreshold = 0.01;  // minimum energy to be considered speech
        const zcrThreshold = 0.1;      // speech has more zero crossings than silence

        const isSpeech = (energy > energyThreshold) && (zcr > zcrThreshold);
        return isSpeech;
    }


    // === NEW FFT-based spectral subtraction ===
    spectralSubtraction(frame) {
        const N = this.fftSize;

        // Fill re/im with windowed input
        for (let i = 0; i < N; i++) {
            this.re[i] = frame[i] * this.window[i];
            this.im[i] = 0;
        }

        // Forward FFT
        this.fft.forward(this.re, this.im);

        // Magnitude & phase
        const mag = new Float32Array(N/2 + 1);
        const phase = new Float32Array(N/2 + 1);
        for (let k = 0; k <= N/2; k++) {
            mag[k] = Math.sqrt(this.re[k] * this.re[k] + this.im[k] * this.im[k]);
            phase[k] = Math.atan2(this.im[k], this.re[k]);
        }

        // (apply spectral subtraction here…)

        // Reconstruct
        for (let k = 0; k <= N/2; k++) {
            this.re[k] = mag[k] * Math.cos(phase[k]);
            this.im[k] = mag[k] * Math.sin(phase[k]);
            if (k !== 0 && k !== N/2) { // mirror spectrum for real signal
                this.re[N-k] = this.re[k];
                this.im[N-k] = -this.im[k];
            }
        }

        // Inverse FFT
        this.fft.inverse(this.re, this.im);

        const reconstructed = new Float32Array(N);
        for (let i = 0; i < N; i++) {
            reconstructed[i] = this.re[i] * this.window[i];
        }

        // Overlap-add
        const outputFrame = new Float32Array(N);
        const halfSize = N / 2;
        for (let i = 0; i < halfSize; i++) {
            outputFrame[i] = reconstructed[i] + this.overlapBuffer[i];
        }
        for (let i = 0; i < halfSize; i++) {
            this.overlapBuffer[i] = reconstructed[i + halfSize];
            outputFrame[i + halfSize] = reconstructed[i + halfSize];
        }

        return outputFrame;
    }

    automaticGainControl(frame) {
    // frame: Float32Array of audio samples
    
    // Step 1: Calculate RMS energy of the frame
    let sumSquares = 0;
    for (let i = 0; i < frame.length; i++) {
        sumSquares += frame[i] * frame[i];
    }
    const rms = Math.sqrt(sumSquares / frame.length) || 1e-6;

    // Step 2: Define target RMS (desired loudness level)
    const targetRMS = 0.05; // tweakable (0.05 ≈ -26 dBFS)

    // Step 3: Compute gain factor
    let gain = targetRMS / rms;

    // Step 4: Smooth gain (avoid sudden jumps)
    const maxGain = 10.0;   // prevent extreme boosting
    const minGain = 0.1;    // prevent muting
    gain = Math.min(maxGain, Math.max(minGain, gain));

    // Step 5: Apply gain to frame
    for (let i = 0; i < frame.length; i++) {
        frame[i] *= gain;
    }

    return frame; // normalized output
}

    
    // Main DSP preprocessing pipeline
    preprocessFrame(frame) {
        // Create a copy to avoid modifying the original
        const processedFrame = new Float32Array(frame);
        
        // Step 1: High-pass filtering
        for (let i = 0; i < processedFrame.length; i++) {
            processedFrame[i] = this.highPassFilter(processedFrame[i]);
        }
        
        // Step 2: Voice Activity Detection
        const isVoiceActive = this.detectVoiceActivity(processedFrame);
        
        // Step 3: Noise gate - mute during silence
        if (!isVoiceActive && this.silenceCounter > 5) {
            const fadeOut = Math.max(0, 1 - (this.silenceCounter - 5) * 0.1);
            for (let i = 0; i < processedFrame.length; i++) {
                processedFrame[i] *= fadeOut;
            }
        }
        
        // Step 4: Spectral subtraction (skip if computationally expensive)
        // Comment out for now to test basic pipeline
        // processedFrame = this.spectralSubtraction(processedFrame);
        
        // Step 5: Automatic Gain Control
        this.automaticGainControl(processedFrame);
        
        return processedFrame;
    }
    
    process(inputs, outputs) {
        const input = inputs[0];
        const output = outputs[0];
        
        if (!input || input.length === 0) return true;
        
        const inputChannel = input[0];
        const outputChannel = output[0];
        
        // Accumulate input samples
        for (let i = 0; i < inputChannel.length; i++) {
            this.inputBuffer.push(inputChannel[i]);
        }
        
        // Process frames when we have enough samples
        while (this.inputBuffer.length >= this.frameSize) {
            // Extract frame
            const frame = new Float32Array(this.frameSize);
            for (let i = 0; i < this.frameSize; i++) {
                frame[i] = this.inputBuffer.shift();
            }
            
            // Apply DSP preprocessing
            const processedFrame = this.preprocessFrame(frame);
            
            // Send to main thread for AI processing
            this.port.postMessage({
                type: "process",
                buffer: processedFrame.buffer
            }, [processedFrame.buffer]);
        }
        
        // Output processed samples
        const samplesToOutput = Math.min(this.outputBuffer.length, outputChannel.length);
        for (let i = 0; i < samplesToOutput; i++) {
            outputChannel[i] = this.outputBuffer.shift() || 0;
        }
        
        // Zero-pad if needed
        for (let i = samplesToOutput; i < outputChannel.length; i++) {
            outputChannel[i] = 0;
        }
        
        return true;
    }
    
    // Receive processed frame from AI model
    receiveProcessedFrame(processedFrame) {
        // Add processed samples to output buffer
        for (let i = 0; i < processedFrame.length; i++) {
            this.outputBuffer.push(processedFrame[i]);
        }
    }
}

// Register the processor
registerProcessor("hybrid-processor", HybridNoiseSuppressionProcessor);

// ==================================================================
// MAIN THREAD INTEGRATION CODE
// ==================================================================

// Update your main.js integration:
