// Hybrid Noise Suppression System: Traditional DSP + AI Model
// Combines preprocessing DSP filters with DeepFilterNet for optimal results

class HybridNoiseSuppressionProcessor extends AudioWorkletProcessor {
    constructor(options) {
        super();
        
        // Frame parameters
        this.frameSize = 480; // 10ms at 48kHz
        this.sampleRate = 48000;
        
        // Buffer for frame accumulation
        this.inputBuffer = [];
        this.outputBuffer = [];
        
        // DSP Components
        this.initDSPComponents();
        
        // Noise estimation for spectral subtraction
        this.noiseProfile = new Float32Array(this.frameSize / 2 + 1);
        this.noiseUpdateCount = 0;
        this.isNoiseEstimated = false;
        
        // VAD parameters
        this.vadThreshold = 0.01;
        this.vadSmoothingFactor = 0.9;
        this.currentEnergy = 0;
        this.silenceCounter = 0;
        
        // AGC parameters
        this.targetLevel = 0.1;
        this.agcGain = 1.0;
        this.agcSmoothingFactor = 0.99;
        
        // Listen for messages from main thread
        this.port.onmessage = (e) => {
            const { type } = e.data;
            if (type === "init") {
                this.frameSize = e.data.frameSize || 480;
                this.initDSPComponents();
            } else if (type === "ai_processed") {
                // Receive AI-processed frame and add to output buffer
                const aiProcessedFrame = new Float32Array(e.data.output);
                this.receiveProcessedFrame(aiProcessedFrame);
            }
        };
    }
    
    initDSPComponents() {
        // High-pass filter coefficients (Butterworth, 60Hz cutoff at 48kHz)
        this.hpf = {
            b0: 0.9968, b1: -1.9937, b2: 0.9968,
            a1: -1.9937, a2: 0.9937,
            x1: 0, x2: 0, y1: 0, y2: 0
        };
        
        // FFT setup for spectral processing
        this.fftSize = this.frameSize;
        this.window = this.createHanningWindow(this.fftSize);
        
        // Overlap-add buffers
        this.overlapBuffer = new Float32Array(this.fftSize / 2);
    }
    
    createHanningWindow(size) {
        const window = new Float32Array(size);
        for (let i = 0; i < size; i++) {
            window[i] = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / (size - 1));
        }
        return window;
    }
    
    // High-pass filter implementation
    highPassFilter(sample) {
        const output = this.hpf.b0 * sample + this.hpf.b1 * this.hpf.x1 + this.hpf.b2 * this.hpf.x2
                      - this.hpf.a1 * this.hpf.y1 - this.hpf.a2 * this.hpf.y2;
        
        this.hpf.x2 = this.hpf.x1;
        this.hpf.x1 = sample;
        this.hpf.y2 = this.hpf.y1;
        this.hpf.y1 = output;
        
        return output;
    }
    
    // Voice Activity Detection
    detectVoiceActivity(frame) {
        // Calculate energy
        let energy = 0;
        for (let i = 0; i < frame.length; i++) {
            energy += frame[i] * frame[i];
        }
        energy = Math.sqrt(energy / frame.length);
        
        // Smooth energy estimate
        this.currentEnergy = this.vadSmoothingFactor * this.currentEnergy + 
                            (1 - this.vadSmoothingFactor) * energy;
        
        const isVoiceActive = this.currentEnergy > this.vadThreshold;
        
        if (!isVoiceActive) {
            this.silenceCounter++;
        } else {
            this.silenceCounter = 0;
        }
        
        return isVoiceActive;
    }
    
    // Automatic Gain Control
    automaticGainControl(frame) {
        // Calculate RMS level
        let rms = 0;
        for (let i = 0; i < frame.length; i++) {
            rms += frame[i] * frame[i];
        }
        rms = Math.sqrt(rms / frame.length);
        
        if (rms > 0.001) { // Avoid division by zero
            const targetGain = this.targetLevel / rms;
            this.agcGain = this.agcSmoothingFactor * this.agcGain + 
                          (1 - this.agcSmoothingFactor) * Math.min(targetGain, 5.0);
        }
        
        // Apply gain with limiting
        for (let i = 0; i < frame.length; i++) {
            frame[i] = Math.max(-1, Math.min(1, frame[i] * this.agcGain));
        }
        
        return frame;
    }
    
    // Simplified DFT for spectral analysis (works with any size)
    dft(signal) {
        const N = signal.length;
        const magnitude = new Float32Array(N / 2 + 1);
        const phase = new Float32Array(N / 2 + 1);
        
        for (let k = 0; k <= N / 2; k++) {
            let real = 0, imag = 0;
            for (let n = 0; n < N; n++) {
                const angle = -2 * Math.PI * k * n / N;
                real += signal[n] * Math.cos(angle);
                imag += signal[n] * Math.sin(angle);
            }
            magnitude[k] = Math.sqrt(real * real + imag * imag);
            phase[k] = Math.atan2(imag, real);
        }
        
        return { magnitude, phase };
    }
    
    // Inverse DFT for signal reconstruction
    idft(magnitude, phase) {
        const N = (magnitude.length - 1) * 2;
        const signal = new Float32Array(N);
        
        for (let n = 0; n < N; n++) {
            let sample = 0;
            for (let k = 0; k < magnitude.length; k++) {
                const angle = 2 * Math.PI * k * n / N;
                const real = magnitude[k] * Math.cos(phase[k]);
                const imag = magnitude[k] * Math.sin(phase[k]);
                sample += real * Math.cos(angle) - imag * Math.sin(angle);
            }
            signal[n] = sample / N;
        }
        
        return signal;
    }
    
    // Spectral Subtraction with proper IDFT
    spectralSubtraction(frame) {
        // Apply window
        const windowed = new Float32Array(frame.length);
        for (let i = 0; i < frame.length; i++) {
            windowed[i] = frame[i] * this.window[i];
        }
        
        // Forward DFT
        const { magnitude, phase } = this.dft(windowed);
        
        // Update noise profile during silence
        if (this.silenceCounter > 10 && this.noiseUpdateCount < 100) {
            const alpha = 0.1;
            for (let i = 0; i < this.noiseProfile.length; i++) {
                this.noiseProfile[i] = alpha * magnitude[i] + (1 - alpha) * this.noiseProfile[i];
            }
            this.noiseUpdateCount++;
            this.isNoiseEstimated = true;
        }
        
        // Apply spectral subtraction if noise profile is available
        if (this.isNoiseEstimated) {
            const alpha = 2.0; // Over-subtraction factor
            const beta = 0.1;  // Spectral floor factor
            
            for (let i = 0; i < magnitude.length; i++) {
                const subtracted = magnitude[i] - alpha * this.noiseProfile[i];
                magnitude[i] = Math.max(subtracted, beta * magnitude[i]);
            }
        }
        
        // Inverse DFT to reconstruct signal
        const reconstructed = this.idft(magnitude, phase);
        
        // Apply window again and overlap-add
        for (let i = 0; i < reconstructed.length; i++) {
            reconstructed[i] *= this.window[i];
        }
        
        // Overlap-add with previous frame
        const output_frame = new Float32Array(this.fftSize);
        const halfSize = this.fftSize / 2;
        
        // Add overlap from previous frame
        for (let i = 0; i < halfSize; i++) {
            output_frame[i] = reconstructed[i] + this.overlapBuffer[i];
        }
        
        // Store current frame's second half for next overlap
        for (let i = 0; i < halfSize; i++) {
            this.overlapBuffer[i] = reconstructed[i + halfSize];
            output_frame[i + halfSize] = reconstructed[i + halfSize];
        }
        
        return output_frame;
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
        //processedFrame = this.spectralSubtraction(processedFrame);
        
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
