class DFProcessor extends AudioWorkletProcessor {
  constructor() {
    super();


    // Defaults; main thread will send real frameSize via "init"
    this.frameSize = 480;


    // Input frame assembly
    this.inputBuffer = new Float32Array(this.frameSize);
    this.inputBufferIdx = 0;


    // Output queue of processed frames (each is Float32Array length=frameSize)
    this.outputQueue = [];
    this.currentOutFrame = null; // Float32Array
    this.currentOutIdx = 0; // read index within currentOutFrame


    // Stats (optional)
    this.processedFrames = 0;


    this.port.onmessage = (event) => {
      const { type } = event.data || {};
      if (type === "init") {
        const { frameSize } = event.data;
        if (Number.isInteger(frameSize) && frameSize > 0) {
          this.frameSize = frameSize;
        }
        this.inputBuffer = new Float32Array(this.frameSize);
        this.inputBufferIdx = 0;
        this.outputQueue = [];
        this.currentOutFrame = null;
        this.currentOutIdx = 0;
      } else if (type === "processed") {
        // Receive a denoised frame from main thread; enqueue for playback
        if (event.data.type === "processed") {
          const outBuf = new Float32Array(event.data.output);
          this.outputQueue.push(outBuf); // enqueue for playback
        }
      };
    }
  }


  _emitFrameToMain() {
    // Send a full input frame to main thread for denoising
    const frame = this.inputBuffer.slice(0); // copy to detach from circular buffer
    // Transfer underlying buffer to avoid copy on the way out
    this.port.postMessage({ type: "process", buffer: frame.buffer }, [frame.buffer]);
    this.inputBufferIdx = 0;
    this.processedFrames++;
  }


  _readFromQueue(outputChannel) {

    let written = 0;

    while (written < outputChannel.length) {
      if (!this.currentOutFrame || this.currentOutIdx >= this.currentOutFrame.length) {
        this.currentOutFrame = this.outputQueue.shift() || null;
        this.currentOutIdx = 0;
        if (!this.currentOutFrame) break; // nothing available
      }
      const available = this.currentOutFrame.length - this.currentOutIdx;
      const toCopy = Math.min(available, outputChannel.length - written);
      outputChannel.set(
        this.currentOutFrame.subarray(this.currentOutIdx, this.currentOutIdx + toCopy),
        written
      );
      this.currentOutIdx += toCopy;
      written += toCopy;
    }
    // If we couldn't fill the whole output (pipeline warmup), zero the rest
    if (written < outputChannel.length) {
    outputChannel.fill(0, written);
    }
  }

  process(inputs, outputs) {
    const input = inputs[0];
    const output = outputs[0];


    if (!input || !input.length || !output || !output.length) return true;


    const inputChannel = input[0]; // mono
    const outputChannel = output[0]; // mono


    // 1) Write available input samples into the input frame buffer
    let readIdx = 0;
    while (readIdx < inputChannel.length) {
      const needed = this.frameSize - this.inputBufferIdx;
      const available = inputChannel.length - readIdx;
      const toCopy = Math.min(needed, available);
      // Copy into inputBuffer
      this.inputBuffer.set(
        inputChannel.subarray(readIdx, readIdx + toCopy),
        this.inputBufferIdx
      );
      this.inputBufferIdx += toCopy;
      readIdx += toCopy;


      // When a full frame is ready, ship it to main thread for denoising
      if (this.inputBufferIdx === this.frameSize) {
        this._emitFrameToMain();
      }
    }


    // 2) Pull denoised samples from the output queue to the audio graph
    this._readFromQueue(outputChannel);


    return true; // keep processor alive
  }
}

registerProcessor("df-processor", DFProcessor);