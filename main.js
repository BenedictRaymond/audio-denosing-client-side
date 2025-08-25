        let audioContext = null;
        let micStream = null;
        let sourceNode = null;
        let gainNode = null;
        let oscillator = null;
        let analyser = null;
        let volumeInterval = null;

        async function loadModel() {
            await wasm_bindgen();
            dfModule = wasm_bindgen;

            const modelPath = './DeepFilterNet3_onnx.tar.gz';
            const response = await fetch(modelPath);
            const modelBytes = new Uint8Array(await response.arrayBuffer());

            const attenLimDb = 20; // Attenuation limit in dB
            dfState = dfModule.df_create(modelBytes, attenLimDb);
            frameSize = dfModule.df_get_frame_length(dfState);
            dfModule.df_set_post_filter_beta(dfState, 1);
            setResult(1, 'Model loaded successfully');
        }
        

        function log(message) {
            console.log(message);
            const logEl = document.getElementById('log');
            logEl.textContent += new Date().toLocaleTimeString() + ': ' + message + '\n';
            logEl.scrollTop = logEl.scrollHeight;
        }

        function setResult(testId, message, isError = false) {
            const el = document.getElementById(`test${testId}-result`);
            el.innerHTML = `<p style="color: ${isError ? '#ff6666' : '#66ff66'}">${message}</p>`;
        }

        async function testDirectAudio() {

            audioContext = new (window.AudioContext || window.webkitAudioContext)();

            micStream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    channelCount: 1,
                    sampleRate: 48000,
                    echoCancellation: false,
                    noiseSuppression: false,
                    autoGainControl: false
                }
            });

            log('Creating direct audio connection...');
                
            sourceNode = audioContext.createMediaStreamSource(micStream);
            gainNode = audioContext.createGain();
            // Start with low volume to prevent feedback
            gainNode.gain.value = 1.5;
                
            sourceNode.connect(gainNode);
            gainNode.connect(audioContext.destination);
                
            log('Direct audio connection established');
            setResult(2, '✅ Direct audio started - You should hear your voice!');
        }

        function stopDirectAudio() {
            try {
                if (gainNode) {
                    gainNode.disconnect();
                    gainNode = null;
                }
                if (sourceNode) {
                    sourceNode.disconnect();
                    sourceNode = null;
                }
                log('Direct audio stopped');
                setResult(2, '⏹️ Direct audio stopped');
            } catch (error) {
                log('Error stopping direct audio: ' + error.message);
            }
        }

        async function initAudioGraph() {
            audioCtx = new (window.AudioContext || window.webkitAudioContext)({ latencyHint: "interactive" });


            // Load the worklet module
            await audioCtx.audioWorklet.addModule("./df-processor.js");


            // Create the Worklet node (mono)
            workletNode = new AudioWorkletNode(audioCtx, "df-processor", {
                numberOfInputs: 1,
                numberOfOutputs: 1,
                channelCount: 1,
                outputChannelCount: [1],
                processorOptions: {},
            });


            // Inform worklet of the frameSize we will use
            workletNode.port.postMessage({ type: "init", frameSize });


            // Connect mic → worklet → destination
            const stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: false } });
            const source = audioCtx.createMediaStreamSource(stream);
            source.connect(workletNode);
            workletNode.connect(audioCtx.destination);


            // Handle frames arriving from the worklet for processing
            workletNode.port.onmessage = (e) => {
                const { type } = e.data || {};
                if (type === "process") {
                    // Input frame arrives as a transferred ArrayBuffer
                    let inBuf = new Float32Array(e.data.buffer);
                    // Allocate output per call (transferable). If you prefer, pool these.
                    let outBuf = new Float32Array(frameSize);
                    let snr = new Float32Array(frameSize);


                    // Run DF denoiser. Some builds return SNR; we ignore it here.
                    // If your API is df_process(state, in, out) or returns the output, adapt accordingly.
                    try {
                        outBuf = dfModule.df_process_frame(dfState, inBuf);
                    } catch (err) {
                        console.error("DF processing error:", err);
                        outBuf.fill(0);
                    }
                    // outBuf = snr.map(sample => -sample);

                    workletNode.port.postMessage({ type: "processed", output: outBuf }, [outBuf.buffer]);

                }
            };
        }

        async function testDenoising() {
            await initAudioGraph();
            if (audioCtx.state === "suspended") await audioCtx.resume();
            console.log("Denoising started. Frame size:", frameSize);
            setResult(3, '✅ Denoising started - You should hear denoised audio!');
        }

        async function stopDenoising() {
            try {
                if (workletNode) {
                    workletNode.disconnect();
                    workletNode = null;
                }
                if (audioCtx) {
                    await audioCtx.close();
                    audioCtx = null;
                }
                setResult(3, '✅ Denoising stopped.');
            } catch (e) {
                console.warn("stopDenoising cleanup warning:", e);
            }
        }
        // Cleanup on page unload
        window.addEventListener('beforeunload', () => {
            stopDirectAudio();
            stopDenoising();
            if (micStream) {
                micStream.getTracks().forEach(track => track.stop());
            }
            if (audioContext) {
                audioContext.close();
            }
        });
