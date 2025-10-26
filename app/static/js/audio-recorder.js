/**
 * VoiceMind AI - Advanced Audio Recording System
 * Handles high-quality audio recording with real-time visualization
 */

class AudioRecorder {
    constructor() {
        this.mediaRecorder = null;
        this.audioContext = null;
        this.analyser = null;
        this.microphone = null;
        this.dataArray = null;
        this.recordingChunks = [];
        this.isRecording = false;
        this.recordingStartTime = null;
        this.sampleRate = 44100;
        this.bitRate = 128000;
    }

    async initialize() {
        try {
            // Request microphone access
            const stream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true,
                    sampleRate: this.sampleRate
                }
            });

            // Initialize audio context
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)({
                sampleRate: this.sampleRate
            });

            // Create analyser for real-time audio analysis
            this.analyser = this.audioContext.createAnalyser();
            this.analyser.fftSize = 2048;
            this.analyser.smoothingTimeConstant = 0.8;

            // Connect microphone to analyser
            this.microphone = this.audioContext.createMediaStreamSource(stream);
            this.microphone.connect(this.analyser);

            // Initialize data array for frequency analysis
            const bufferLength = this.analyser.frequencyBinCount;
            this.dataArray = new Uint8Array(bufferLength);

            // Initialize media recorder
            this.mediaRecorder = new MediaRecorder(stream, {
                mimeType: this.getSupportedMimeType(),
                audioBitsPerSecond: this.bitRate
            });

            this.setupMediaRecorderEvents();

            return true;
        } catch (error) {
            console.error('Failed to initialize audio recorder:', error);
            throw new Error('Microphone access denied or not available');
        }
    }

    getSupportedMimeType() {
        const types = [
            'audio/webm;codecs=opus',
            'audio/webm',
            'audio/mp4',
            'audio/ogg;codecs=opus',
            'audio/wav'
        ];

        for (const type of types) {
            if (MediaRecorder.isTypeSupported(type)) {
                return type;
            }
        }

        return 'audio/webm'; // Fallback
    }

    setupMediaRecorderEvents() {
        this.mediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) {
                this.recordingChunks.push(event.data);
            }
        };

        this.mediaRecorder.onstop = () => {
            this.processRecording();
        };

        this.mediaRecorder.onerror = (event) => {
            console.error('MediaRecorder error:', event.error);
            this.handleRecordingError(event.error);
        };
    }

    async startRecording() {
        if (!this.mediaRecorder) {
            await this.initialize();
        }

        if (this.mediaRecorder.state === 'inactive') {
            this.recordingChunks = [];
            this.isRecording = true;
            this.recordingStartTime = Date.now();
            
            // Start recording with time slices for real-time processing
            this.mediaRecorder.start(100);
            
            // Recording started
            return true;
        }

        return false;
    }

    stopRecording() {
        if (this.mediaRecorder && this.mediaRecorder.state === 'recording') {
            this.mediaRecorder.stop();
            this.isRecording = false;
            // Recording stopped
            return true;
        }

        return false;
    }

    processRecording() {
        if (this.recordingChunks.length === 0) {
            console.error('No recording data available');
            return null;
        }

        const mimeType = this.mediaRecorder.mimeType || 'audio/webm';
        const audioBlob = new Blob(this.recordingChunks, { type: mimeType });
        
        // Create audio file with proper naming
        const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
        const audioFile = new File([audioBlob], `recording-${timestamp}.webm`, {
            type: mimeType,
            lastModified: Date.now()
        });

        // Recording processed successfully

        return audioFile;
    }

    getRecordingDuration() {
        if (!this.recordingStartTime) return 0;
        return (Date.now() - this.recordingStartTime) / 1000;
    }

    getAudioLevel() {
        if (!this.analyser || !this.dataArray) return 0;

        this.analyser.getByteFrequencyData(this.dataArray);
        
        // Calculate RMS (Root Mean Square) for audio level
        let sum = 0;
        for (let i = 0; i < this.dataArray.length; i++) {
            sum += this.dataArray[i] * this.dataArray[i];
        }
        
        const rms = Math.sqrt(sum / this.dataArray.length);
        return rms / 255; // Normalize to 0-1
    }

    getFrequencyData() {
        if (!this.analyser || !this.dataArray) return null;

        this.analyser.getByteFrequencyData(this.dataArray);
        return Array.from(this.dataArray);
    }

    getTimeDomainData() {
        if (!this.analyser) return null;

        const bufferLength = this.analyser.frequencyBinCount;
        const dataArray = new Uint8Array(bufferLength);
        this.analyser.getByteTimeDomainData(dataArray);
        
        return Array.from(dataArray);
    }

    handleRecordingError(error) {
        console.error('Recording error:', error);
        
        let errorMessage = 'Recording failed. ';
        switch (error.name) {
            case 'NotAllowedError':
                errorMessage += 'Microphone access denied.';
                break;
            case 'NotFoundError':
                errorMessage += 'No microphone found.';
                break;
            case 'NotReadableError':
                errorMessage += 'Microphone is being used by another application.';
                break;
            default:
                errorMessage += 'Please try again.';
        }

        // Dispatch custom event for error handling
        const errorEvent = new CustomEvent('recordingError', {
            detail: { error, message: errorMessage }
        });
        document.dispatchEvent(errorEvent);
    }

    cleanup() {
        if (this.mediaRecorder && this.mediaRecorder.state === 'recording') {
            this.mediaRecorder.stop();
        }

        if (this.audioContext && this.audioContext.state !== 'closed') {
            this.audioContext.close();
        }

        if (this.microphone) {
            this.microphone.disconnect();
        }

        this.isRecording = false;
        this.recordingChunks = [];
        // Audio recorder cleaned up
    }

    // Utility methods for audio analysis
    detectSilence(threshold = 0.01, duration = 1000) {
        const audioLevel = this.getAudioLevel();
        return audioLevel < threshold;
    }

    getAudioQuality() {
        if (!this.analyser || !this.dataArray) return 'unknown';

        const audioLevel = this.getAudioLevel();
        
        if (audioLevel < 0.01) return 'silent';
        if (audioLevel < 0.1) return 'low';
        if (audioLevel < 0.5) return 'good';
        if (audioLevel < 0.8) return 'high';
        return 'very_high';
    }

    // Real-time audio monitoring
    startMonitoring(callback) {
        if (!this.analyser) return;

        const monitor = () => {
            if (this.isRecording) {
                const audioLevel = this.getAudioLevel();
                const frequencyData = this.getFrequencyData();
                const timeDomainData = this.getTimeDomainData();
                
                callback({
                    audioLevel,
                    frequencyData,
                    timeDomainData,
                    quality: this.getAudioQuality(),
                    isSilent: this.detectSilence()
                });
                
                requestAnimationFrame(monitor);
            }
        };

        monitor();
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = AudioRecorder;
}