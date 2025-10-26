/**
 * VoiceMind AI - Advanced Waveform Visualizer
 * Real-time audio waveform visualization with modern animations
 */

class WaveformVisualizer {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');
        this.audioContext = null;
        this.analyser = null;
        this.dataArray = null;
        this.animationId = null;
        this.isVisualizing = false;
        
        // Visual settings
        this.settings = {
            barWidth: 3,
            barGap: 1,
            barColor: '#6366f1',
            barColorActive: '#f59e0b',
            backgroundColor: '#f8fafc',
            gridColor: '#e2e8f0',
            maxBars: 200,
            smoothing: 0.8,
            animationSpeed: 0.1
        };
        
        this.initializeCanvas();
    }

    initializeCanvas() {
        // Set canvas size
        this.canvas.width = this.canvas.offsetWidth * window.devicePixelRatio;
        this.canvas.height = this.canvas.offsetHeight * window.devicePixelRatio;
        this.ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
        
        // Set canvas style
        this.canvas.style.width = this.canvas.offsetWidth + 'px';
        this.canvas.style.height = this.canvas.offsetHeight + 'px';
        
        // Draw initial state
        this.drawIdleState();
        
        // Handle resize
        window.addEventListener('resize', this.handleResize.bind(this));
    }

    handleResize() {
        this.canvas.width = this.canvas.offsetWidth * window.devicePixelRatio;
        this.canvas.height = this.canvas.offsetHeight * window.devicePixelRatio;
        this.ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
        
        this.canvas.style.width = this.canvas.offsetWidth + 'px';
        this.canvas.style.height = this.canvas.offsetHeight + 'px';
        
        if (!this.isVisualizing) {
            this.drawIdleState();
        }
    }

    async startVisualization(stream) {
        try {
            // Create audio context
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
            
            // Create analyser
            this.analyser = this.audioContext.createAnalyser();
            this.analyser.fftSize = 2048;
            this.analyser.smoothingTimeConstant = this.settings.smoothing;
            
            // Connect stream to analyser
            const source = this.audioContext.createMediaStreamSource(stream);
            source.connect(this.analyser);
            
            // Initialize data array
            const bufferLength = this.analyser.frequencyBinCount;
            this.dataArray = new Uint8Array(bufferLength);
            
            this.isVisualizing = true;
            this.animate();
            
            // Waveform visualization started
            
        } catch (error) {
            console.error('Failed to start waveform visualization:', error);
        }
    }

    stopVisualization() {
        this.isVisualizing = false;
        
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
            this.animationId = null;
        }
        
        if (this.audioContext) {
            this.audioContext.close();
            this.audioContext = null;
        }
        
        this.drawIdleState();
        // Waveform visualization stopped
    }

    animate() {
        if (!this.isVisualizing) return;
        
        this.analyser.getByteFrequencyData(this.dataArray);
        this.draw();
        
        this.animationId = requestAnimationFrame(() => this.animate());
    }

    draw() {
        const canvasWidth = this.canvas.offsetWidth;
        const canvasHeight = this.canvas.offsetHeight;
        
        // Clear canvas
        this.ctx.clearRect(0, 0, canvasWidth, canvasHeight);
        
        // Draw background
        this.drawBackground(canvasWidth, canvasHeight);
        
        // Draw grid
        this.drawGrid(canvasWidth, canvasHeight);
        
        // Draw waveform
        this.drawWaveform(canvasWidth, canvasHeight);
        
        // Draw center line
        this.drawCenterLine(canvasWidth, canvasHeight);
    }

    drawBackground(width, height) {
        // Gradient background
        const gradient = this.ctx.createLinearGradient(0, 0, 0, height);
        gradient.addColorStop(0, '#f8fafc');
        gradient.addColorStop(1, '#f1f5f9');
        
        this.ctx.fillStyle = gradient;
        this.ctx.fillRect(0, 0, width, height);
        
        // Border
        this.ctx.strokeStyle = this.settings.gridColor;
        this.ctx.lineWidth = 1;
        this.ctx.strokeRect(0, 0, width, height);
    }

    drawGrid(width, height) {
        this.ctx.strokeStyle = this.settings.gridColor;
        this.ctx.lineWidth = 0.5;
        this.ctx.setLineDash([2, 2]);
        
        // Horizontal grid lines
        const gridSpacing = height / 8;
        for (let i = 1; i < 8; i++) {
            const y = i * gridSpacing;
            this.ctx.beginPath();
            this.ctx.moveTo(0, y);
            this.ctx.lineTo(width, y);
            this.ctx.stroke();
        }
        
        // Vertical grid lines
        const verticalSpacing = width / 10;
        for (let i = 1; i < 10; i++) {
            const x = i * verticalSpacing;
            this.ctx.beginPath();
            this.ctx.moveTo(x, 0);
            this.ctx.lineTo(x, height);
            this.ctx.stroke();
        }
        
        this.ctx.setLineDash([]);
    }

    drawWaveform(width, height) {
        if (!this.dataArray) return;
        
        const barWidth = this.settings.barWidth;
        const barGap = this.settings.barGap;
        const totalBarWidth = barWidth + barGap;
        const maxBars = Math.min(this.settings.maxBars, Math.floor(width / totalBarWidth));
        
        // Calculate which frequency bins to use
        const binSize = Math.floor(this.dataArray.length / maxBars);
        const centerY = height / 2;
        
        for (let i = 0; i < maxBars; i++) {
            const startBin = i * binSize;
            const endBin = Math.min(startBin + binSize, this.dataArray.length);
            
            // Calculate average amplitude for this bar
            let sum = 0;
            for (let j = startBin; j < endBin; j++) {
                sum += this.dataArray[j];
            }
            const amplitude = sum / (endBin - startBin);
            
            // Normalize amplitude (0-255 to 0-1)
            const normalizedAmplitude = amplitude / 255;
            
            // Calculate bar height
            const barHeight = Math.max(2, normalizedAmplitude * (height * 0.8));
            
            // Calculate position
            const x = i * totalBarWidth + barGap;
            const y = centerY - barHeight / 2;
            
            // Choose color based on amplitude
            const color = normalizedAmplitude > 0.7 ? this.settings.barColorActive : this.settings.barColor;
            
            // Draw bar with gradient
            this.drawBar(x, y, barWidth, barHeight, color, normalizedAmplitude);
        }
    }

    drawBar(x, y, width, height, color, intensity) {
        // Create gradient for the bar
        const gradient = this.ctx.createLinearGradient(0, y, 0, y + height);
        gradient.addColorStop(0, this.adjustColorOpacity(color, 0.8));
        gradient.addColorStop(0.5, color);
        gradient.addColorStop(1, this.adjustColorOpacity(color, 0.6));
        
        this.ctx.fillStyle = gradient;
        this.ctx.fillRect(x, y, width, height);
        
        // Add highlight for high intensity
        if (intensity > 0.5) {
            this.ctx.fillStyle = this.adjustColorOpacity('#ffffff', 0.3);
            this.ctx.fillRect(x, y, width, height * 0.3);
        }
    }

    drawCenterLine(width, height) {
        const centerY = height / 2;
        
        this.ctx.strokeStyle = this.settings.gridColor;
        this.ctx.lineWidth = 1;
        this.ctx.setLineDash([1, 1]);
        
        this.ctx.beginPath();
        this.ctx.moveTo(0, centerY);
        this.ctx.lineTo(width, centerY);
        this.ctx.stroke();
        
        this.ctx.setLineDash([]);
    }

    drawIdleState() {
        const canvasWidth = this.canvas.offsetWidth;
        const canvasHeight = this.canvas.offsetHeight;
        
        // Clear canvas
        this.ctx.clearRect(0, 0, canvasWidth, canvasHeight);
        
        // Draw background
        this.drawBackground(canvasWidth, canvasHeight);
        
        // Draw grid
        this.drawGrid(canvasWidth, canvasHeight);
        
        // Draw idle message
        this.ctx.fillStyle = this.settings.gridColor;
        this.ctx.font = '14px Inter, sans-serif';
        this.ctx.textAlign = 'center';
        this.ctx.textBaseline = 'middle';
        this.ctx.fillText('Ready to record...', canvasWidth / 2, canvasHeight / 2);
        
        // Draw center line
        this.drawCenterLine(canvasWidth, canvasHeight);
    }

    adjustColorOpacity(color, opacity) {
        // Convert hex to rgba
        const hex = color.replace('#', '');
        const r = parseInt(hex.substr(0, 2), 16);
        const g = parseInt(hex.substr(2, 2), 16);
        const b = parseInt(hex.substr(4, 2), 16);
        
        return `rgba(${r}, ${g}, ${b}, ${opacity})`;
    }

    // Public methods for customization
    updateSettings(newSettings) {
        this.settings = { ...this.settings, ...newSettings };
    }

    getSettings() {
        return { ...this.settings };
    }

    // Method to draw a static waveform from audio data
    drawStaticWaveform(audioData, width, height) {
        this.ctx.clearRect(0, 0, width, height);
        this.drawBackground(width, height);
        this.drawGrid(width, height);
        
        if (!audioData || audioData.length === 0) {
            this.drawIdleState();
            return;
        }
        
        const centerY = height / 2;
        const barWidth = this.settings.barWidth;
        const barGap = this.settings.barGap;
        const totalBarWidth = barWidth + barGap;
        const maxBars = Math.min(audioData.length, Math.floor(width / totalBarWidth));
        
        for (let i = 0; i < maxBars; i++) {
            const amplitude = audioData[i] / 255;
            const barHeight = Math.max(2, amplitude * (height * 0.8));
            
            const x = i * totalBarWidth + barGap;
            const y = centerY - barHeight / 2;
            
            const color = amplitude > 0.7 ? this.settings.barColorActive : this.settings.barColor;
            this.drawBar(x, y, barWidth, barHeight, color, amplitude);
        }
        
        this.drawCenterLine(width, height);
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = WaveformVisualizer;
}