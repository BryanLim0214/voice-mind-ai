/**
 * VoiceMind AI - Modern Web Application Logic
 * Advanced mental health screening through voice analysis
 */

// Polyfill for older browsers
if (!navigator.mediaDevices) {
    navigator.mediaDevices = {};
}

if (!navigator.mediaDevices.getUserMedia) {
    navigator.mediaDevices.getUserMedia = function(constraints) {
        const getUserMedia = navigator.getUserMedia || navigator.webkitGetUserMedia || navigator.mozGetUserMedia || navigator.msGetUserMedia;
        
        if (!getUserMedia) {
            return Promise.reject(new Error('getUserMedia is not supported in this browser'));
        }
        
        return new Promise(function(resolve, reject) {
            getUserMedia.call(navigator, constraints, resolve, reject);
        });
    };
}

class VoiceMindApp {
    constructor() {
        this.audioRecorder = null;
        this.audioContext = null;
        this.mediaRecorder = null;
        this.audioChunks = [];
        this.isRecording = false;
        this.recordingTimer = null;
        this.recordingStartTime = null;
        this.audioFile = null;
        this.analysisResults = null;
        
        this.initializeApp();
    }

    initializeApp() {
        this.checkBrowserCompatibility();
        this.setupEventListeners();
        this.initializeAudioRecorder();
        this.setupFileUpload();
        this.initializeWaveform();
        this.showWelcomeMessage();
    }

    checkBrowserCompatibility() {
        // Check if we're on HTTPS or localhost
        const isSecure = location.protocol === 'https:' || location.hostname === 'localhost' || location.hostname === '127.0.0.1';
        
        if (!isSecure) {
            this.showError('Microphone access requires HTTPS or localhost. Please access this site via localhost:5000 instead of your IP address.');
            this.showRecordingNotice();
            return false;
        }
        
        // Check for required APIs
        const hasGetUserMedia = !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
        const hasMediaRecorder = !!window.MediaRecorder;
        
        if (!hasGetUserMedia) {
            // Try legacy getUserMedia
            const legacyGetUserMedia = navigator.getUserMedia || navigator.webkitGetUserMedia || navigator.mozGetUserMedia || navigator.msGetUserMedia;
            if (!legacyGetUserMedia) {
                this.showError('Your browser does not support microphone access. Please use Chrome, Firefox, or Safari.');
                return false;
            }
        }
        
        if (!hasMediaRecorder) {
            this.showError('Your browser does not support audio recording. Please use Chrome, Firefox, or Safari.');
            return false;
        }
        
        // Browser compatibility verified
        return true;
    }

    showRecordingNotice() {
        const notice = document.createElement('div');
        notice.className = 'recording-notice';
        notice.innerHTML = `
            <div class="notice-content">
                <h3><i class="fas fa-microphone-slash"></i> Microphone Access Notice</h3>
                <p>To use the recording feature, please access this site using one of these URLs:</p>
                <ul>
                    <li><strong>localhost:5000</strong> (recommended)</li>
                    <li><strong>127.0.0.1:5000</strong></li>
                </ul>
                <p>Chrome requires HTTPS for microphone access when using IP addresses. Using localhost bypasses this requirement.</p>
                <button onclick="window.location.href='http://localhost:5000'" class="medical-btn primary">
                    <i class="fas fa-external-link-alt"></i> Switch to localhost
                </button>
            </div>
        `;
        
        // Style the notice
        notice.style.cssText = `
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: white;
            border: 2px solid #ef4444;
            border-radius: 12px;
            padding: 30px;
            box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1);
            z-index: 1000;
            max-width: 500px;
            text-align: center;
        `;
        
        document.body.appendChild(notice);
        
        // Add backdrop
        const backdrop = document.createElement('div');
        backdrop.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.5);
            z-index: 999;
        `;
        document.body.appendChild(backdrop);
    }

    setupEventListeners() {
        // File upload events
        const fileUploadArea = document.getElementById('fileUploadArea');
        const audioFileInput = document.getElementById('audioFileInput');

        if (fileUploadArea) {
            fileUploadArea.addEventListener('dragover', this.handleDragOver.bind(this));
            fileUploadArea.addEventListener('dragleave', this.handleDragLeave.bind(this));
            fileUploadArea.addEventListener('drop', this.handleDrop.bind(this));
        }
        
        if (audioFileInput) {
            audioFileInput.addEventListener('change', this.handleFileSelect.bind(this));
        }
        
        // File input is now handled by the label element

        // Recording events
        const recordButton = document.getElementById('recordButton');
        const stopButton = document.getElementById('stopButton');

        if (recordButton) {
            recordButton.addEventListener('click', this.startRecording.bind(this));
        }
        if (stopButton) {
            stopButton.addEventListener('click', this.stopRecording.bind(this));
        }

        // Download report
        const downloadReportBtn = document.getElementById('downloadReport');
        if (downloadReportBtn) {
            downloadReportBtn.addEventListener('click', this.downloadReport.bind(this));
        }
    }

    initializeAudioRecorder() {
        this.audioRecorder = new AudioRecorder();
    }

    setupFileUpload() {
        // File upload system with drag and drop support
    }

    initializeWaveform() {
        this.waveform = new WaveformVisualizer('waveformCanvas');
    }

    showWelcomeMessage() {
        this.updateStatus('System Ready', 'success');
    }

    // File Upload Methods
    handleDragOver(e) {
        e.preventDefault();
        e.currentTarget.classList.add('dragover');
    }

    handleDragLeave(e) {
        e.preventDefault();
        e.currentTarget.classList.remove('dragover');
    }

    handleDrop(e) {
        e.preventDefault();
        e.currentTarget.classList.remove('dragover');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            this.processAudioFile(files[0]);
        }
    }

    handleFileSelect(e) {
        const file = e.target.files[0];
        if (file) {
            this.processAudioFile(file);
        }
    }

    processAudioFile(file) {
        // Validate file type
        const validTypes = ['audio/wav', 'audio/mp3', 'audio/flac', 'audio/ogg', 'audio/webm'];
        const validExtensions = ['.wav', '.mp3', '.flac', '.ogg', '.webm'];
        
        // Check MIME type or file extension
        const hasValidType = validTypes.includes(file.type);
        const hasValidExtension = validExtensions.some(ext => file.name.toLowerCase().endsWith(ext));
        
        if (!hasValidType && !hasValidExtension) {
            this.showError('Please select a valid audio file (WAV, MP3, FLAC, OGG, WEBM)');
            return;
        }

        // Validate file size (max 50MB)
        if (file.size > 50 * 1024 * 1024) {
            this.showError('File size too large. Please select a file smaller than 50MB.');
            return;
        }

        this.audioFile = file;
        this.displayAudioInfo(file);
        this.showAnalysisButton();
        this.updateStatus('Audio file ready for analysis', 'success');
    }

    displayAudioInfo(file) {
        const audioInfo = document.getElementById('audioInfo');
        const audioDetails = document.getElementById('audioDetails');
        
        const fileSize = (file.size / (1024 * 1024)).toFixed(2);
        const fileType = file.type.split('/')[1].toUpperCase();
        
        audioDetails.innerHTML = `
            <div class="info-grid">
                <div class="info-item">
                    <strong>File Name:</strong> ${file.name}
                </div>
                <div class="info-item">
                    <strong>File Size:</strong> ${fileSize} MB
                </div>
                <div class="info-item">
                    <strong>File Type:</strong> ${fileType}
                </div>
                <div class="info-item">
                    <strong>Status:</strong> <span class="status-ready">Ready for Analysis</span>
                </div>
            </div>
        `;
        
        audioInfo.style.display = 'block';
        audioInfo.classList.add('animate-fade-in');
    }

    // Recording Methods
    async startRecording() {
        try {
            // Starting recording
            
            // Check if getUserMedia is available
            if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                throw new Error('getUserMedia is not supported in this browser');
            }
            
            // Request microphone access
            const stream = await navigator.mediaDevices.getUserMedia({ 
                audio: {
                    echoCancellation: true,
                    noiseSuppression: true,
                    sampleRate: 16000,
                    channelCount: 1
                } 
            });
            
            // Microphone access granted
            
            // Create MediaRecorder
            // Try WAV format first, fallback to WebM
            let mimeType = 'audio/wav';
            if (!MediaRecorder.isTypeSupported('audio/wav')) {
                mimeType = 'audio/webm;codecs=opus';
            }
            
            this.mediaRecorder = new MediaRecorder(stream, {
                mimeType: mimeType
            });
            
            this.audioChunks = [];
            this.mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    this.audioChunks.push(event.data);
                }
            };
            
            this.mediaRecorder.onstop = () => {
                // Recording stopped, processing...
                this.processRecording();
                stream.getTracks().forEach(track => track.stop());
            };
            
            this.mediaRecorder.onerror = (event) => {
                console.error('MediaRecorder error:', event.error);
                this.showError('Recording error occurred. Please try again.');
                this.stopRecording();
            };
            
            // Start recording
            this.mediaRecorder.start(100); // Collect data every 100ms
            this.isRecording = true;
            this.recordingStartTime = Date.now();
            
            // Update UI
            this.updateRecordingUI(true);
            this.startRecordingTimer();
            this.startRecordingPrompts();
            
            // Initialize waveform if available
            if (this.waveform) {
                this.waveform.startVisualization(stream);
            }
            
            this.updateStatus('Recording in progress...', 'recording');
            // Recording started successfully
            
        } catch (error) {
            console.error('Error starting recording:', error);
            let errorMessage = 'Unable to access microphone. ';
            
            if (error.message === 'getUserMedia is not supported in this browser') {
                errorMessage += 'Your browser does not support microphone access. Please use Chrome, Firefox, or Safari.';
            } else if (error.name === 'NotAllowedError' && location.protocol !== 'https:' && location.hostname !== 'localhost') {
                errorMessage += 'Microphone access requires HTTPS. Please use localhost:5000 instead of your IP address, or enable HTTPS.';
            } else if (error.name === 'NotAllowedError') {
                errorMessage += 'Please allow microphone access and try again.';
            } else if (error.name === 'NotFoundError') {
                errorMessage += 'No microphone found. Please connect a microphone.';
            } else if (error.name === 'NotReadableError') {
                errorMessage += 'Microphone is being used by another application.';
            } else if (error.name === 'TypeError') {
                errorMessage += 'Browser compatibility issue. Please use a modern browser.';
            } else {
                errorMessage += 'Please check your microphone and try again.';
            }
            
            this.showError(errorMessage);
        }
    }

    stopRecording() {
        // Stopping recording
        
        if (this.mediaRecorder && this.isRecording) {
            this.mediaRecorder.stop();
            this.isRecording = false;
            this.updateRecordingUI(false);
            this.stopRecordingTimer();
            this.stopRecordingPrompts();
            
            if (this.waveform) {
                this.waveform.stopVisualization();
            }
            
            this.updateStatus('Recording completed', 'success');
            // Recording stopped successfully
        }
    }

    startRecordingPrompts() {
        const prompts = [
            "Please speak naturally about your day...",
            "Tell me about something that made you happy recently...",
            "Describe how you've been feeling lately...",
            "Talk about your sleep patterns...",
            "Share something that's been on your mind...",
            "Describe your energy levels...",
            "Talk about your social interactions...",
            "Share your thoughts on the future..."
        ];
        
        let promptIndex = 0;
        this.promptInterval = setInterval(() => {
            if (this.isRecording && promptIndex < prompts.length) {
                this.showRecordingPrompt(prompts[promptIndex]);
                promptIndex++;
            }
        }, 8000); // Show new prompt every 8 seconds
    }

    stopRecordingPrompts() {
        if (this.promptInterval) {
            clearInterval(this.promptInterval);
            this.promptInterval = null;
        }
        this.hideRecordingPrompt();
    }

    showRecordingPrompt(text) {
        let promptElement = document.getElementById('recordingPrompt');
        if (!promptElement) {
            promptElement = document.createElement('div');
            promptElement.id = 'recordingPrompt';
            promptElement.className = 'recording-prompt';
            document.querySelector('.recording-content').appendChild(promptElement);
        }
        
        promptElement.innerHTML = `
            <div class="prompt-content">
                <i class="fas fa-microphone"></i>
                <span>${text}</span>
            </div>
        `;
        promptElement.style.display = 'block';
    }

    hideRecordingPrompt() {
        const promptElement = document.getElementById('recordingPrompt');
        if (promptElement) {
            promptElement.style.display = 'none';
        }
    }

    updateRecordingUI(isRecording) {
        const recordButton = document.getElementById('recordButton');
        const stopButton = document.getElementById('stopButton');
        const recordingStatus = document.getElementById('recordingStatus');
        
        if (isRecording) {
            if (recordButton) recordButton.style.display = 'none';
            if (stopButton) stopButton.style.display = 'inline-flex';
            if (recordingStatus) recordingStatus.style.display = 'flex';
            
            // Add recording class to body for global styling
            document.body.classList.add('recording-active');
        } else {
            if (recordButton) recordButton.style.display = 'inline-flex';
            if (stopButton) stopButton.style.display = 'none';
            if (recordingStatus) recordingStatus.style.display = 'none';
            
            // Remove recording class
            document.body.classList.remove('recording-active');
        }
    }

    startRecordingTimer() {
        this.recordingTimer = setInterval(() => {
            const elapsed = Date.now() - this.recordingStartTime;
            const seconds = Math.floor(elapsed / 1000);
            const minutes = Math.floor(seconds / 60);
            const remainingSeconds = seconds % 60;
            
            const timerDisplay = document.getElementById('recordingTimer');
            if (timerDisplay) {
                timerDisplay.textContent = `${minutes.toString().padStart(2, '0')}:${remainingSeconds.toString().padStart(2, '0')}`;
            }
            
            // Auto-stop at 60 seconds
            if (seconds >= 60) {
                this.stopRecording();
                this.showMessage('Recording completed automatically after 60 seconds.', 'success');
            }
        }, 1000);
    }

    stopRecordingTimer() {
        if (this.recordingTimer) {
            clearInterval(this.recordingTimer);
            this.recordingTimer = null;
        }
    }

    processRecording() {
        if (this.audioChunks.length === 0) {
            this.showError('No audio data recorded. Please try again.');
            return;
        }
        
        const audioBlob = new Blob(this.audioChunks, { type: 'audio/webm' });
        const duration = this.getRecordingDuration();
        
        // Check minimum duration (at least 5 seconds)
        if (duration < 5) {
            this.showError('Recording too short. Please record at least 5 seconds of audio.');
            return;
        }
        
        this.audioFile = new File([audioBlob], `recording_${Date.now()}.webm`, { 
            type: 'audio/webm',
            lastModified: Date.now()
        });
        
        // Recording processed successfully
        
        this.displayAudioInfo(this.audioFile);
        this.showAnalysisButton();
        this.addRestartButton();
        
        // Show success message for short recordings
        if (duration < 30) {
            this.showMessage(`Recording completed (${duration.toFixed(1)}s). Analysis will work with shorter recordings.`, 'success');
        }
    }

    addRestartButton() {
        const recordingContent = document.querySelector('.recording-content');
        if (!recordingContent) return;
        
        // Remove existing restart button
        const existingRestart = document.getElementById('restartRecording');
        if (existingRestart) {
            existingRestart.remove();
        }
        
        // Add restart button
        const restartButton = document.createElement('button');
        restartButton.id = 'restartRecording';
        restartButton.className = 'medical-btn secondary restart-btn';
        restartButton.innerHTML = '<i class="fas fa-redo"></i> Record Again';
        restartButton.onclick = () => this.restartRecording();
        
        recordingContent.appendChild(restartButton);
    }

    restartRecording() {
        // Reset recording state
        this.audioChunks = [];
        this.audioFile = null;
        this.isRecording = false;
        
        // Hide analysis section
        const analysisSection = document.getElementById('analysisSection');
        if (analysisSection) {
            analysisSection.style.display = 'none';
        }
        
        // Hide audio info
        const audioInfo = document.getElementById('audioInfo');
        if (audioInfo) {
            audioInfo.style.display = 'none';
        }
        
        // Remove restart button
        const restartButton = document.getElementById('restartRecording');
        if (restartButton) {
            restartButton.remove();
        }
        
        // Reset UI
        this.updateRecordingUI(false);
        this.updateStatus('Ready to record', 'success');
        
        // Recording restarted
    }

    getRecordingDuration() {
        if (!this.recordingStartTime) return 0;
        return (Date.now() - this.recordingStartTime) / 1000;
    }

    showAnalysisButton() {
        const analysisSection = document.getElementById('analysisSection');
        analysisSection.style.display = 'block';
        analysisSection.classList.add('animate-fade-in');
        
        // Scroll to analysis section
        analysisSection.scrollIntoView({ behavior: 'smooth' });
        
        // Auto-start analysis after a short delay
        setTimeout(() => {
            this.analyzeAudio();
        }, 1000);
    }

    // Analysis Methods
    async analyzeAudio() {
        if (!this.audioFile) {
            this.showError('No audio file available for analysis');
            return;
        }

        this.showLoadingState();
        this.updateStatus('Analyzing voice patterns...', 'analyzing');

        try {
            const formData = new FormData();
            formData.append('audio', this.audioFile);

            const response = await fetch('/analyze', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const results = await response.json();
            this.analysisResults = results;
            
            this.hideLoadingState();
            this.displayResults(results);
            this.updateStatus('Analysis completed successfully', 'success');
            
        } catch (error) {
            console.error('Analysis error:', error);
            this.hideLoadingState();
            this.showError('Analysis failed. Please try again.');
            this.updateStatus('Analysis failed', 'error');
        }
    }

    showLoadingState() {
        const loadingSpinner = document.getElementById('loadingSpinner');
        const resultsDisplay = document.getElementById('resultsDisplay');
        
        loadingSpinner.style.display = 'block';
        resultsDisplay.style.display = 'none';
        
        // Animate progress bar
        this.animateProgressBar();
    }

    hideLoadingState() {
        const loadingSpinner = document.getElementById('loadingSpinner');
        loadingSpinner.style.display = 'none';
    }

    animateProgressBar() {
        const progressFill = document.getElementById('progressFill');
        let progress = 0;
        
        const interval = setInterval(() => {
            progress += Math.random() * 15;
            if (progress > 100) {
                progress = 100;
                clearInterval(interval);
            }
            progressFill.style.width = progress + '%';
        }, 200);
    }

    displayResults(results) {
        // Displaying analysis results
        const resultsDisplay = document.getElementById('resultsDisplay');
        const disorderResults = document.getElementById('disorderResults');
        const clinicalReport = document.getElementById('clinicalReport');
        const reportContent = document.getElementById('reportContent');
        
        // Display disorder results
        this.displayDisorderResults(results.results, disorderResults);
        
        // Display clinical report
        if (results.clinical_report) {
            reportContent.innerHTML = this.formatClinicalReport(results.clinical_report);
            clinicalReport.style.display = 'block';
        }
        
        // Update confidence indicator
        this.updateConfidenceIndicator(results);
        
        resultsDisplay.style.display = 'block';
        resultsDisplay.classList.add('animate-fade-in');
    }

    displayDisorderResults(results, container) {
        // Displaying disorder-specific results
        const disorders = ['depression', 'anxiety', 'ptsd', 'cognitive_decline'];
        const disorderNames = {
            'depression': 'Depression',
            'anxiety': 'Anxiety',
            'ptsd': 'PTSD',
            'cognitive_decline': 'Cognitive Decline'
        };
        
        const disorderIcons = {
            'depression': 'fas fa-heart-broken',
            'anxiety': 'fas fa-exclamation-triangle',
            'ptsd': 'fas fa-shield-alt',
            'cognitive_decline': 'fas fa-brain'
        };

        container.innerHTML = '';

        disorders.forEach(disorder => {
            if (results[disorder]) {
                const score = results[disorder].probability || results[disorder];
                const riskLevel = this.getRiskLevel(score);
                const riskColor = this.getRiskColor(riskLevel);
                
                const card = document.createElement('div');
                card.className = 'disorder-card animate-bounce-in';
                card.style.animationDelay = `${disorders.indexOf(disorder) * 0.1}s`;
                
                card.innerHTML = `
                    <div class="disorder-header">
                        <div class="disorder-icon ${disorder}">
                            <i class="${disorderIcons[disorder]}"></i>
                        </div>
                        <div class="disorder-name">${disorderNames[disorder]}</div>
                    </div>
                    <div class="risk-score">
                        <div class="score-label">
                            <span>Risk Score</span>
                            <span>${(score * 100).toFixed(1)}%</span>
                        </div>
                        <div class="score-bar">
                            <div class="score-fill ${riskLevel}" style="width: ${score * 100}%"></div>
                        </div>
                        <div class="confidence-info">
                            <span>Risk Level: ${riskLevel.toUpperCase()}</span>
                            <span>Confidence: High</span>
                        </div>
                    </div>
                `;
                
                container.appendChild(card);
            }
        });
    }

    getRiskLevel(score) {
        if (score < 0.3) return 'low';
        if (score < 0.7) return 'moderate';
        return 'high';
    }

    getRiskColor(riskLevel) {
        const colors = {
            'low': '#10b981',
            'moderate': '#f59e0b',
            'high': '#ef4444'
        };
        return colors[riskLevel];
    }

    formatClinicalReport(report) {
        if (typeof report === 'string') {
            return `<p>${report}</p>`;
        }
        
        let html = '';
        if (report.summary) {
            html += `<h5>Summary</h5><p>${report.summary}</p>`;
        }
        if (report.recommendations) {
            html += `<h5>Recommendations</h5><ul>`;
            report.recommendations.forEach(rec => {
                html += `<li>${rec}</li>`;
            });
            html += `</ul>`;
        }
        if (report.notes) {
            html += `<h5>Clinical Notes</h5><p>${report.notes}</p>`;
        }
        
        return html || '<p>No detailed report available.</p>';
    }

    updateConfidenceIndicator(results) {
        const confidenceValue = document.getElementById('overallConfidence');
        const confidenceFill = document.getElementById('confidenceFill');
        
        // Calculate overall confidence (average of all confidence scores)
        const disorderResults = results.results || {};
        const confidenceScores = Object.values(disorderResults).map(disorder => 
            disorder.confidence || 0.5
        );
        
        const avgConfidence = confidenceScores.length > 0 ? 
            confidenceScores.reduce((a, b) => a + b, 0) / confidenceScores.length : 0.5;
        
        const confidencePercent = (avgConfidence * 100).toFixed(1);
        confidenceValue.textContent = `${confidencePercent}%`;
        confidenceFill.style.width = `${confidencePercent}%`;
    }

    // Utility Methods
    updateStatus(message, type = 'info') {
        const statusIndicator = document.querySelector('.status-indicator');
        if (!statusIndicator) {
            // Status indicator not found, skipping status update
            return;
        }
        
        const statusDot = statusIndicator.querySelector('.status-dot');
        const statusText = statusIndicator.querySelector('span');
        
        if (statusText) {
            statusText.textContent = message;
        }
        
        // Update status dot color
        if (statusDot) {
            statusDot.className = 'status-dot';
            if (type === 'success') statusDot.style.background = '#10b981';
            else if (type === 'error') statusDot.style.background = '#ef4444';
            else if (type === 'recording') statusDot.style.background = '#f59e0b';
            else if (type === 'analyzing') statusDot.style.background = '#6366f1';
            else statusDot.style.background = '#6b7280';
        }
    }

    showError(message) {
        this.showMessage(message, 'error');
    }

    showMessage(message, type = 'info') {
        // Create a modern notification
        const notificationDiv = document.createElement('div');
        notificationDiv.className = 'notification';
        
        let icon, bgColor;
        switch (type) {
            case 'error':
                icon = 'fas fa-exclamation-circle';
                bgColor = '#ef4444';
                break;
            case 'success':
                icon = 'fas fa-check-circle';
                bgColor = '#10b981';
                break;
            case 'warning':
                icon = 'fas fa-exclamation-triangle';
                bgColor = '#f59e0b';
                break;
            default:
                icon = 'fas fa-info-circle';
                bgColor = '#3b82f6';
        }
        
        notificationDiv.innerHTML = `
            <div class="notification-content">
                <i class="${icon}"></i>
                <span>${message}</span>
                <button class="notification-close" onclick="this.parentElement.parentElement.remove()">
                    <i class="fas fa-times"></i>
                </button>
            </div>
        `;
        
        // Add notification styles
        notificationDiv.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: ${bgColor};
            color: white;
            padding: 15px 20px;
            border-radius: 8px;
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
            z-index: 1000;
            animation: slideInRight 0.3s ease-out;
            max-width: 400px;
        `;
        
        document.body.appendChild(notificationDiv);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (notificationDiv.parentElement) {
                notificationDiv.remove();
            }
        }, 5000);
    }

    downloadReport() {
        if (!this.analysisResults) {
            this.showError('No analysis results available to download');
            return;
        }

        const reportData = {
            timestamp: new Date().toISOString(),
            results: this.analysisResults.results,
            clinical_report: this.analysisResults.clinical_report,
            audio_file: this.audioFile ? this.audioFile.name : 'recording.webm'
        };

        const blob = new Blob([JSON.stringify(reportData, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        
        const a = document.createElement('a');
        a.href = url;
        a.download = `voicemind-report-${new Date().toISOString().split('T')[0]}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        
        this.updateStatus('Report downloaded successfully', 'success');
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.voiceMindApp = new VoiceMindApp();
});

// Add CSS for error notifications
const style = document.createElement('style');
style.textContent = `
    @keyframes slideInRight {
        from {
            transform: translateX(100%);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    .error-notification {
        font-family: 'Inter', sans-serif;
    }
    
    .error-content {
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    .error-close {
        background: none;
        border: none;
        color: white;
        cursor: pointer;
        padding: 0;
        margin-left: 10px;
    }
    
    .info-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 15px;
    }
    
    .info-item {
        padding: 10px;
        background: var(--bg-primary);
        border-radius: var(--radius-sm);
        border: 1px solid var(--border-color);
    }
    
    .status-ready {
        color: var(--success-color);
        font-weight: 600;
    }
`;
document.head.appendChild(style);