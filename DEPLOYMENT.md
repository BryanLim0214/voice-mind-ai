# Deployment Guide

## Quick Start

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Install FFmpeg**
   - Windows: Run `install_ffmpeg.ps1` as Administrator
   - Or download from https://ffmpeg.org/download.html

3. **Start the Application**
   ```bash
   python start.py
   ```

4. **Access the Application**
   - Open your browser to `http://localhost:5000`
   - Allow microphone access when prompted
   - Record or upload audio for analysis

## Production Deployment

### Using Gunicorn (Recommended)
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app.api:app
```

### Using Docker
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 5000

CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app.api:app"]
```

## Environment Variables

- `FLASK_ENV`: Set to `production` for production deployment
- `MAX_CONTENT_LENGTH`: Maximum file upload size (default: 16MB)

## Security Notes

- This is a research tool, not for clinical use
- No real patient data should be processed
- Ensure proper firewall configuration for production
- Consider HTTPS for production deployments

## Troubleshooting

### Common Issues

1. **FFmpeg not found**: Ensure FFmpeg is installed and in PATH
2. **Microphone access denied**: Check browser permissions
3. **File upload fails**: Check file size limits and supported formats
4. **Models not loading**: Ensure model files are present in `models/` directory

### Supported Audio Formats
- WAV, MP3, FLAC, OGG, WEBM
- Maximum file size: 16MB
- Recommended: 16kHz sample rate, mono channel

## Performance

- Typical analysis time: 2-5 seconds
- Memory usage: ~500MB with all models loaded
- Concurrent users: Tested up to 10 simultaneous analyses
