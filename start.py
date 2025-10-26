#!/usr/bin/env python3
"""
VoiceMind AI - Clinical Mental Health Screening Platform
Startup script for the web application.
"""

import sys
import os
import subprocess
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed."""
    try:
        import flask
        import numpy
        import pandas
        import librosa
        import soundfile
        print("✅ All required packages are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please run: pip install -r requirements.txt")
        return False

def check_ffmpeg():
    """Check if FFmpeg is available."""
    ffmpeg_path = Path("ffmpeg-master-latest-win64-gpl/bin/ffmpeg.exe")
    if ffmpeg_path.exists():
        print("✅ FFmpeg is available")
        return True
    else:
        print("⚠️  FFmpeg not found. Audio processing may be limited.")
        print("Run: .\\install_ffmpeg.ps1")
        return False

def main():
    """Main startup function."""
    print("🎤 VoiceMind AI - Clinical Mental Health Screening Platform")
    print("=" * 60)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check FFmpeg
    check_ffmpeg()
    
    print("\n🚀 Starting web application...")
    print("📱 Open your browser to: http://localhost:5000")
    print("⏹️  Press Ctrl+C to stop the server")
    print("=" * 60)
    
    # Start the Flask application
    try:
        from app.api import app
        app.run(debug=True, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\n👋 Server stopped. Goodbye!")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
