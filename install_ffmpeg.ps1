# PowerShell script to install FFmpeg
Write-Host "Installing FFmpeg for WebM audio processing..." -ForegroundColor Green

# Create ffmpeg directory
$ffmpegDir = "C:\ffmpeg"
if (!(Test-Path $ffmpegDir)) {
    New-Item -ItemType Directory -Path $ffmpegDir -Force
    Write-Host "Created directory: $ffmpegDir" -ForegroundColor Yellow
}

# Download FFmpeg
$downloadUrl = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip"
$zipFile = "$env:TEMP\ffmpeg.zip"

Write-Host "Downloading FFmpeg..." -ForegroundColor Yellow
try {
    Invoke-WebRequest -Uri $downloadUrl -OutFile $zipFile -UseBasicParsing
    Write-Host "Download completed!" -ForegroundColor Green
} catch {
    Write-Host "Download failed: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "Please download manually from: https://ffmpeg.org/download.html" -ForegroundColor Yellow
    exit 1
}

# Extract FFmpeg
Write-Host "Extracting FFmpeg..." -ForegroundColor Yellow
try {
    Expand-Archive -Path $zipFile -DestinationPath $ffmpegDir -Force
    Write-Host "Extraction completed!" -ForegroundColor Green
} catch {
    Write-Host "Extraction failed: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}

# Find the extracted folder
$extractedFolder = Get-ChildItem -Path $ffmpegDir -Directory | Where-Object { $_.Name -like "ffmpeg-*" } | Select-Object -First 1

if ($extractedFolder) {
    # Move contents to ffmpeg directory
    $sourcePath = $extractedFolder.FullName
    $binPath = "$ffmpegDir\bin"
    
    if (Test-Path "$sourcePath\bin") {
        Copy-Item -Path "$sourcePath\bin\*" -Destination $binPath -Force
        Write-Host "FFmpeg binaries copied to: $binPath" -ForegroundColor Green
    }
    
    # Remove the extracted folder
    Remove-Item -Path $sourcePath -Recurse -Force
}

# Add to PATH
$binPath = "$ffmpegDir\bin"
$currentPath = [Environment]::GetEnvironmentVariable("PATH", "User")

if ($currentPath -notlike "*$binPath*") {
    [Environment]::SetEnvironmentVariable("PATH", "$currentPath;$binPath", "User")
    Write-Host "Added FFmpeg to PATH" -ForegroundColor Green
} else {
    Write-Host "FFmpeg already in PATH" -ForegroundColor Yellow
}

# Clean up
Remove-Item -Path $zipFile -Force -ErrorAction SilentlyContinue

Write-Host "FFmpeg installation completed!" -ForegroundColor Green
Write-Host "Please restart your terminal and browser for changes to take effect." -ForegroundColor Yellow
Write-Host "Then try recording again!" -ForegroundColor Cyan
