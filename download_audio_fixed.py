#!/usr/bin/env python3
"""
Fixed script to download actual audio from Typeform's Mux streaming URLs
"""

import json
import requests
import os
from pathlib import Path
import subprocess

def download_actual_audio():
    """Convert the JSON files to actual audio"""
    
    input_dir = Path("data/positive")
    
    # Get all the fake MP4 files (which are actually JSON)
    json_files = list(input_dir.glob("*.mp4"))
    print(f"Found {len(json_files)} files to process")
    
    downloaded = 0
    failed = 0
    
    for json_file in json_files:
        try:
            # Read the JSON content
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Get the playback URL
            playback_url = data.get('playback_url')
            
            if playback_url:
                # The URL is an m3u8 stream, we need to download it with ffmpeg
                output_file = json_file.with_suffix('.wav')
                
                print(f"Downloading: {json_file.stem}")
                
                # Use ffmpeg to download and convert in one step
                cmd = [
                    'ffmpeg',
                    '-i', playback_url,
                    '-ar', '16000',  # 16kHz sample rate
                    '-ac', '1',      # Mono
                    '-y',            # Overwrite
                    str(output_file),
                    '-loglevel', 'error'
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    downloaded += 1
                    print(f"  ✓ Converted to: {output_file.name}")
                    
                    # Delete the JSON file
                    json_file.unlink()
                else:
                    failed += 1
                    print(f"  ✗ Failed: {result.stderr}")
            
        except Exception as e:
            print(f"  ✗ Error processing {json_file.name}: {e}")
            failed += 1
    
    print(f"\n{'='*50}")
    print(f"✅ Successfully downloaded: {downloaded}")
    print(f"❌ Failed: {failed}")
    
    # Clean up any remaining .mp4 files
    remaining = list(input_dir.glob("*.mp4"))
    if remaining:
        print(f"\nCleaning up {len(remaining)} JSON files...")
        for f in remaining:
            f.unlink()

if __name__ == "__main__":
    print("🎵 TYPEFORM AUDIO DOWNLOADER - FIXED VERSION")
    print("="*50)
    print("This will download the actual audio from the streaming URLs\n")
    
    download_actual_audio()
    
    print("\n✅ Done! Check your data/positive folder for WAV files")