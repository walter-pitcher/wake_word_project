#!/usr/bin/env python3
"""
Download Hey Magis recordings from Typeform using the API
Since Typeform doesn't export direct audio URLs, we need to use their API
"""

import pandas as pd
import requests
import os
from pathlib import Path
import time
import json

class TypeformAudioDownloader:
    def __init__(self, form_id="", access_token=""):
        """
        Initialize the Typeform downloader
        
        Args:
            form_id: Your Typeform form ID (from the URL)
            access_token: Your Typeform Personal Access Token
        """
        self.form_id = form_id
        self.access_token = access_token
        self.base_url = "https://api.typeform.com"
        
    def get_api_credentials(self):
        """Get API credentials from user if not provided"""
        print("\n📋 Typeform API Setup")
        print("=" * 60)
        print("To download audio files, we need your Typeform API credentials.")
        print("\n1. Get your Form ID:")
        print("   - Go to your Typeform")
        print("   - Look at the URL: typeform.com/to/YOUR_FORM_ID")
        print("   - Copy the YOUR_FORM_ID part")
        
        if not self.form_id:
            self.form_id = input("\nEnter your Form ID: ").strip()
        
        print("\n2. Get your Personal Access Token:")
        print("   - Go to: https://admin.typeform.com/account/personal-access-tokens")
        print("   - Click 'Generate a new token'")
        print("   - Copy the token")
        
        if not self.access_token:
            self.access_token = input("\nEnter your Personal Access Token: ").strip()
    
    def download_with_api(self, output_dir="data/positive"):
        """Download all audio responses using Typeform API"""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        headers = {
            "Authorization": f"Bearer {self.access_token}"
        }
        
        # Get all responses
        print("\n📥 Fetching responses from Typeform API...")
        
        # API endpoint for responses
        url = f"{self.base_url}/forms/{self.form_id}/responses"
        
        # Parameters for the API call
        params = {
            "page_size": 1000,  # Max allowed
            "completed": "true"
        }
        
        try:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()
            
            items = data.get('items', [])
            print(f"✅ Found {len(items)} responses")
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 403:
                print("❌ Authentication failed. Please check your access token.")
            elif e.response.status_code == 404:
                print("❌ Form not found. Please check your form ID.")
            else:
                print(f"❌ API Error: {e}")
            return
        except Exception as e:
            print(f"❌ Error fetching responses: {e}")
            return
        
        # Download audio files
        downloaded = 0
        failed = 0
        skipped = 0
        
        print("\n📥 Downloading audio files...")
        print("=" * 60)
        
        for idx, item in enumerate(items, 1):
            response_id = item.get('response_id', f'response_{idx}')
            token = item.get('token', response_id)
            
            # Debug: Show what we're looking at for first few responses
            if idx <= 3:  # Show first 3 responses in detail
                print(f"\n🔍 Response {idx} structure:")
                print(f"   Response ID: {response_id}")
                answers = item.get('answers', [])
                print(f"   Number of answers: {len(answers)}")
                
                for ans_idx, answer in enumerate(answers):
                    field_info = answer.get('field', {})
                    print(f"\n   Answer {ans_idx + 1}:")
                    print(f"     - Type: {answer.get('type')}")
                    print(f"     - Field ID: {field_info.get('id')}")
                    print(f"     - Field Type: {field_info.get('type')}")
                    print(f"     - Field Ref: {field_info.get('ref')}")
                    
                    # Check different possible audio field types
                    if answer.get('type') == 'file_url':
                        print(f"     - FILE URL: {answer.get('file_url', 'None')[:50]}...")
                    elif answer.get('type') == 'url':
                        print(f"     - URL: {answer.get('url', 'None')[:50]}...")
                    elif field_info.get('type') in ['file_upload', 'audio', 'voice']:
                        print(f"     - Possible audio field!")
                        print(f"     - Content: {answer}")
                
                print("-" * 40)
            
            # Look for audio answers - check multiple possible formats
            answers = item.get('answers', [])
            found_audio = False
            
            for answer in answers:
                file_url = None
                
                # Handle multi_format type (which contains audio recordings)
                if answer.get('type') == 'multi_format':
                    # The audio URL is in multi_format.audio_url!
                    multi_format_data = answer.get('multi_format', {})
                    file_url = multi_format_data.get('audio_url')
                    
                    # Also capture transcript if available (for debugging)
                    transcript = multi_format_data.get('audio_transcript', '')
                    if transcript and idx <= 5:
                        print(f"   Transcript: '{transcript}'")
                
                # Check other format types (just in case)
                elif answer.get('type') == 'file_url':
                    file_url = answer.get('file_url')
                elif answer.get('type') == 'url':
                    file_url = answer.get('url')
                
                # Also check if field type indicates audio
                field_type = answer.get('field', {}).get('type', '')
                if field_type in ['file_upload', 'audio', 'voice'] and not file_url:
                    # Might need to construct URL differently
                    field_id = answer.get('field', {}).get('id')
                    if field_id:
                        print(f"   Found audio field {field_id} but no direct URL")
                
                if file_url:
                    # Determine file extension from URL
                    if '.mp4' in file_url:
                        ext = '.mp4'
                    elif '.wav' in file_url:
                        ext = '.wav'
                    elif '.webm' in file_url:
                        ext = '.webm'
                    elif '.m4a' in file_url:
                        ext = '.m4a'
                    else:
                        ext = '.mp4'  # Default for audio
                    
                    filename = f"hey_magis_{response_id[:10]}{ext}"
                    filepath = output_path / filename
                    
                    print(f"\nDownloading {downloaded + 1}: {filename}")
                    print(f"  URL: {file_url[:80]}...")
                    
                    try:
                        # Download with auth header
                        headers = {"Authorization": f"Bearer {self.access_token}"}
                        file_response = requests.get(file_url, headers=headers, timeout=30)
                        file_response.raise_for_status()
                        
                        # Save the file
                        with open(filepath, 'wb') as f:
                            f.write(file_response.content)
                        
                        downloaded += 1
                        found_audio = True
                        print(f"  ✓ Saved: {filename}")
                        
                        # Small delay
                        time.sleep(0.5)
                        
                    except Exception as e:
                        print(f"  ✗ Failed: {e}")
                        failed += 1
                    
                    break  # Found audio for this response
            
            if not found_audio:
                skipped += 1
                if skipped <= 5:  # Show first 5 skipped
                    print(f"⏭️  Skipped response {idx} - no audio file found")
        
        print("\n" + "=" * 60)
        print("DOWNLOAD COMPLETE")
        print("=" * 60)
        print(f"✅ Successfully downloaded: {downloaded}")
        print(f"❌ Failed: {failed}")
        print(f"\n📁 Files saved to: {output_path.absolute()}")
        
        if downloaded > 0:
            print("\n⚠️  Note: Files are in MP4 format. You may need to convert to WAV.")
            print("Use this command to convert all MP4 to WAV:")
            print('for f in *.mp4; do ffmpeg -i "$f" -ar 16000 -ac 1 "${f%.mp4}.wav"; done')
    
    def download_manual_method(self, excel_file, output_dir="data/positive"):
        """
        Alternative: Guide user through manual download since audio URLs aren't in the export
        """
        print("\n📋 Manual Download Method")
        print("=" * 60)
        print("Since your Excel export doesn't contain direct audio URLs,")
        print("you have two options:\n")
        
        print("Option 1: Use Typeform API (Recommended)")
        print("  - Requires a Personal Access Token")
        print("  - Can download all files automatically\n")
        
        print("Option 2: Manual Download")
        print("  - Go to Typeform Results > Responses")
        print("  - Click each response")
        print("  - Download audio individually")
        print("  - Time-consuming for 273 recordings!\n")
        
        choice = input("Would you like to use the API method? (yes/no): ").strip().lower()
        
        if choice == 'yes':
            self.get_api_credentials()
            self.download_with_api(output_dir)
        else:
            print("\n📝 Manual Download Instructions:")
            print("1. Go to your Typeform Results")
            print("2. Click on each response")
            print("3. Click 'Download audio'")
            print("4. Save all files to:", Path(output_dir).absolute())
            print("\nThis will take a while for 273 recordings...")
            print("The API method is much faster!")


def convert_mp4_to_wav(input_dir="data/positive"):
    """Convert MP4 files from Typeform to WAV format"""
    import subprocess
    
    input_path = Path(input_dir)
    mp4_files = list(input_path.glob("*.mp4"))
    
    if not mp4_files:
        print("No MP4 files to convert")
        return
    
    print(f"\n🔄 Converting {len(mp4_files)} MP4 files to WAV...")
    
    converted = 0
    failed = 0
    
    for mp4_file in mp4_files:
        wav_file = mp4_file.with_suffix('.wav')
        
        try:
            cmd = [
                'ffmpeg', '-i', str(mp4_file),
                '-ar', '16000',  # 16kHz sample rate
                '-ac', '1',       # Mono
                '-y',             # Overwrite
                str(wav_file)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                converted += 1
                print(f"✓ Converted: {mp4_file.name}")
                mp4_file.unlink()  # Delete MP4 after conversion
            else:
                failed += 1
                print(f"✗ Failed: {mp4_file.name}")
        except FileNotFoundError:
            print("\n❌ ffmpeg not found! Please install it:")
            print("Download from: https://ffmpeg.org/download.html")
            break
        except Exception as e:
            failed += 1
            print(f"✗ Error: {e}")
    
    if converted > 0:
        print(f"\n✅ Converted {converted} files to WAV format")


if __name__ == "__main__":
    print("=" * 60)
    print("🎤 TYPEFORM AUDIO DOWNLOADER")
    print("=" * 60)
    print("\nThis script will download your 273 Hey Magis recordings")
    
    downloader = TypeformAudioDownloader()
    
    print("\nChoose method:")
    print("1. API Method (Automatic - Recommended)")
    print("2. I already downloaded them manually")
    print("3. Show me manual instructions")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == "1":
        downloader.get_api_credentials()
        downloader.download_with_api()
        
        # Offer to convert MP4 to WAV
        convert = input("\nConvert MP4 files to WAV? (yes/no): ").strip().lower()
        if convert == 'yes':
            convert_mp4_to_wav()
            
    elif choice == "2":
        print("\nGreat! Make sure all audio files are in: data/positive/")
        convert = input("Do you need to convert MP4 to WAV? (yes/no): ").strip().lower()
        if convert == 'yes':
            convert_mp4_to_wav()
            
    elif choice == "3":
        downloader.download_manual_method(None)
    
    print("\n✅ Done! Next step: Run trim_recordings.py to remove silence")
