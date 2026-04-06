#!/usr/bin/env python3
"""
Trim silence from the beginning of Hey Magis recordings
Ensures the wake word is properly positioned in the audio
"""

import os
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
import shutil

def detect_speech_onset(audio, sr=16000, threshold_db=-30):
    """
    Detect where speech actually starts in the audio
    """
    # Convert to dB
    audio_db = librosa.amplitude_to_db(np.abs(audio), ref=np.max)
    
    # Find where audio exceeds threshold
    speech_indices = np.where(audio_db > threshold_db)[0]
    
    if len(speech_indices) > 0:
        # Back up a tiny bit (50ms) to catch the very start
        start_idx = max(0, speech_indices[0] - int(0.05 * sr))
        return start_idx
    
    return 0

def is_silent_recording(audio, threshold_db=-40):
    """
    Check if a recording is essentially silent (no speech)
    """
    # Calculate RMS energy
    rms = np.sqrt(np.mean(audio**2))
    
    # Convert to dB
    if rms > 0:
        rms_db = 20 * np.log10(rms)
    else:
        return True  # Completely silent
    
    # Check if below threshold
    if rms_db < threshold_db:
        return True
    
    # Also check if max amplitude is very low
    if np.max(np.abs(audio)) < 0.01:
        return True
    
    # Check if there's enough variation (not just noise)
    if np.std(audio) < 0.001:
        return True
    
    return False

def trim_audio(input_path, output_path, target_duration=2.0, padding_before=0.1):
    """
    Trim silence from beginning and create consistent 2-second clips
    
    Args:
        input_path: Path to input WAV file
        output_path: Path to save trimmed file
        target_duration: Target duration in seconds (2.0 for wake words)
        padding_before: Seconds of padding to keep before speech starts
    
    Returns:
        (success, delay, is_empty) - success flag, silence delay, and whether file is empty
    """
    
    try:
        # Load audio
        audio, sr = librosa.load(input_path, sr=16000)
        
        # Check if recording is essentially silent/empty
        if is_silent_recording(audio):
            return False, 0, True  # Mark as empty recording
        
        # Find where speech starts
        speech_start = detect_speech_onset(audio, sr)
        
        # If no speech detected at all (but not completely silent)
        if speech_start == 0 and np.max(np.abs(audio)) < 0.05:
            return False, 0, True  # Likely an empty recording
        
        # Add small padding before speech
        start_with_padding = max(0, speech_start - int(padding_before * sr))
        
        # Trim from the start
        trimmed_audio = audio[start_with_padding:]
        
        # Ensure target duration
        target_samples = int(target_duration * sr)
        
        if len(trimmed_audio) > target_samples:
            # If too long, take first 2 seconds
            final_audio = trimmed_audio[:target_samples]
        else:
            # If too short, pad with silence at the end
            padding_needed = target_samples - len(trimmed_audio)
            final_audio = np.pad(trimmed_audio, (0, padding_needed), mode='constant')
        
        # Normalize audio (but check for division by zero)
        max_val = np.max(np.abs(final_audio))
        if max_val > 0:
            final_audio = final_audio / max_val
        
        # Save
        sf.write(output_path, final_audio, sr)
        
        return True, speech_start / sr, False  # Return success, delay, not empty
        
    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        return False, 0, False

def process_all_recordings(input_dir, output_dir, backup_dir=None, remove_silent_dir=None):
    """
    Process all recordings in a directory
    """
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Optional: Create backup of originals
    if backup_dir:
        backup_path = Path(backup_dir)
        backup_path.mkdir(parents=True, exist_ok=True)
    
    # Optional: Move silent/bad recordings
    if remove_silent_dir:
        silent_path = Path(remove_silent_dir)
        silent_path.mkdir(parents=True, exist_ok=True)
    
    wav_files = list(input_path.glob("*.wav"))
    
    print(f"Found {len(wav_files)} WAV files to process")
    print("=" * 50)
    
    stats = {
        'processed': 0,
        'had_silence': 0,
        'empty': 0,
        'failed': 0,
        'max_delay': 0,
        'total_delay': 0
    }
    
    empty_files = []
    
    for i, wav_file in enumerate(wav_files, 1):
        print(f"Processing {i}/{len(wav_files)}: {wav_file.name}")
        
        # Backup original if requested
        if backup_dir:
            shutil.copy2(wav_file, backup_path / wav_file.name)
        
        # Process the file
        output_file = output_path / wav_file.name
        success, delay, is_empty = trim_audio(wav_file, output_file)
        
        if is_empty:
            stats['empty'] += 1
            empty_files.append(wav_file.name)
            print(f"  ⚠️  EMPTY/SILENT RECORDING - Skipped")
            
            # Move to silent folder if specified
            if remove_silent_dir:
                shutil.move(str(wav_file), silent_path / wav_file.name)
                print(f"     Moved to {remove_silent_dir}/")
            
            # Remove the output file if it was created
            if output_file.exists():
                output_file.unlink()
                
        elif success:
            stats['processed'] += 1
            stats['total_delay'] += delay
            
            if delay > 0.1:  # More than 100ms of silence
                stats['had_silence'] += 1
                print(f"  ✓ Trimmed {delay:.2f}s of silence")
            else:
                print(f"  ✓ No significant silence")
            
            if delay > stats['max_delay']:
                stats['max_delay'] = delay
        else:
            stats['failed'] += 1
            print(f"  ✗ Failed to process")
    
    # Print summary
    print("\n" + "=" * 50)
    print("PROCESSING COMPLETE")
    print("=" * 50)
    print(f"✓ Successfully processed: {stats['processed']}/{len(wav_files)}")
    print(f"✓ Files with silence trimmed: {stats['had_silence']}")
    
    if stats['empty'] > 0:
        print(f"⚠️  Empty/silent recordings found: {stats['empty']}")
        print(f"   These files had no speech:")
        for ef in empty_files[:10]:  # Show first 10
            print(f"   - {ef}")
        if len(empty_files) > 10:
            print(f"   ... and {len(empty_files) - 10} more")
    
    if stats['processed'] > 0:
        avg_delay = stats['total_delay'] / stats['processed']
        print(f"✓ Average silence removed: {avg_delay:.2f}s")
        print(f"✓ Maximum silence removed: {stats['max_delay']:.2f}s")
    
    if stats['failed'] > 0:
        print(f"✗ Failed to process: {stats['failed']} files")
    
    print(f"\n📁 Trimmed files saved to: {output_path}")
    print(f"📊 USABLE RECORDINGS: {stats['processed']} files")
    
    if backup_dir:
        print(f"📁 Original files backed up to: {backup_path}")
    if remove_silent_dir and stats['empty'] > 0:
        print(f"📁 Silent recordings moved to: {silent_path}")

def analyze_recordings(input_dir):
    """
    Analyze recordings to see how much silence they have
    """
    input_path = Path(input_dir)
    wav_files = list(input_path.glob("*.wav"))
    
    print(f"Analyzing {len(wav_files)} recordings for silence...")
    print("=" * 50)
    
    silence_stats = []
    
    for wav_file in wav_files:
        try:
            audio, sr = librosa.load(wav_file, sr=16000)
            speech_start = detect_speech_onset(audio, sr)
            delay = speech_start / sr
            
            silence_stats.append({
                'file': wav_file.name,
                'delay': delay,
                'duration': len(audio) / sr
            })
        except:
            pass
    
    # Sort by delay
    silence_stats.sort(key=lambda x: x['delay'], reverse=True)
    
    # Show worst offenders
    print("\nFiles with most silence at beginning:")
    for stat in silence_stats[:10]:
        bar = '█' * int(stat['delay'] * 20)
        print(f"{stat['file'][:30]:30s} {bar} {stat['delay']:.2f}s")
    
    # Summary
    delays = [s['delay'] for s in silence_stats]
    print("\n" + "=" * 50)
    print(f"Average silence: {np.mean(delays):.2f}s")
    print(f"Maximum silence: {np.max(delays):.2f}s")
    print(f"Files with >0.5s silence: {sum(1 for d in delays if d > 0.5)}")
    print(f"Files with >1.0s silence: {sum(1 for d in delays if d > 1.0)}")
    
    return silence_stats

if __name__ == "__main__":
    print("=" * 60)
    print("🎤 HEY MAGIS RECORDING TRIMMER")
    print("=" * 60)
    print("\nThis tool will remove silence from the beginning of your recordings")
    print("and ensure they're all exactly 2 seconds long.\n")
    
    # For Windows paths in PowerShell
    if os.name == 'nt':  # Windows
        default_input = r"data\positive"
        default_output = r"data\positive_trimmed"
        default_backup = r"data\positive_original"
    else:
        default_input = "data/positive"
        default_output = "data/positive_trimmed"
        default_backup = "data/positive_original"
    
    print("Options:")
    print("1. Analyze recordings (see how much silence they have)")
    print("2. Trim all recordings")
    print("3. Trim and replace originals")
    print("4. Trim to new folder\n")
    
    choice = input("Choose option (1-4): ").strip()
    
    if choice == "1":
        # Just analyze
        input_dir = input(f"Input directory [{default_input}]: ").strip() or default_input
        stats = analyze_recordings(input_dir)
        
        # Also check for empty recordings
        empty_count = 0
        print("\n" + "=" * 50)
        print("Checking for empty/silent recordings...")
        
        for stat in stats:
            try:
                audio, sr = librosa.load(Path(input_dir) / stat['file'], sr=16000)
                if is_silent_recording(audio):
                    empty_count += 1
                    print(f"  ⚠️ EMPTY: {stat['file']}")
            except:
                pass
        
        if empty_count > 0:
            print(f"\n⚠️  Found {empty_count} empty/silent recordings that should be removed!")
        
    elif choice == "2":
        # Trim all recordings to new folder
        input_dir = input(f"Input directory [{default_input}]: ").strip() or default_input
        output_dir = input(f"Output directory [{default_output}]: ").strip() or default_output
        backup_dir = input(f"Backup directory (or press Enter to skip): ").strip()
        
        print("\n⚠️  The script will automatically skip empty/silent recordings.")
        remove_silent = input("Move silent recordings to separate folder? (yes/no): ").strip().lower()
        
        if remove_silent == "yes":
            silent_dir = "data/silent_recordings"
            process_all_recordings(input_dir, output_dir, 
                                 backup_dir if backup_dir else None,
                                 silent_dir)
        else:
            process_all_recordings(input_dir, output_dir, 
                                 backup_dir if backup_dir else None)
        
    elif choice == "3":
        # Trim and replace
        input_dir = input(f"Directory to process [{default_input}]: ").strip() or default_input
        backup_dir = input(f"Backup directory [{default_backup}]: ").strip() or default_backup
        
        print(f"\n⚠️  WARNING: This will replace all files in {input_dir}")
        confirm = input("Are you sure? (yes/no): ").strip().lower()
        
        if confirm == "yes":
            # Process to temp directory first
            temp_dir = "data/temp_trimmed"
            process_all_recordings(input_dir, temp_dir, backup_dir)
            
            # Then copy back
            temp_path = Path(temp_dir)
            input_path = Path(input_dir)
            
            for wav_file in temp_path.glob("*.wav"):
                shutil.copy2(wav_file, input_path / wav_file.name)
            
            # Clean up temp
            shutil.rmtree(temp_dir)
            print(f"\n✓ Original files replaced with trimmed versions")
            print(f"✓ Backups saved to: {backup_dir}")
            
    elif choice == "4":
        # Custom paths
        input_dir = input("Enter input directory path: ").strip()
        output_dir = input("Enter output directory path: ").strip()
        process_all_recordings(input_dir, output_dir)
    
    print("\n✅ Done! Your recordings are ready for training.")
    print("\nNext step: Run the training script with your trimmed recordings!")
