#!/usr/bin/env python3
"""
Train Hey Magis Wake Word Model - Production Ready
Uses your 200+ recordings to create a TFLite model for FlutterFlow
"""

import os
import numpy as np
import tensorflow as tf
from pathlib import Path
import librosa
import json
import random
from datetime import datetime

print("🎯 HEY MAGIS WAKE WORD TRAINER")
print("=" * 60)

class HeyMagisTrainer:
    def __init__(self):
        self.sample_rate = 16000
        self.duration = 2.0  # seconds
        self.n_mfcc = 40
        self.model = None
        
    def augment_audio(self, audio, sr):
        """Create variations of audio for better training"""
        augmented = []
        
        # Original
        augmented.append(audio)
        
        # Speed variations
        augmented.append(librosa.effects.time_stretch(audio, rate=0.9))
        augmented.append(librosa.effects.time_stretch(audio, rate=1.1))
        
        # Pitch variations
        augmented.append(librosa.effects.pitch_shift(audio, sr=sr, n_steps=-1))
        augmented.append(librosa.effects.pitch_shift(audio, sr=sr, n_steps=1))
        
        # Add slight noise
        noise = np.random.normal(0, 0.005, len(audio))
        augmented.append(audio + noise)
        
        return augmented
    
    def extract_features(self, audio_path, augment=False):
        """Extract MFCC features from audio file"""
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=self.sample_rate, duration=self.duration)
            
            # Normalize length
            target_length = int(self.sample_rate * self.duration)
            if len(audio) < target_length:
                audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
            else:
                audio = audio[:target_length]
            
            features = []
            
            # Get augmented versions if training positive samples
            if augment:
                audio_variants = self.augment_audio(audio, sr)
            else:
                audio_variants = [audio]
            
            for audio_sample in audio_variants:
                # Ensure correct length after augmentation
                if len(audio_sample) < target_length:
                    audio_sample = np.pad(audio_sample, (0, target_length - len(audio_sample)), mode='constant')
                elif len(audio_sample) > target_length:
                    audio_sample = audio_sample[:target_length]
                
                # Extract MFCC
                mfcc = librosa.feature.mfcc(
                    y=audio_sample,
                    sr=sr,
                    n_mfcc=self.n_mfcc,
                    n_fft=512,
                    hop_length=160,
                    n_mels=40
                )
                
                # Transpose to (time, features)
                mfcc = mfcc.T
                
                # Normalize
                mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-10)
                
                features.append(mfcc)
            
            return features
            
        except Exception as e:
            print(f"  ⚠️ Error processing {audio_path}: {e}")
            return None
    
    def prepare_dataset(self):
        """Prepare the complete dataset"""
        print("\n📊 Preparing dataset...")
        
        X = []
        y = []
        
        # =====================================================
        # POSITIVE SAMPLES - Your "Hey Magis" recordings
        # =====================================================
        positive_dir = Path("data/positive_trimmed")
        positive_files = list(positive_dir.glob("*.wav"))
        
        print(f"\n🎤 Processing {len(positive_files)} Hey Magis recordings...")
        
        for i, audio_file in enumerate(positive_files):
            if (i + 1) % 50 == 0:
                print(f"   Processed {i + 1}/{len(positive_files)} positive samples...")
            features_list = self.extract_features(str(audio_file), augment=True)
            if features_list:
                for features in features_list:
                    X.append(features)
                    y.append(1)
        
        positive_count = len([y_val for y_val in y if y_val == 1])
        print(f"✅ Created {positive_count} positive samples (with augmentation)")
        
        # =====================================================
        # NEGATIVE SAMPLES - Your similar phrase recordings
        # =====================================================
        negative_dir = Path("data/negative")
        
        # Load YOUR recordings (similar phrases like "Hey Marcus", etc.)
        your_recordings = list(negative_dir.glob("Recording*.wav"))
        print(f"\n🔇 Processing {len(your_recordings)} of your similar phrase recordings...")
        
        your_negative_count = 0
        for audio_file in your_recordings:
            # Augment your recordings too since we have fewer of them
            features_list = self.extract_features(str(audio_file), augment=True)
            if features_list:
                for features in features_list:
                    X.append(features)
                    y.append(0)
                    your_negative_count += 1
        
        print(f"✅ Created {your_negative_count} negative samples from your recordings (with augmentation)")
        
        # =====================================================
        # NEGATIVE SAMPLES - Speech Commands dataset
        # =====================================================
        categories = [
            'yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 
            'stop', 'go', 'zero', 'one', 'two', 'three', 'four', 
            'five', 'six', 'seven', 'eight', 'nine', 'bed', 'bird',
            'cat', 'dog', 'happy', 'house', 'tree', 'wow', 'marvin',
            'sheila', 'backward', 'forward', 'follow', 'learn', 'visual'
        ]
        
        # Calculate how many Speech Commands samples we need
        # Target: 3:1 negative to positive ratio
        target_negative = positive_count * 3
        remaining_needed = target_negative - your_negative_count
        per_category = max(50, remaining_needed // len(categories))
        
        print(f"\n📚 Processing Speech Commands dataset...")
        print(f"   Target: {remaining_needed} samples from {len(categories)} categories")
        
        speech_commands_count = 0
        for category in categories:
            category_dir = negative_dir / category
            if category_dir.exists():
                files = list(category_dir.glob("*.wav"))
                # Randomly sample from each category
                sample_files = random.sample(files, min(per_category, len(files)))
                
                for audio_file in sample_files:
                    features_list = self.extract_features(str(audio_file), augment=False)
                    if features_list:
                        for features in features_list:
                            X.append(features)
                            y.append(0)
                            speech_commands_count += 1
        
        print(f"✅ Loaded {speech_commands_count} negative samples from Speech Commands")
        
        # =====================================================
        # BACKGROUND NOISE (for robustness)
        # =====================================================
        background_dir = negative_dir / "_background_noise_"
        if background_dir.exists():
            print(f"\n🔊 Processing background noise samples...")
            noise_files = list(background_dir.glob("*.wav"))
            noise_count = 0
            
            for noise_file in noise_files:
                # Extract multiple segments from each noise file
                try:
                    audio, sr = librosa.load(str(noise_file), sr=self.sample_rate)
                    segment_length = int(self.sample_rate * self.duration)
                    
                    # Extract up to 20 random segments per noise file
                    for _ in range(min(20, len(audio) // segment_length)):
                        start = random.randint(0, len(audio) - segment_length)
                        segment = audio[start:start + segment_length]
                        
                        # Extract MFCC
                        mfcc = librosa.feature.mfcc(
                            y=segment, sr=sr, n_mfcc=self.n_mfcc,
                            n_fft=512, hop_length=160, n_mels=40
                        )
                        mfcc = mfcc.T
                        mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-10)
                        
                        X.append(mfcc)
                        y.append(0)
                        noise_count += 1
                except Exception as e:
                    print(f"  ⚠️ Error with noise file {noise_file}: {e}")
            
            print(f"✅ Added {noise_count} background noise samples")
        
        # Convert to arrays
        X = np.array(X)
        y = np.array(y)
        
        print(f"\n" + "=" * 50)
        print(f"📈 DATASET STATISTICS")
        print(f"=" * 50)
        print(f"   Total samples: {len(X)}")
        print(f"   Positive (Hey Magis): {sum(y==1)}")
        print(f"   Negative (Other): {sum(y==0)}")
        print(f"   Ratio: 1:{sum(y==0)/max(1, sum(y==1)):.1f}")
        print(f"   Feature shape: {X.shape}")
        print(f"=" * 50)
        
        # Split into train/test
        from sklearn.model_selection import train_test_split
        return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    def build_model(self, input_shape):
        """Build the wake word detection model"""
        
        model = tf.keras.Sequential([
            # Input layer
            tf.keras.layers.Input(shape=input_shape),
            
            # CNN layers for feature extraction
            tf.keras.layers.Conv1D(64, 3, activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling1D(2),
            tf.keras.layers.Dropout(0.25),
            
            tf.keras.layers.Conv1D(128, 3, activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling1D(2),
            tf.keras.layers.Dropout(0.25),
            
            tf.keras.layers.Conv1D(256, 3, activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.GlobalMaxPooling1D(),
            
            # Dense layers
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.5),
            
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.5),
            
            # Output layer
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        # Compile with optimized settings
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.AUC(name='auc')
            ]
        )
        
        return model
    
    def train(self):
        """Train the model"""
        print("\n🚀 Starting training...")
        
        # Prepare data
        X_train, X_test, y_train, y_test = self.prepare_dataset()
        
        # Store for later use
        self.X_train = X_train
        
        # Build model
        self.model = self.build_model(X_train.shape[1:])
        print("\n📐 Model architecture:")
        self.model.summary()
        
        # Create models directory
        Path("models").mkdir(exist_ok=True)
        
        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=0.00001,
                verbose=1
            ),
            tf.keras.callbacks.ModelCheckpoint(
                'models/hey_magis_best.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train
        print("\n⏳ Training model (this may take 20-45 minutes)...")
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=100,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate
        print("\n" + "=" * 50)
        print("📊 FINAL EVALUATION")
        print("=" * 50)
        results = self.model.evaluate(X_test, y_test, verbose=0)
        
        print(f"   Accuracy:  {results[1]:.2%}")
        print(f"   Precision: {results[2]:.2%}")
        print(f"   Recall:    {results[3]:.2%}")
        print(f"   AUC:       {results[4]:.3f}")
        
        # Calculate F1 score
        f1 = 2 * (results[2] * results[3]) / (results[2] + results[3] + 1e-10)
        print(f"   F1 Score:  {f1:.2%}")
        print("=" * 50)
        
        return history
    
    def convert_to_tflite(self):
        """Convert model to TensorFlow Lite for mobile deployment"""
        print("\n📱 Converting to TensorFlow Lite...")
        
        # Convert the model
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        
        # Optimizations for mobile
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        
        # Representative dataset for quantization
        X_train = self.X_train
        def representative_dataset():
            for i in range(min(100, len(X_train))):
                yield [X_train[i:i+1].astype(np.float32)]
        
        converter.representative_dataset = representative_dataset
        
        # Convert
        tflite_model = converter.convert()
        
        # Save the model
        tflite_path = 'models/hey_magis.tflite'
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        
        model_size_mb = len(tflite_model) / (1024 * 1024)
        print(f"✅ TFLite model saved: {tflite_path}")
        print(f"   Size: {model_size_mb:.2f} MB")
        
        # Create metadata file
        metadata = {
            "model_name": "Hey Magis Wake Word",
            "version": "1.0.0",
            "created": datetime.now().isoformat(),
            "wake_phrase": "hey magis",
            "sample_rate": self.sample_rate,
            "duration_seconds": self.duration,
            "mfcc_features": self.n_mfcc,
            "threshold": 0.7,
            "input_shape": list(self.X_train.shape[1:]),
            "training_samples": {
                "positive": "209 real recordings with augmentation",
                "negative": "61 similar phrases + Speech Commands dataset"
            }
        }
        
        metadata_path = 'models/model_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Metadata saved: {metadata_path}")
        
        return tflite_model
    
    def test_model(self, test_audio_path):
        """Test the TFLite model with a sample audio"""
        print(f"\n🧪 Testing model with: {test_audio_path}")
        
        # Load TFLite model
        interpreter = tf.lite.Interpreter(model_path='models/hey_magis.tflite')
        interpreter.allocate_tensors()
        
        # Get input/output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Extract features
        features = self.extract_features(test_audio_path, augment=False)[0]
        input_data = np.expand_dims(features, axis=0).astype(np.float32)
        
        # Run inference
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        
        # Get prediction
        prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]
        
        print(f"   Prediction: {prediction:.2%}")
        print(f"   Result: {'✅ Wake word detected!' if prediction > 0.7 else '❌ Not detected'}")
        
        return prediction


def main():
    """Main training pipeline"""
    
    # Check if data directories exist
    if not Path("data/positive_trimmed").exists():
        print("❌ Error: data/positive_trimmed not found!")
        print("   Please make sure you're running from the wake_word_project folder")
        return
    
    if not Path("data/negative").exists():
        print("❌ Error: data/negative not found!")
        print("   Please make sure you're running from the wake_word_project folder")
        return
    
    # Count files
    positive_count = len(list(Path("data/positive_trimmed").glob("*.wav")))
    negative_recordings = len(list(Path("data/negative").glob("Recording*.wav")))
    speech_command_folders = len([d for d in Path("data/negative").iterdir() if d.is_dir()])
    
    print(f"\n📁 Dataset found:")
    print(f"   Positive samples (Hey Magis): {positive_count}")
    print(f"   Your negative recordings: {negative_recordings}")
    print(f"   Speech Commands folders: {speech_command_folders}")
    
    if positive_count == 0:
        print("\n❌ No positive samples found! Add your Hey Magis recordings first.")
        return
    
    # Create models directory
    Path("models").mkdir(exist_ok=True)
    
    # Initialize trainer
    trainer = HeyMagisTrainer()
    
    # Train the model
    history = trainer.train()
    
    # Convert to TFLite
    tflite_model = trainer.convert_to_tflite()
    
    # Test with a sample
    positive_files = list(Path("data/positive_trimmed").glob("*.wav"))
    negative_files = list(Path("data/negative").glob("Recording*.wav"))
    
    print("\n" + "=" * 60)
    print("🧪 TESTING MODEL")
    print("=" * 60)
    
    if positive_files:
        print("\nTesting with a POSITIVE sample (should detect):")
        trainer.test_model(str(positive_files[0]))
    
    if negative_files:
        print("\nTesting with a NEGATIVE sample (should NOT detect):")
        trainer.test_model(str(negative_files[0]))
    
    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETE!")
    print("=" * 60)
    print("\n📦 Generated Files (in 'models' folder):")
    print("   1. hey_magis.tflite - Model for FlutterFlow")
    print("   2. model_metadata.json - Configuration")
    print("   3. hey_magis_best.h5 - Keras model backup")
    print("\n📱 Next Steps:")
    print("   1. Copy hey_magis.tflite to FlutterFlow assets")
    print("   2. Add the wake word detection custom action")
    print("   3. Test on your device!")
    print("\n🎯 Your 'Hey Magis' wake word is ready for deployment!")


if __name__ == "__main__":
    main()