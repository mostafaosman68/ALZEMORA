import os
import torch
import torchaudio
import numpy as np
import sounddevice as sd
import time
import wave
import threading
from sklearn.metrics.pairwise import cosine_similarity

# Set environment variables before importing speechbrain
os.environ["SPEECHBRAIN_STRATEGY"] = "copy"

# Now import SpeechBrain
from speechbrain.inference.speaker import EncoderClassifier

class VoiceIdentifier:
    def __init__(self, samples_dir, threshold=0.65):
        """Initialize the voice identifier."""
        # Use GPU if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Load speaker recognition model
        print("Loading speaker recognition model...")
        self.verification = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb", 
            run_opts={"device": self.device}
        )
        
        # Storage for known speaker embeddings
        self.speaker_embeddings = {}
        self.threshold = threshold
        
        # Load voice samples
        self.load_voice_samples(samples_dir)
        
        # Audio recording parameters
        self.sample_rate = 16000
        
    def load_voice_samples(self, samples_dir):
        """Load and process voice samples from the directory."""
        print(f"Loading voice samples from {samples_dir}...")
        
        # Group files by speaker folder
        speaker_files = {}
        
        # Iterate through each folder in the samples directory
        for folder_name in os.listdir(samples_dir):
            folder_path = os.path.join(samples_dir, folder_name)
            
            # Skip if not a directory
            if not os.path.isdir(folder_path):
                continue
                
            speaker_name = folder_name
            speaker_files[speaker_name] = []
            
            # Get all audio files in this speaker's folder
            for filename in os.listdir(folder_path):
                if filename.endswith(('.wav', '.mp3', '.flac')):
                    speaker_files[speaker_name].append(os.path.join(folder_path, filename))
        
        # Process each speaker's files
        for speaker, files in speaker_files.items():
            print(f"Processing {len(files)} files for speaker: {speaker}")
            self.register_speaker(speaker, files)
            
        print(f"Loaded {len(self.speaker_embeddings)} speakers")
        for name in self.speaker_embeddings:
            print(f"  - {name}")
            
    def extract_embedding(self, audio_file):
        """Extract voice embedding from an audio file."""
        try:
            signal, fs = torchaudio.load(audio_file)
            
            # Convert to mono if stereo
            if signal.shape[0] > 1:
                signal = torch.mean(signal, dim=0).unsqueeze(0)
                
            # Resample if needed
            if fs != 16000:
                resampler = torchaudio.transforms.Resample(fs, 16000)
                signal = resampler(signal)
            
            # Get embedding
            with torch.no_grad():
                embedding = self.verification.encode_batch(signal.to(self.device))
            
            return embedding.squeeze().cpu().numpy()
        except Exception as e:
            print(f"Error processing {audio_file}: {e}")
            return None
    
    def register_speaker(self, name, audio_files):
        """Register a new speaker with multiple audio samples."""
        embeddings = []
        
        # Process each audio file
        for audio_file in audio_files:
            embedding = self.extract_embedding(audio_file)
            if embedding is not None:
                embeddings.append(embedding)
        
        if embeddings:
            # Store the average embedding for the speaker
            self.speaker_embeddings[name] = np.mean(embeddings, axis=0)
            print(f"Speaker '{name}' registered successfully with {len(embeddings)} samples")
        else:
            print(f"Failed to register speaker '{name}' - no valid embeddings")
    
    def identify_speaker(self, audio_data=None, audio_file=None):
        """Identify the speaker from audio data or file."""
        if not self.speaker_embeddings:
            return "No speakers registered", 0.0
        
        # Handle audio file
        if audio_file is not None:
            test_embedding = self.extract_embedding(audio_file)
        # Handle raw audio data
        elif audio_data is not None:
            # Convert to torch tensor
            signal = torch.FloatTensor(audio_data).unsqueeze(0)
            
            # Get embedding
            with torch.no_grad():
                test_embedding = self.verification.encode_batch(signal.to(self.device))
                test_embedding = test_embedding.squeeze().cpu().numpy()
        else:
            return "No audio provided", 0.0
        
        # Compare with all registered speakers
        best_score = -1
        best_speaker = "Unknown"
        
        for name, embedding in self.speaker_embeddings.items():
            # Calculate cosine similarity
            similarity = cosine_similarity([test_embedding], [embedding])[0][0]
            print(f"Similarity with {name}: {similarity:.4f}")
            
            if similarity > best_score:
                best_score = similarity
                best_speaker = name
        
        # Only return a match if confidence exceeds threshold
        if best_score >= self.threshold:
            return best_speaker, best_score
        else:
            return "Unknown", best_score
    
    def record_and_identify(self, duration=8):  # Increased from 5 to 8 seconds
        """Record audio for a fixed duration and identify the speaker."""
        print(f"Recording for {duration} seconds...")
        print("Please speak continuously during the recording...")
        
        # Create buffer for recording
        frames = []
        
        # Callback function to collect audio data
        def callback(indata, frame_count, time_info, status):
            if status:
                print(f"Error in recording: {status}")
            frames.append(indata.copy())
        
        # Record audio
        with sd.InputStream(samplerate=self.sample_rate, channels=1, callback=callback):
            sd.sleep(int(duration * 1000))  # Sleep for duration in milliseconds
        
        print("Recording finished.")
        
        # Convert frames to continuous array
        audio_data = np.concatenate(frames, axis=0).flatten()
        
        # Basic preprocessing: normalize audio amplitude
        if np.abs(audio_data).max() > 0:
            audio_data = audio_data / np.abs(audio_data).max() * 0.9
        
        # Save the recording to a WAV file
        filename = f"recording_{int(time.time())}.wav"
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(self.sample_rate)
            wf.writeframes((audio_data * 32767).astype(np.int16).tobytes())
        
        print(f"Saved recording to {filename}")
        
        # Process and identify the speaker
        speaker, confidence = self.identify_speaker(audio_file=filename)
        
        print(f"Identified speaker: {speaker} (confidence: {confidence:.2f})")
        return speaker, confidence, filename
# Example usage
if __name__ == "__main__":
    # Initialize with directory containing voice samples
    identifier = VoiceIdentifier(
        samples_dir=r"C:/Users/ITD/Desktop/GUI/Data", 
        threshold=0.45  # Lower threshold since we're having trouble with recognition
    )
    
    while True:
        input("Press Enter to start recording (or Ctrl+C to quit)...")
        try:
            speaker, confidence, filename = identifier.record_and_identify(duration=5)
            print("\n")
        except KeyboardInterrupt:
            print("\nProgram terminated by user")
            break