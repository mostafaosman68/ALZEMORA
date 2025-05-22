import os
import torch
import torchaudio
import numpy as np
import threading
import time
import sounddevice as sd
from sklearn.metrics.pairwise import cosine_similarity

# Set environment variables before importing speechbrain
os.environ["SPEECHBRAIN_STRATEGY"] = "copy"

# Now import SpeechBrain
from speechbrain.inference.speaker import EncoderClassifier

class RealtimeVoiceIdentifier:
    def __init__(self, samples_dir, threshold=0.75):
        """Initialize the real-time voice identifier."""
        # Use GPU if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Load speaker recognition model - using EncoderClassifier directly to avoid symlink issues
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
        self.is_recording = False
        self.recording_thread = None
        self.buffer_duration = 5  # seconds
        
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
            
            if similarity > best_score:
                best_score = similarity
                best_speaker = name
        
        # Only return a match if confidence exceeds threshold
        if best_score >= self.threshold:
            return best_speaker, best_score
        else:
            return "Unknown", best_score
    
    def start_realtime_identification(self, buffer_seconds=5, update_interval=1.0):
        """Start real-time speaker identification."""
        self.is_recording = True
        self.buffer_duration = buffer_seconds
        
        # Start recording in a separate thread
        self.recording_thread = threading.Thread(target=self._record_and_identify, 
                                               args=(update_interval,))
        self.recording_thread.daemon = True
        self.recording_thread.start()
        
        print("Real-time voice identification started. Speak into the microphone.")
        print("Press Ctrl+C to stop.")
    
    def stop_realtime_identification(self):
        """Stop real-time identification."""
        self.is_recording = False
        if self.recording_thread:
            self.recording_thread.join(timeout=1.0)
        print("Real-time voice identification stopped")
    
    def _record_and_identify(self, update_interval):
        """Record audio and perform identification in real-time."""
        # Create a rolling buffer
        buffer_size = int(self.buffer_duration * self.sample_rate)
        audio_buffer = np.zeros(buffer_size, dtype=np.float32)
        
        def audio_callback(indata, frames, time, status):
            """Callback for audio processing."""
            if status:
                print(f"Audio callback status: {status}")
            
            # Update the rolling buffer
            audio_buffer[:buffer_size-frames] = audio_buffer[frames:]
            audio_buffer[buffer_size-frames:] = indata[:, 0]
        
        # Start audio stream
        with sd.InputStream(samplerate=self.sample_rate, channels=1, 
                          callback=audio_callback):
            
            last_process_time = time.time()
            
            while self.is_recording:
                current_time = time.time()
                
                # Process every update_interval seconds
                if current_time - last_process_time >= update_interval:
                    # Run voice identification on current buffer
                    speaker, confidence = self.identify_speaker(audio_data=audio_buffer)
                    print(f"Detected speaker: {speaker} (confidence: {confidence:.2f})")
                    
                    last_process_time = current_time
                
                time.sleep(0.01)  # Small sleep to prevent CPU overuse


# Example usage
if __name__ == "__main__":
    # Initialize with directory containing voice samples
    identifier = RealtimeVoiceIdentifier(samples_dir=r"C:/Users/ITD/Desktop/GUI/Data", threshold=0.45)
    
    try:
        # Start real-time identification
        identifier.start_realtime_identification(buffer_seconds=3, update_interval=1.0)
        
        # Keep running until Ctrl+C
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        # Stop on Ctrl+C
        identifier.stop_realtime_identification()
        print("Program terminated by user")