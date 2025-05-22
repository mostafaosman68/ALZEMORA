import os
import time
import wave
import sounddevice as sd
import numpy as np

def record_training_sample(output_dir, speaker_name, duration=5, sample_rate=16000):
    """
    Record an audio sample and save it to the specified directory.
    
    Args:
        output_dir: Base directory for samples
        speaker_name: Name of the speaker (will create/use subfolder with this name)
        duration: Recording duration in seconds
        sample_rate: Audio sample rate
    """
    # Create speaker directory if it doesn't exist
    speaker_dir = os.path.join(output_dir, speaker_name)
    os.makedirs(speaker_dir, exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = int(time.time())
    filename = os.path.join(speaker_dir, f"{speaker_name}_{timestamp}.wav")
    
    print(f"Recording {duration} seconds for {speaker_name}...")
    print("Please speak continuously...")
    
    # Record audio
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    sd.wait()  # Wait until recording is finished
    
    # Normalize audio
    if np.abs(audio_data).max() > 0:
        audio_data = audio_data / np.abs(audio_data).max() * 0.9
    
    # Save as WAV file
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes((audio_data * 32767).astype(np.int16).tobytes())
    
    print(f"Saved to {filename}")
    return filename

def record_multiple_samples(output_dir, speaker_name, num_samples=5, duration=5):
    """Record multiple samples for a speaker with pauses between recordings."""
    print(f"Going to record {num_samples} samples for {speaker_name}")
    print("Each sample will be {duration} seconds long")
    
    recorded_files = []
    
    for i in range(num_samples):
        input(f"Press Enter to start recording sample {i+1}/{num_samples}...")
        filename = record_training_sample(output_dir, speaker_name, duration)
        recorded_files.append(filename)
        
        if i < num_samples - 1:
            print("Recording complete. Get ready for the next sample.")
            time.sleep(1)
    
    print(f"Recorded {num_samples} samples for {speaker_name}.")
    return recorded_files

# Example usage
if __name__ == "__main__":
    # Set your output directory and speaker name
    output_dir = r"C:/Users/ITD/Desktop/GUI/Data"
    speaker_name = "Mostafa"  # Change to the desired speaker name
    
    # Record 10 samples, each 8 seconds long
    recorded_files = record_multiple_samples(
        output_dir=output_dir,
        speaker_name=speaker_name,
        num_samples=10,
        duration=8  # Longer duration for better quality samples
    )
    
    print("\nAll recordings complete!")
    print(f"Saved {len(recorded_files)} files:")
    for file in recorded_files:
        print(f"  - {file}")