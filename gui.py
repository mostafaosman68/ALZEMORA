import os
import torch
import torchaudio
import numpy as np
import threading
import time
import sounddevice as sd
from sklearn.metrics.pairwise import cosine_similarity
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Set environment variables before importing speechbrain
os.environ["SPEECHBRAIN_STRATEGY"] = "copy"

# Now import SpeechBrain
from speechbrain.inference.speaker import EncoderClassifier
def download_model_once(model_name="speechbrain/spkrec-ecapa-voxceleb", save_dir="./pretrained_models"):
    """Download the model files once and save them locally."""
    import os
    import ssl
    import requests
    from speechbrain.pretrained import EncoderClassifier
    
    # Create a custom SSL context that's more permissive
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    
    # Temporarily disable SSL verification for requests
    old_context = ssl._create_default_https_context
    ssl._create_default_https_context = ssl._create_unverified_context
    
    # Create the directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    try:
        # Download the model once
        print(f"Downloading model {model_name} to {save_dir}...")
        EncoderClassifier.from_hparams(
            source=model_name,
            savedir=save_dir,
            run_opts={"device": "cpu"}  # Use CPU for download only
        )
        print("Model downloaded successfully!")
        return True
    except Exception as e:
        print(f"Error downloading model: {e}")
        return False
    finally:
        # Restore the original SSL context
        ssl._create_default_https_context = old_context

class VoiceIdentifierGUI:
    def __init__(self, samples_dir, threshold=0.45):
        """Initialize the voice identifier with GUI."""
        # Use GPU if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Set up model directory
        model_dir = "./pretrained_models"
        
        # Try to download the model if needed (only happens once)
        if not os.path.exists(model_dir):
            download_model_once(save_dir=model_dir)
        
        # Load speaker recognition model
        print("Loading speaker recognition model...")
        try:
            # Try loading from local directory first
            self.verification = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb", 
                savedir=model_dir,
                run_opts={"device": self.device}
            )
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Trying alternative loading method...")
            # If that fails, try with SSL workaround
            import ssl
            old_context = ssl._create_default_https_context
            ssl._create_default_https_context = ssl._create_unverified_context
            try:
                self.verification = EncoderClassifier.from_hparams(
                    source="speechbrain/spkrec-ecapa-voxceleb", 
                    run_opts={"device": self.device}
                )
            finally:
                ssl._create_default_https_context = old_context
    
        # Storage for known speaker embeddings
        self.speaker_embeddings = {}
        self.threshold = threshold
        
        # Audio recording parameters
        self.sample_rate = 16000
        self.is_recording = False
        self.recording_thread = None
        
        # Load voice samples
        self.load_voice_samples(samples_dir)
        
        # Create GUI
        self.create_gui()
        
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
            return "No speakers registered", 0.0, {}
        
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
            return "No audio provided", 0.0, {}
        
        # Compare with all registered speakers
        best_score = -1
        best_speaker = "Unknown"
        all_scores = {}
        
        for name, embedding in self.speaker_embeddings.items():
            # Calculate cosine similarity
            similarity = cosine_similarity([test_embedding], [embedding])[0][0]
            all_scores[name] = similarity
            
            if similarity > best_score:
                best_score = similarity
                best_speaker = name
        
        # Only return a match if confidence exceeds threshold
        if best_score >= self.threshold:
            return best_speaker, best_score, all_scores
        else:
            return "Unknown", best_score, all_scores
    
    def create_gui(self):
        """Create the GUI for voice identification."""
        self.root = tk.Tk()
        self.root.title("Voice Identification System")
        self.root.geometry("800x600")
        self.root.configure(bg="#f0f0f0")  # Light gray background
        
        # Set custom styles
        self.setup_styles()
        
        # Create main frame
        main_frame = ttk.Frame(self.root, padding="10", style="Main.TFrame")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title label
        title_label = ttk.Label(main_frame, text="Voice Identification System", 
                               font=("Arial", 22, "bold"), style="Title.TLabel")
        title_label.pack(pady=10)
        
        # Status frame with gradient background
        status_frame = ttk.LabelFrame(main_frame, text="Status", padding="10", style="Card.TLabelframe")
        status_frame.pack(fill=tk.X, pady=10, padx=10)
        
        # Status label with background color
        self.status_var = tk.StringVar(value="Not Listening")
        status_label = ttk.Label(status_frame, textvariable=self.status_var, 
                               font=("Arial", 12), style="Status.TLabel")
        status_label.pack(pady=5, fill=tk.X)
        
        # Result frame with shadow effect
        result_frame = ttk.LabelFrame(main_frame, text="Identification Result", 
                                    padding="10", style="Card.TLabelframe")
        result_frame.pack(fill=tk.X, pady=10, padx=10)
        
        # Speaker label with distinct styling
        self.speaker_var = tk.StringVar(value="No speaker detected")
        speaker_label = ttk.Label(result_frame, textvariable=self.speaker_var, 
                                font=("Arial", 18, "bold"), style="Speaker.TLabel")
        speaker_label.pack(pady=5)
        
        # Confidence label
        self.confidence_var = tk.StringVar(value="")
        confidence_label = ttk.Label(result_frame, textvariable=self.confidence_var, 
                                   font=("Arial", 12), style="Confidence.TLabel")
        confidence_label.pack(pady=5)
        
        # Styled confidence meter
        self.confidence_frame = tk.Frame(result_frame, bg="#e0e0e0", height=30, bd=1, relief=tk.SOLID)
        self.confidence_frame.pack(fill=tk.X, pady=10, padx=5)
        
        self.confidence_meter = tk.Canvas(self.confidence_frame, height=28, bg="#e0e0e0", 
                                        bd=0, highlightthickness=0)
        self.confidence_meter.pack(fill=tk.X)
        self.confidence_value = 0
        self.draw_confidence_meter()
        
        # Create frame for bar chart
        chart_frame = ttk.LabelFrame(main_frame, text="Speaker Confidence Scores", 
                                   padding="10", style="Card.TLabelframe")
        chart_frame.pack(fill=tk.BOTH, expand=True, pady=10, padx=10)
        
        # Create matplotlib figure for bar chart with custom styling
        plt.style.use('ggplot')  # Use a nicer style for the plot
        self.fig, self.ax = plt.subplots(figsize=(8, 4))
        self.fig.patch.set_facecolor('#f8f8f8')  # Light background
        self.canvas = FigureCanvasTkAgg(self.fig, master=chart_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Buttons frame
        button_frame = ttk.Frame(main_frame, style="Main.TFrame")
        button_frame.pack(fill=tk.X, pady=15, padx=10)
        
        # Record button with custom styling
        self.record_button = tk.Button(button_frame, text="Record and Identify", 
                                     command=self.record_and_identify_once,
                                     bg="#4CAF50", fg="white", font=("Arial", 12),
                                     activebackground="#3e8e41", height=2,
                                     relief=tk.RAISED, bd=2)
        self.record_button.pack(side=tk.LEFT, padx=5)
        
        # Add hover effect to record button
        self.record_button.bind("<Enter>", lambda e: self.button_hover(e, "#3e8e41"))
        self.record_button.bind("<Leave>", lambda e: self.button_leave(e, "#4CAF50"))
        
        # Start continuous listening button with custom styling
        self.start_button = tk.Button(button_frame, text="Start Continuous Listening", 
                                    command=self.start_continuous_identification,
                                    bg="#2196F3", fg="white", font=("Arial", 12),
                                    activebackground="#0b7dda", height=2,
                                    relief=tk.RAISED, bd=2)
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        # Add hover effect to start button
        self.start_button.bind("<Enter>", lambda e: self.button_hover(e, "#0b7dda"))
        self.start_button.bind("<Leave>", lambda e: self.button_leave(e, "#2196F3"))
        
        # Stop button with custom styling
        self.stop_button = tk.Button(button_frame, text="Stop Listening", 
                                   command=self.stop_identification,
                                   bg="#f44336", fg="white", font=("Arial", 12),
                                   activebackground="#d32f2f", height=2,
                                   relief=tk.RAISED, bd=2, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        # Add hover effect to stop button when enabled
        self.stop_button.bind("<Enter>", lambda e: self.button_hover(e, "#d32f2f", check_state=True))
        self.stop_button.bind("<Leave>", lambda e: self.button_leave(e, "#f44336", check_state=True))
        
        # Exit button with custom styling
        exit_button = tk.Button(button_frame, text="Exit", 
                               command=self.exit_application,
                               bg="#9E9E9E", fg="white", font=("Arial", 12),
                               activebackground="#757575", height=2,
                               relief=tk.RAISED, bd=2)
        exit_button.pack(side=tk.RIGHT, padx=5)
        
        # Add hover effect to exit button
        exit_button.bind("<Enter>", lambda e: self.button_hover(e, "#757575"))
        exit_button.bind("<Leave>", lambda e: self.button_leave(e, "#9E9E9E"))
        
        # Initialize the bar chart
        self.update_bar_chart({})
        
        # Set up closing protocol
        self.root.protocol("WM_DELETE_WINDOW", self.exit_application)
        
        # Animation for initial appearance
        self.animate_startup()
    
    def setup_styles(self):
        """Set up custom styles for ttk widgets."""
        style = ttk.Style()
        
        # Main frame style
        style.configure("Main.TFrame", background="#f0f0f0")
        
        # Title label style
        style.configure("Title.TLabel", foreground="#2c3e50", background="#f0f0f0",
                        font=("Arial", 22, "bold"))
        
        # Status label style
        style.configure("Status.TLabel", foreground="#444", background="#e8e8e8",
                        font=("Arial", 12))
        
        # Speaker label style
        style.configure("Speaker.TLabel", foreground="#2c3e50", background="#f0f0f0",
                        font=("Arial", 18, "bold"))
        
        # Confidence label style
        style.configure("Confidence.TLabel", foreground="#555", background="#f0f0f0",
                        font=("Arial", 12))
        
        # Card frame style
        style.configure("Card.TLabelframe", background="#f8f8f8", borderwidth=2)
        style.configure("Card.TLabelframe.Label", font=("Arial", 11, "bold"), 
                        foreground="#555", background="#f0f0f0")
    
    def button_hover(self, event, color, check_state=False):
        """Change button color on hover."""
        if check_state and event.widget["state"] == tk.DISABLED:
            return
        event.widget.config(bg=color)
    
    def button_leave(self, event, color, check_state=False):
        """Restore button color when mouse leaves."""
        if check_state and event.widget["state"] == tk.DISABLED:
            return
        event.widget.config(bg=color)
    
    def draw_confidence_meter(self):
        """Draw the custom confidence meter."""
        self.confidence_meter.delete("all")
        width = self.confidence_meter.winfo_width()
        if width <= 1:  # Not yet fully created
            width = 790  # Default width
        
        # Draw background
        self.confidence_meter.create_rectangle(0, 0, width, 28, fill="#e0e0e0", outline="")
        
        # Draw value - with gradient from blue to green
        if self.confidence_value > 0:
            value_width = width * (self.confidence_value / 100)
            
            # Create gradient
            if self.confidence_value < 50:
                # From red to yellow
                r = int(255)
                g = int(self.confidence_value * 2 * 255 / 100)
                b = 0
            else:
                # From yellow to green
                r = int(255 - (self.confidence_value - 50) * 2 * 255 / 100)
                g = int(255)
                b = 0
            
            color = f"#{r:02x}{g:02x}{b:02x}"
            self.confidence_meter.create_rectangle(0, 0, value_width, 28, 
                                               fill=color, outline="")
            
            # Add gloss effect
            self.confidence_meter.create_rectangle(0, 0, value_width, 14, 
                                               fill="white", outline="", stipple="gray25")
            
            # Add text on meter
            if value_width > 30:  # Only show text if there's room
                self.confidence_meter.create_text(value_width/2, 14, 
                                             text=f"{self.confidence_value:.0f}%",
                                             fill="black", font=("Arial", 10, "bold"))
        
        # Draw border
        self.confidence_meter.create_rectangle(0, 0, width, 28, outline="#aaa")
    
    def update_bar_chart(self, scores):
        """Update the bar chart with current scores."""
        self.ax.clear()
        
        if not scores:
            # No scores available
            self.ax.text(0.5, 0.5, "No data available", 
                        horizontalalignment='center', verticalalignment='center',
                        transform=self.ax.transAxes, fontsize=14, color='#555')
            self.ax.set_facecolor('#f8f8f8')
        else:
            # Sort scores for better visualization
            names = list(scores.keys())
            values = [scores[name] for name in names]
            
            # Create bar chart with custom styling
            bars = self.ax.bar(names, values, width=0.6, 
                             edgecolor='white', linewidth=1)
            
            # Add threshold line
            self.ax.axhline(y=self.threshold, color='#ff7043', linestyle='--', 
                          linewidth=2, label=f'Threshold ({self.threshold})')
            
            # Color each bar based on threshold
            for i, bar in enumerate(bars):
                if values[i] >= self.threshold:
                    bar.set_color('#4caf50')  # Green for above threshold
                else:
                    bar.set_color('#9e9e9e')  # Gray for below threshold
                
                # Add shine effect to bars
                self.ax.bar(i, values[i] * 0.5, width=0.6, bottom=values[i] * 0.5,
                          color='white', alpha=0.3, linewidth=0)
            
            # Add value labels on top of bars
            for i, v in enumerate(values):
                self.ax.text(i, v + 0.02, f"{v:.2f}", ha='center', fontsize=11,
                           fontweight='bold', color='#333')
            
            # Set labels and title
            self.ax.set_ylabel('Confidence Score', fontsize=12, fontweight='bold')
            self.ax.set_ylim(0, 1.0)  # Confidence scores are between 0 and 1
            
            # Improve grid and background
            self.ax.set_facecolor('#f8f8f8')
            self.ax.grid(color='white', linestyle='-', linewidth=1, alpha=0.7)
            
            # Style the spines
            for spine in self.ax.spines.values():
                spine.set_visible(False)
            
            # Add legend with custom styling
            self.ax.legend(frameon=True, facecolor='#f0f0f0', edgecolor='#ddd',
                         fontsize=10)
        
        # Update the canvas
        self.canvas.draw()
    
    def animate_startup(self):
        """Animate the initial appearance of the app."""
        # Store original positions
        original_geometries = {}
        
        # Hide all widgets initially
        for child in self.root.winfo_children():
            child.pack_forget()
        
        # Repack main frame
        main_frame = None
        for child in self.root.winfo_children():
            if isinstance(child, ttk.Frame):
                main_frame = child
                child.pack(fill=tk.BOTH, expand=True)
                break
        
        if main_frame:
            # Animate children with delay
            children = list(main_frame.winfo_children())
            
            def show_next_widget(index=0):
                if index < len(children):
                    children[index].pack_forget()  # Remove temporarily
                    # Determine pack parameters based on widget type
                    if isinstance(children[index], ttk.Label) and index == 0:  # Title
                        children[index].pack(pady=10)
                    elif isinstance(children[index], ttk.LabelFrame):  # Frames
                        children[index].pack(fill=tk.X, pady=10, padx=10)
                    elif isinstance(children[index], ttk.Frame):  # Button frame
                        children[index].pack(fill=tk.X, pady=15, padx=10)
                    
                    # Schedule next widget
                    self.root.after(150, lambda: show_next_widget(index + 1))
            
            # Start animation
            show_next_widget()
    
    def animate_confidence_meter(self, target_value):
        """Animate the confidence meter to a target value."""
        current = self.confidence_value
        target = target_value * 100  # Convert to percentage
        
        def update_meter(current, target):
            # Calculate step size
            step = (target - current) / 10
            if abs(step) < 0.1:
                step = 1 if step > 0 else -1
            
            # Update current value
            current += step
            
            # Check if we've reached the target
            if (step > 0 and current >= target) or (step < 0 and current <= target):
                current = target
                done = True
            else:
                done = False
            
            # Update meter
            self.confidence_value = current
            self.draw_confidence_meter()
            
            # Continue animation if not done
            if not done:
                self.root.after(30, lambda: update_meter(current, target))
        
        # Start animation
        update_meter(current, target)
    
    def flash_status(self, status, times=3):
        """Flash the status message to get attention."""
        original_bg = self.status_var.get()
        colors = ["#f44336", "#e8e8e8"]  # Red and normal
        
        def flash(count=0):
            if count < times * 2:
                if count % 2 == 0:
                    self.status_var.set(status)
                    for label in self.root.winfo_children()[0].winfo_children():
                        if isinstance(label, ttk.Label) and label.cget("textvariable") == str(self.status_var):
                            label.configure(style="Alert.TLabel")
                else:
                    self.status_var.set(status)
                    for label in self.root.winfo_children()[0].winfo_children():
                        if isinstance(label, ttk.Label) and label.cget("textvariable") == str(self.status_var):
                            label.configure(style="Status.TLabel")
                self.root.after(200, lambda: flash(count + 1))
            else:
                self.status_var.set(status)
                for label in self.root.winfo_children()[0].winfo_children():
                    if isinstance(label, ttk.Label) and label.cget("textvariable") == str(self.status_var):
                        label.configure(style="Status.TLabel")
        
        # Create alert style
        style = ttk.Style()
        style.configure("Alert.TLabel", foreground="white", background="#f44336",
                        font=("Arial", 12, "bold"))
        
        # Start flashing
        flash()
    
    def record_and_identify_once(self):
        """Record audio for a set duration and identify the speaker once."""
        # Disable buttons during recording
        self.record_button.config(state=tk.DISABLED, relief=tk.SUNKEN)
        self.start_button.config(state=tk.DISABLED)
        
        # Update status with animation
        self.flash_status("Recording...")
        self.speaker_var.set("Listening...")
        self.confidence_var.set("")
        self.animate_confidence_meter(0)
        self.root.update()
        
        # Start recording in a separate thread
        threading.Thread(target=self._record_once, daemon=True).start()
    
    def _record_once(self):
        """Record audio for a set duration and process it."""
        duration = 5  # seconds
        
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
        
        # Convert frames to continuous array
        audio_data = np.concatenate(frames, axis=0).flatten()
        
        # Normalize audio data
        if np.abs(audio_data).max() > 0:
            audio_data = audio_data / np.abs(audio_data).max() * 0.9
        
        # Process and identify the speaker
        speaker, confidence, all_scores = self.identify_speaker(audio_data=audio_data)
        
        # Update GUI from the main thread
        self.root.after(0, lambda: self._update_gui_after_recording(speaker, confidence, all_scores))
    
    def _update_gui_after_recording(self, speaker, confidence, all_scores):
        """Update GUI after recording is complete."""
        # Update status
        self.status_var.set("Recording complete")
        
        # Update speaker with animation if identified
        if speaker != "Unknown":
            # Flash the speaker name
            original_text = self.speaker_var.get()
            self.speaker_var.set("")
            self.root.after(200, lambda: self.speaker_var.set(speaker))
        else:
            self.speaker_var.set(speaker)
        
        # Update confidence with animation
        self.confidence_var.set(f"Confidence: {confidence:.2f}")
        self.animate_confidence_meter(confidence)
        
        # Update bar chart with animation
        self.update_bar_chart({})  # Clear first
        self.root.after(300, lambda: self.update_bar_chart(all_scores))  # Then update
        
        # Re-enable buttons with slight delay for better UX
        self.root.after(400, lambda: self._reenable_buttons())
        
        # Log to console
        print(f"Detected: {speaker} (confidence: {confidence:.2f})")
    
    def _reenable_buttons(self):
        """Re-enable buttons after recording."""
        self.record_button.config(state=tk.NORMAL, relief=tk.RAISED)
        self.start_button.config(state=tk.NORMAL)
    
    def start_continuous_identification(self):
        """Start continuous real-time voice identification."""
        if self.is_recording:
            return
        
        self.is_recording = True
        self.flash_status("Continuously listening...")
        self.speaker_var.set("Waiting for voice...")
        self.confidence_var.set("")
        self.animate_confidence_meter(0)
        
        # Update button states with visual effect
        self.record_button.config(state=tk.DISABLED)
        self.start_button.config(state=tk.DISABLED, relief=tk.SUNKEN)
        
        # Animate the stop button becoming enabled
        self.stop_button.config(bg="#aaa")  # Start with gray
        for i in range(5):
            self.root.after(i*100, lambda shade=i: 
                          self.stop_button.config(
                              bg=f"#{255-shade*20:02x}{70+shade*10:02x}{70+shade*10:02x}"))
        
        self.root.after(500, lambda: self.stop_button.config(state=tk.NORMAL))
        
        # Start recording in a separate thread
        self.recording_thread = threading.Thread(target=self._record_continuously)
        self.recording_thread.daemon = True
        self.recording_thread.start()
    
    def stop_identification(self):
        """Stop continuous identification."""
        self.is_recording = False
        self.flash_status("Stopped listening")
        
        # Animate button state changes
        self.stop_button.config(state=tk.DISABLED, relief=tk.SUNKEN)
        
        # Gradually restore button colors
        for i in range(5):
            self.root.after(i*100, lambda shade=i: 
                          self.stop_button.config(
                              bg=f"#{244-shade*30:02x}{67+shade*30:02x}{54+shade*30:02x}"))
        
        # Re-enable other buttons after animation
        self.root.after(500, lambda: self._after_stop_animation())
        
        if self.recording_thread:
            self.recording_thread.join(timeout=1.0)
    
    def _after_stop_animation(self):
        """Actions to take after stop animation completes."""
        self.record_button.config(state=tk.NORMAL)
        self.start_button.config(state=tk.NORMAL, relief=tk.RAISED)
        self.stop_button.config(relief=tk.RAISED, bg="#f44336")
    
    def _record_continuously(self):
        """Record audio and perform identification continuously."""
        buffer_seconds = 5
        update_interval = 3.0  # Process every 3 seconds
        
        # Create a rolling buffer
        buffer_size = int(buffer_seconds * self.sample_rate)
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
                    # Normalize audio data before processing
                    if np.abs(audio_buffer).max() > 0:
                        normalized_buffer = audio_buffer / np.abs(audio_buffer).max() * 0.9
                    else:
                        normalized_buffer = audio_buffer
                    
                    # Run voice identification on current buffer
                    speaker, confidence, all_scores = self.identify_speaker(audio_data=normalized_buffer)
                    
                    # Update GUI from the main thread
                    self.root.after(0, lambda s=speaker, c=confidence, a=all_scores: 
                                   self._update_gui(s, c, a))
                    
                    last_process_time = current_time
                    time.sleep(0.01)  # Small sleep to prevent CPU overuse
    
    def _update_gui(self, speaker, confidence, all_scores):
        """Update GUI with identification results."""
        # Check if previously identified speaker is different
        previous_speaker = self.speaker_var.get()
        speaker_changed = (previous_speaker != speaker and 
                          previous_speaker != "Waiting for voice..." and
                          previous_speaker != "No speaker detected" and
                          previous_speaker != "Listening...")
        
        # Update speaker label with animation if speaker changed
        if speaker_changed:
            # Flash transition effect for speaker change
            self.speaker_var.set("")
            self.root.after(100, lambda: self.speaker_var.set(speaker))
        else:
            self.speaker_var.set(speaker)
        
        # Update confidence label and meter with animation
        self.confidence_var.set(f"Confidence: {confidence:.2f}")
        self.animate_confidence_meter(confidence)
        
        # Update bar chart with animation if values changed significantly
        current_scores = {}
        try:
            # Try to extract values from current chart if it exists
            if hasattr(self, 'ax') and self.ax.patches:
                for i, patch in enumerate(self.ax.patches):
                    if hasattr(patch, 'get_height'):
                        if i < len(self.ax.get_xticklabels()):
                            name = self.ax.get_xticklabels()[i].get_text()
                            current_scores[name] = patch.get_height()
        except:
            pass
        
        # Check if scores changed significantly
        scores_changed = False
        for name, score in all_scores.items():
            if name in current_scores:
                if abs(current_scores[name] - score) > 0.05:
                    scores_changed = True
                    break
            else:
                scores_changed = True
                break
        
        if scores_changed or not current_scores:
            self.update_bar_chart(all_scores)
        
        # Log to console
        print(f"Detected: {speaker} (confidence: {confidence:.2f})")
        
        # Highlight the status if recognized speaker
        if speaker != "Unknown":
            if not self.status_var.get().startswith("Identified"):
                status_text = f"Identified: {speaker}"
                self.status_var.set(status_text)
    
    def exit_application(self):
        """Clean up and exit the application."""
        # Animate exit
        for widget in self.root.winfo_children():
            widget.pack_forget()
        
        # Create exit message
        exit_label = ttk.Label(self.root, text="Thank you for using\nVoice Identification System", 
                              font=("Arial", 18, "bold"), style="Title.TLabel")
        exit_label.pack(expand=True, fill=tk.BOTH)
        
        # Fade out animation
        self.root.update()
        original_bg = self.root.cget("bg")
        
        def fade_out():
            for i in range(10):
                shade = 240 - i*10
                self.root.config(bg=f"#{shade:02x}{shade:02x}{shade:02x}")
                exit_label.config(foreground=f"#{0+i*25:02x}{0+i*25:02x}{0+i*25:02x}")
                self.root.update()
                time.sleep(0.05)
            
            # Destroy after animation
            self.stop_identification()
            self.root.quit()
            self.root.destroy()
        
        # Run fade animation in a thread to avoid hanging
        threading.Thread(target=fade_out, daemon=True).start()
        
        # Set a fallback destroy in case animation fails
        self.root.after(1000, self.root.destroy)
        
    def run(self):
        """Run the GUI application."""
        # Initial animations
        self.root.mainloop()


# Example usage
if __name__ == "__main__":
    # Initialize with directory containing voice samples
    identifier = VoiceIdentifierGUI(
        samples_dir=r"C:/Users/ITD/Desktop/GUI/Data", 
        threshold=0.45  # Using a lower threshold based on previous results
    )
    
    # Start the GUI
    identifier.run()