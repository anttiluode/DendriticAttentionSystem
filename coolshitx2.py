"""
Improved Adaptive Dendritic Attention System with Speaker Neurons

Changes in this version:
1. Fixed numerical overflow issues in the attention system
2. Added proper frequency exploration with three strategies:
   - Sinusoidal exploration to encourage movement across the frequency range
   - Better scaling of adjustments to prevent getting stuck at min/max
   - Improved learning mechanism with decay and noise
3. Added safety checks for NaN values
4. Improved visualization with frequency history tracking
"""

# Import block from the original code remains the same
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
import numpy as np
import pyaudio
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D  # noqa
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import logging
from PIL import Image, ImageTk

# Configure logging
logging.basicConfig(
    filename='adaptive_dendritic_system.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.DEBUG
)
logger = logging.getLogger(__name__)

# ---------------------------
# Improved DendriticAttention Class
# ---------------------------
class DendriticAttention:
    """
    Implements an adaptive attention system using dendritic growth principles.
    Observes input patterns and adjusts its focus based on pattern recognition.
    """
    def __init__(self, input_size=(64, 64), n_dendrites=1000):
        self.input_size = input_size
        self.n_dendrites = n_dendrites
        
        # Initialize dendrite positions and growth directions
        self.positions = np.random.rand(n_dendrites, 2) * np.array([input_size[0], input_size[1]])
        self.directions = self.normalize(np.random.randn(n_dendrites, 2))
        self.strengths = np.ones(n_dendrites) * 0.5
        
        # Attention field - represents where system is focusing
        self.attention_field = np.ones(input_size)
        
        # Pattern memory - what the system expects to see
        self.expected_pattern = None
        self.memory_strength = 0.0
        
        # Response parameters - how it affects speaker neurons
        self.response_vectors = np.random.randn(4, 5) * 0.1  # Initialize with small values
        self.attention_width = 0.5  # Initial width (0-1 scale)
        self.stability_measure = 0.5  # How stable the system thinks patterns are
        
        # Activity history
        self.activity_history = []
        self.match_history = []
        
        # Exploration parameters
        self.exploration_rate = 1.0  # Start with high exploration
        self.exploration_decay = 0.9999  # Slow decay
        self.frequency_memory = {}  # Track which frequencies we've explored
        
        # Start time for sinusoidal exploration
        self.start_time = time.time()
        
    def normalize(self, vectors):
        """Normalize vectors to unit length, with NaN protection"""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.maximum(norms, 1e-8)
        return vectors / norms
        
    def compute_match(self, input_data, expected):
        """Calculate how well the input matches expectations"""
        if input_data.shape != expected.shape:
            return 0.0
        
        # Normalized correlation coefficient
        input_flat = input_data.flatten()
        expected_flat = expected.flatten()
        
        # Remove mean
        input_centered = input_flat - np.mean(input_flat)
        expected_centered = expected_flat - np.mean(expected_flat)
        
        # Calculate correlation
        num = np.sum(input_centered * expected_centered)
        den = np.sqrt(np.sum(input_centered**2) * np.sum(expected_centered**2))
        
        # Avoid division by zero
        if den < 1e-8:
            return 0.0
            
        correlation = num / den
        
        # Ensure result is in range [-1, 1], then scale to [0, 1]
        correlation = np.clip(correlation, -1.0, 1.0)
        scaled = (correlation + 1) / 2
        return scaled
        
    def update(self, input_data):
        """Process input and update attention field"""
        # Resize input if needed
        if input_data.shape != self.input_size:
            # Simple resize by averaging blocks
            h_ratio = input_data.shape[0] / self.input_size[0]
            w_ratio = input_data.shape[1] / self.input_size[1]
            resized = np.zeros(self.input_size)
            for i in range(self.input_size[0]):
                for j in range(self.input_size[1]):
                    h_start, h_end = int(i * h_ratio), int((i + 1) * h_ratio)
                    w_start, w_end = int(j * w_ratio), int((j + 1) * w_ratio)
                    resized[i, j] = np.mean(input_data[h_start:max(h_start+1, h_end), 
                                                      w_start:max(w_start+1, w_end)])
            input_data = resized
            
        # Compute pattern match between input and expected pattern
        if self.expected_pattern is not None:
            match_score = self.compute_match(input_data, self.expected_pattern)
            self.match_history.append(match_score)
            if len(self.match_history) > 50:
                self.match_history.pop(0)
        else:
            # No expectation yet, just store the pattern
            self.expected_pattern = input_data.copy()
            self.memory_strength = 0.1
            match_score = 1.0
            self.match_history = [1.0]
            
        # Calculate stability based on match history
        if len(self.match_history) > 5:
            match_variance = np.var(self.match_history[-5:])
            self.stability_measure = 1.0 - min(1.0, match_variance * 10)
        
        # Update attention width based on match score (iris effect)
        target_width = 0.3 if match_score > 0.7 else 0.8
        self.attention_width = 0.95 * self.attention_width + 0.05 * target_width
            
        # Create new attention field
        self.dilate_attention()
            
        # Grow dendrites toward areas with high input activity
        self.grow_dendrites(input_data)
        
        # Extract features from attention pattern and input
        self.extract_features(input_data)
            
        # Update expected pattern with slow learning
        if self.expected_pattern is not None:
            self.expected_pattern = (0.9 * self.expected_pattern + 
                                    0.1 * input_data * self.attention_field)
            
        # Decay exploration rate
        self.exploration_rate *= self.exploration_decay
        self.exploration_rate = max(0.1, self.exploration_rate)
        
        return self.attention_field
        
    def dilate_attention(self):
        """Update attention field based on attention width (iris effect)"""
        # Create a Gaussian attention field centered in the image
        x, y = np.meshgrid(
            np.linspace(-1, 1, self.input_size[1]),
            np.linspace(-1, 1, self.input_size[0])
        )
        
        # Calculate radial distance from center
        distance = np.sqrt(x**2 + y**2)
        
        # Create Gaussian with width determined by attention_width (larger = wider)
        sigma = 0.2 + self.attention_width * 1.0  # Map attention_width to reasonable sigma
        self.attention_field = np.exp(-(distance**2 / (2.0 * sigma**2)))
        
    def grow_dendrites(self, input_data):
        """Grow dendrites based on input activity"""
        # Update dendrite positions based on strengths and directions
        for i in range(self.n_dendrites):
            x, y = self.positions[i].astype(int) % self.input_size
            try:
                activity = input_data[x, y]
            except IndexError:
                # Handle boundary cases
                x = min(x, self.input_size[0] - 1)
                y = min(y, self.input_size[1] - 1)
                activity = input_data[x, y]
                
            # Update strengths based on activity
            self.strengths[i] = 0.95 * self.strengths[i] + 0.05 * activity
            
            # Only grow the stronger dendrites
            if self.strengths[i] > 0.3:
                # Calculate gradient of input (first approximation)
                grad_x, grad_y = 0, 0
                if x > 0 and x < self.input_size[0] - 1:
                    grad_x = input_data[x+1, y] - input_data[x-1, y]
                if y > 0 and y < self.input_size[1] - 1:
                    grad_y = input_data[x, y+1] - input_data[x, y-1]
                
                # Update direction with gradient influence
                if abs(grad_x) > 0.01 or abs(grad_y) > 0.01:
                    gradient = np.array([grad_x, grad_y])
                    gradient_norm = np.linalg.norm(gradient)
                    if gradient_norm > 0:
                        gradient = gradient / gradient_norm
                        self.directions[i] = 0.8 * self.directions[i] + 0.2 * gradient
                        self.directions[i] = self.directions[i] / (np.linalg.norm(self.directions[i]) + 1e-8)
                
                # Move dendrite tip in the current direction
                growth_rate = self.strengths[i] * 0.1  # Stronger dendrites grow faster
                self.positions[i] += self.directions[i] * growth_rate
                
                # Wrap around boundaries
                self.positions[i] = self.positions[i] % np.array(self.input_size)
    
    def extract_features(self, input_data):
        """Extract features from current state to guide neuron tuning"""
        # Measure total activity
        total_activity = np.mean(input_data * self.attention_field)
        
        # Measure activity variance in different regions
        h, w = self.input_size
        top_left = np.mean(input_data[:h//2, :w//2])
        top_right = np.mean(input_data[:h//2, w//2:])
        bottom_left = np.mean(input_data[h//2:, :w//2])
        bottom_right = np.mean(input_data[h//2:, w//2:])
        
        # Store key features
        self.activity_vector = np.array([
            total_activity,
            top_left - bottom_right,  # Diagonal balance 1
            top_right - bottom_left,  # Diagonal balance 2
            self.stability_measure    # Overall stability
        ])
        
        # Clip values to prevent extreme adjustments
        self.activity_vector = np.clip(self.activity_vector, -1.0, 1.0)
        
        self.activity_history.append(total_activity)
        if len(self.activity_history) > 100:
            self.activity_history.pop(0)
    
    def get_frequency_adjustments(self, num_neurons=5, current_freqs=None):
        """
        Calculate frequency adjustments for speaker neurons with exploration
        
        Args:
            num_neurons: Number of speaker neurons to adjust
            current_freqs: Current frequencies for each neuron (for exploration logic)
            
        Returns:
            Array of frequency adjustments for each neuron
        """
        # Ensure we have the right number of outputs
        if self.response_vectors.shape[1] != num_neurons:
            self.response_vectors = np.random.randn(4, num_neurons) * 0.1
            
        # Calculate base adjustments from activity vector and response matrix
        # But first clip the values to prevent overflow
        clipped_activity = np.clip(self.activity_vector, -1.0, 1.0)
        raw_adjustments = np.dot(clipped_activity, self.response_vectors)
        
        # Scale to reasonable values, using clipped attention width
        attention_width = np.clip(self.attention_width, 0.1, 1.0)
        scaled_adjustments = raw_adjustments * attention_width * 0.2
        
        # EXPLORATION STRATEGY 1: Sinusoidal exploration
        # Add time-based sinusoidal exploration to encourage full range
        elapsed_time = time.time() - self.start_time
        
        # Different frequencies for different neurons
        for i in range(num_neurons):
            # Period increases with neuron index to create varied patterns
            period = 5.0 + i * 2.0  # seconds per cycle
            # Amplitude decreases over time as exploration rate decays
            amplitude = 0.3 * self.exploration_rate
            
            # Add sinusoidal component
            scaled_adjustments[i] += amplitude * np.sin(elapsed_time * (2 * np.pi / period))
        
        # EXPLORATION STRATEGY 2: Target frequency regions
        # Push frequencies toward unexplored regions
        if current_freqs is not None:
            for i, freq in enumerate(current_freqs):
                # Quantize the frequency space into 10 bins
                freq_bin = int(freq / 200)  # 0-10 bins for 0-2000 Hz
                
                # Initialize frequency memory if needed
                if i not in self.frequency_memory:
                    self.frequency_memory[i] = np.ones(10) * 0.1  # Initial low count for all bins
                
                # Increment the count for the current bin (memory of where we've been)
                self.frequency_memory[i][min(freq_bin, 9)] += 0.1
                
                # Calculate pull toward unexplored regions
                # Invert and normalize the memory to create an attraction field
                attraction = 1.0 / (self.frequency_memory[i] + 0.1)
                attraction /= np.sum(attraction)
                
                # Find the bin with strongest attraction (least explored)
                target_bin = np.argmax(attraction)
                
                # Calculate adjustment to move toward that bin
                target_freq = (target_bin + 0.5) * 200  # Center of the bin
                freq_diff = target_freq - freq
                
                # Scale adjustment based on distance and exploration rate
                bin_adjustment = np.sign(freq_diff) * min(abs(freq_diff) * 0.01, 0.2) * self.exploration_rate
                
                # Add to adjustments
                scaled_adjustments[i] += bin_adjustment
        
        # EXPLORATION STRATEGY 3: Noise injection with adaptive scaling
        # Add noise scaled by exploration rate and stability
        noise_scale = 0.1 * self.exploration_rate * (1.0 - self.stability_measure)
        noise = np.random.randn(num_neurons) * noise_scale
        scaled_adjustments += noise
        
        # Final safety clip to prevent extreme changes
        return np.clip(scaled_adjustments, -0.5, 0.5)
        
    def learn_response(self, adjustments, match_delta):
        """Learn which adjustments lead to better pattern matching"""
        # Only proceed if we have valid data
        if np.isnan(match_delta) or np.any(np.isnan(adjustments)):
            return
            
        # Learning rate decays with stability
        learn_rate = 0.01 * (1.0 - self.stability_measure * 0.5)
        
        # If match score improved, reinforce those adjustments
        if match_delta > 0:
            # For each feature
            for i in range(len(self.activity_vector)):
                # Skip if activity component is NaN
                if np.isnan(self.activity_vector[i]):
                    continue
                    
                # Update response vector (activity to frequency adjustment mapping)
                # Clip the activity value to prevent overflow
                activity = np.clip(self.activity_vector[i], -1.0, 1.0)
                
                for j in range(len(adjustments)):
                    # Skip if adjustment is NaN
                    if np.isnan(adjustments[j]):
                        continue
                        
                    # Update rule with sign of activity and adjustment
                    self.response_vectors[i, j] += learn_rate * activity * adjustments[j]
        else:
            # If match got worse, do the opposite next time
            for i in range(len(self.activity_vector)):
                # Skip if activity component is NaN
                if np.isnan(self.activity_vector[i]):
                    continue
                    
                # Clip activity value
                activity = np.clip(self.activity_vector[i], -1.0, 1.0)
                
                for j in range(len(adjustments)):
                    # Skip if adjustment is NaN
                    if np.isnan(adjustments[j]):
                        continue
                        
                    # Reverse adjustment with lower learning rate
                    self.response_vectors[i, j] -= learn_rate * activity * adjustments[j] * 0.5
        
        # Apply regularization to prevent extreme values
        self.response_vectors = np.clip(self.response_vectors, -1.0, 1.0)
    
    def get_visualization(self):
        """Generate visualization of the attention field and dendrites"""
        vis_img = np.zeros((*self.input_size, 3))
        
        # Add attention field (blue channel)
        vis_img[:, :, 2] = self.attention_field
        
        # Add active dendrites (green channel)
        for i in range(self.n_dendrites):
            if self.strengths[i] > 0.2:
                x, y = self.positions[i].astype(int) % self.input_size
                try:
                    # Brightness based on strength
                    vis_img[x, y, 1] = min(1.0, vis_img[x, y, 1] + self.strengths[i])
                except IndexError:
                    pass
                    
        # Add expected pattern (red channel)
        if self.expected_pattern is not None:
            vis_img[:, :, 0] = self.expected_pattern * 0.7
            
        return (vis_img * 255).astype(np.uint8)


# ---------------------------
# PhysicalSpeakerNeuron Class
# ---------------------------
class PhysicalSpeakerNeuron:
    """
    Handles Audio Output and Waveform Generation.
    Supports multiple waveform types: Sine, Square, Saw, Triangle, and Noise.
    """
    def __init__(self, sample_rate=44100, buffer_size=4096, waveform="Sine",
                 base_freq=1000, amplitude=0.3, input_device=None, output_device=None):
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.waveform = waveform  # "Sine", "Square", "Saw", "Triangle", or "Noise"
        self.base_freq = base_freq
        self.amplitude = amplitude
        self.phase = 0.0
        self.audio = pyaudio.PyAudio()
        self.input_device = input_device
        self.output_device = output_device
        # Track frequency history
        self.frequency_history = []
        self.setup_output()
    
    def setup_output(self):
        try:
            self.output_stream = self.audio.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=self.sample_rate,
                output=True,
                output_device_index=self.output_device,
                frames_per_buffer=self.buffer_size
            )
            logger.info(f"Opened output device {self.output_device}")
        except Exception as e:
            logger.error(f"Failed to open output device: {e}")
            raise e
    
    def generate_wave(self):
        # Track frequency
        self.frequency_history.append(self.base_freq)
        if len(self.frequency_history) > 100:
            self.frequency_history.pop(0)
            
        # Generate time vector for one buffer
        t = np.linspace(0, self.buffer_size/self.sample_rate, self.buffer_size, endpoint=False)
        # For most waveforms, we use phase and frequency. For Noise, we ignore phase.
        if self.waveform == "Sine":
            wave = np.sin(2 * np.pi * self.base_freq * t + self.phase)
        elif self.waveform == "Square":
            wave = np.sign(np.sin(2 * np.pi * self.base_freq * t + self.phase))
        elif self.waveform == "Saw":
            frac = (2 * np.pi * self.base_freq * t + self.phase) / (2 * np.pi)
            frac = frac % 1.0
            wave = 2.0 * frac - 1.0
        elif self.waveform == "Triangle":
            frac = (2 * np.pi * self.base_freq * t + self.phase) / (2 * np.pi)
            frac = frac % 1.0
            wave = 2.0 * np.abs(2.0 * frac - 1.0) - 1.0
        elif self.waveform == "Noise":
            # White noise between -1 and 1.
            wave = np.random.uniform(low=-1, high=1, size=self.buffer_size)
        else:
            wave = np.sin(2 * np.pi * self.base_freq * t + self.phase)
        
        # For non-noise waveforms, update phase:
        if self.waveform != "Noise":
            phase_inc = 2 * np.pi * self.base_freq * (self.buffer_size / self.sample_rate)
            self.phase = (self.phase + phase_inc) % (2 * np.pi)
        
        wave *= self.amplitude
        return wave.astype(np.float32)
    
    def forward(self):
        wave = self.generate_wave()
        self.output_stream.write(wave.tobytes())
        return wave
    
    def cleanup(self):
        try:
            self.output_stream.stop_stream()
            self.output_stream.close()
            self.audio.terminate()
            logger.info("Cleaned up neuron audio")
        except Exception as e:
            logger.error(f"Cleanup error: {e}")


# ---------------------------
# AdaptiveDendriticNeuroUI Class
# ---------------------------
class AdaptiveDendriticNeuroUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Adaptive Dendritic System with Speaker Neurons")
        self.sample_rate = 44100
        self.buffer_size = 4096
        
        self.neurons = []
        self.num_neurons = 5  # Default number
        self.wave_history = []  # For storing combined wave buffers
        
        # Setup PyAudio and default output device
        self.p = pyaudio.PyAudio()
        self.default_output = None
        for i in range(self.p.get_device_count()):
            try:
                dev_info = self.p.get_device_info_by_index(i)
                if dev_info['maxOutputChannels'] > 0:
                    self.default_output = i
                    break
            except Exception as e:
                logger.error(f"Error getting device info: {e}")
        
        # Initialize the dendritic attention system
        self.attention_system = DendriticAttention(input_size=(64, 64), n_dendrites=1000)
        
        # Flag to control system state
        self.running = False
        self.phase_image = None
        
        # Variables to track adjustments
        self.last_adjustments = np.zeros(self.num_neurons)
        self.last_match_score = 0.5
        self.adjustment_counter = 0
        
        # Setup UI layout
        self.setup_layout()
        self.create_neuron_controls()

    def setup_layout(self):
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=3)  # Plot area
        self.root.columnconfigure(1, weight=1)  # Control area
        
        # Left frame: Plot area with tabs
        self.left_frame = ttk.Frame(self.root)
        self.left_frame.grid(row=0, column=0, sticky="nsew")
        
        # Create a notebook (tabbed interface)
        self.notebook = ttk.Notebook(self.left_frame)
        self.notebook.pack(fill="both", expand=True)
        
        # Tab 1: 3D Phase Space
        self.phase_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.phase_tab, text="3D Phase Space")
        
        self.fig_3d = plt.Figure(figsize=(8, 6))
        self.ax3d = self.fig_3d.add_subplot(111, projection='3d')
        self.ax3d.set_title("Combined 3D Phase Space")
        self.canvas_3d = FigureCanvasTkAgg(self.fig_3d, self.phase_tab)
        self.canvas_3d.get_tk_widget().pack(fill="both", expand=True)
        
        # Tab 2: Attention Visualization
        self.attn_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.attn_tab, text="Attention Field")
        
        self.attn_canvas = tk.Canvas(self.attn_tab, bg="black")
        self.attn_canvas.pack(fill="both", expand=True)
        
        # Tab 3: Activity Metrics
        self.activity_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.activity_tab, text="Activity Metrics")
        
        self.fig_activity = plt.Figure(figsize=(8, 6))
        self.ax_activity = self.fig_activity.add_subplot(111)
        self.ax_activity.set_title("System Activity")
        self.canvas_activity = FigureCanvasTkAgg(self.fig_activity, self.activity_tab)
        self.canvas_activity.get_tk_widget().pack(fill="both", expand=True)
        
        # Tab 4: Frequency History
        self.freq_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.freq_tab, text="Frequency History")
        
        self.fig_freq = plt.Figure(figsize=(8, 6))
        self.ax_freq = self.fig_freq.add_subplot(111)
        self.ax_freq.set_title("Neuron Frequency History")
        self.canvas_freq = FigureCanvasTkAgg(self.fig_freq, self.freq_tab)
        self.canvas_freq.get_tk_widget().pack(fill="both", expand=True)
        
        # Right frame: Control panel with scrollbar
        self.control_frame = tk.Frame(self.root)
        self.control_frame.grid(row=0, column=1, sticky="nsew")
        self.control_canvas = tk.Canvas(self.control_frame)
        self.control_scrollbar = ttk.Scrollbar(self.control_frame, orient="vertical", command=self.control_canvas.yview)
        self.control_scrollable_frame = tk.Frame(self.control_canvas)
        self.control_scrollable_frame.bind(
            "<Configure>",
            lambda e: self.control_canvas.configure(scrollregion=self.control_canvas.bbox("all"))
        )
        self.control_canvas.create_window((0, 0), window=self.control_scrollable_frame, anchor="nw")
        self.control_canvas.configure(yscrollcommand=self.control_scrollbar.set)
        self.control_canvas.pack(side="left", fill="both", expand=True)
        self.control_scrollbar.pack(side="right", fill="y")
        
        # System state indicators
        state_frame = tk.LabelFrame(self.control_scrollable_frame, text="System State", padx=5, pady=5)
        state_frame.pack(fill="x", padx=5, pady=5)
        
        # Attention width indicator
        tk.Label(state_frame, text="Attention Width:").grid(row=0, column=0, sticky="w", padx=2, pady=2)
        self.attn_width_var = tk.DoubleVar(value=0.5)
        ttk.Progressbar(state_frame, variable=self.attn_width_var, maximum=1.0, length=100).grid(row=0, column=1, sticky="ew", padx=2, pady=2)
        
        # Stability indicator
        tk.Label(state_frame, text="Pattern Stability:").grid(row=1, column=0, sticky="w", padx=2, pady=2)
        self.stability_var = tk.DoubleVar(value=0.5)
        ttk.Progressbar(state_frame, variable=self.stability_var, maximum=1.0, length=100).grid(row=1, column=1, sticky="ew", padx=2, pady=2)
        
        # Match score indicator
        tk.Label(state_frame, text="Match Score:").grid(row=2, column=0, sticky="w", padx=2, pady=2)
        self.match_var = tk.DoubleVar(value=0.5)
        ttk.Progressbar(state_frame, variable=self.match_var, maximum=1.0, length=100).grid(row=2, column=1, sticky="ew", padx=2, pady=2)
        
        # Exploration rate indicator
        tk.Label(state_frame, text="Exploration Rate:").grid(row=3, column=0, sticky="w", padx=2, pady=2)
        self.exploration_var = tk.DoubleVar(value=1.0)
        ttk.Progressbar(state_frame, variable=self.exploration_var, maximum=1.0, length=100).grid(row=3, column=1, sticky="ew", padx=2, pady=2)
        
        # Frame for individual neuron controls
        self.neuron_controls_frame = tk.LabelFrame(self.control_scrollable_frame, text="Speaker Neurons")
        self.neuron_controls_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # System control buttons
        controls_frame = tk.Frame(self.control_scrollable_frame)
        controls_frame.pack(fill="x", padx=5, pady=5)
        
        # Start/Stop button
        self.start_stop_button = ttk.Button(controls_frame, text="Start", command=self.toggle_running)
        self.start_stop_button.pack(side="left", fill="x", expand=True, padx=5, pady=5)
        
        # Reset button
        self.reset_button = ttk.Button(controls_frame, text="Reset System", command=self.reset_system)
        self.reset_button.pack(side="left", fill="x", expand=True, padx=5, pady=5)
    
    def create_neuron_controls(self):
        # Remove old controls if any
        for widget in self.neuron_controls_frame.winfo_children():
            widget.destroy()
        self.neurons = []
        self.neuron_freq_vars = []
        self.neuron_wave_vars = []
        
        # Use default output device from earlier
        dev_index = self.default_output
        
        # Create control frames
        control_frames = []
        for i in range(self.num_neurons):
            frame = ttk.LabelFrame(self.neuron_controls_frame, text=f"Neuron {i+1}")
            frame.pack(fill="x", pady=5, padx=5)
            control_frames.append(frame)
            
            # Waveform selection combobox
            wave_frame = ttk.Frame(frame)
            wave_frame.pack(fill="x", pady=2)
            
            ttk.Label(wave_frame, text="Waveform:").pack(side="left", padx=2)
            wave_var = tk.StringVar(value="Sine")
            self.neuron_wave_vars.append(wave_var)
            wave_combo = ttk.Combobox(wave_frame, textvariable=wave_var,
                                      values=["Sine", "Square", "Saw", "Triangle", "Noise"],
                                      state="readonly", width=10)
            wave_combo.pack(side="left", padx=2)
            
            # Frequency slider frame
            freq_frame = ttk.Frame(frame)
            freq_frame.pack(fill="x", pady=2)
            
            ttk.Label(freq_frame, text="Frequency (Hz):").pack(side="left", padx=2)
            # Distribute initial frequencies across the spectrum
            init_freq = 200 + (i * 350)  # More evenly distributed initial frequencies
            freq_var = tk.DoubleVar(value=init_freq)
            self.neuron_freq_vars.append(freq_var)
            freq_slider = ttk.Scale(freq_frame, from_=20, to=2000, orient="horizontal", 
                                   variable=freq_var, length=200)
            freq_slider.pack(side="left", fill="x", expand=True, padx=2)
            
            # Frequency display label
            freq_label = ttk.Label(freq_frame, text=f"{int(freq_var.get())} Hz")
            freq_label.pack(side="left", padx=2)
            
            # Update label when frequency changes
            def update_label(var, label=freq_label):
                label.config(text=f"{int(var.get())} Hz")
            freq_var.trace_add("write", lambda n, i, m, var=freq_var, label=freq_label: update_label(var, label))
            
            # Create the neuron instance
            neuron = PhysicalSpeakerNeuron(
                sample_rate=self.sample_rate,
                buffer_size=self.buffer_size,
                waveform=wave_var.get(),
                base_freq=freq_var.get(),
                amplitude=0.3 / self.num_neurons,  # Scale amplitude to avoid clipping
                output_device=dev_index
            )
            self.neurons.append(neuron)
            
            # Update neuron when waveform changes
            def wave_changed(var, idx, mode, n=i, wv=wave_var):
                self.neurons[n].waveform = wv.get()
            wave_var.trace_add("write", wave_changed)
            
            # Update neuron when frequency changes
            def freq_changed(var, idx, mode, n=i, fv=freq_var):
                self.neurons[n].base_freq = fv.get()
            freq_var.trace_add("write", freq_changed)
    
    def update_neuron_frequency(self, idx, freq, adjust_slider=True):
        """Update neuron frequency with bounds checking"""
        try:
            # Ensure frequency is within reasonable bounds
            freq = max(20, min(2000, freq))
            
            # Update the neuron object
            self.neurons[idx].base_freq = freq
            
            # If requested, update the slider too
            if adjust_slider:
                self.neuron_freq_vars[idx].set(freq)
        except Exception as e:
            logger.error(f"Error updating neuron frequency: {e}")
    
    def toggle_running(self):
        self.running = not self.running
        if self.running:
            self.start_stop_button.config(text="Stop")
            # Start processing in separate thread
            self.processing_thread = threading.Thread(target=self.processing_loop, daemon=True)
            self.processing_thread.start()
        else:
            self.start_stop_button.config(text="Start")
    
    def reset_system(self):
        """Reset all system components"""
        try:
            # Recreate the attention system
            self.attention_system = DendriticAttention(input_size=(64, 64), n_dendrites=1000)
            
            # Reset wave history
            self.wave_history = []
            
            # Reset metrics
            self.last_match_score = 0.5
            self.last_adjustments = np.zeros(self.num_neurons)
            self.adjustment_counter = 0
            
            # Reset UI indicators
            self.attn_width_var.set(0.5)
            self.stability_var.set(0.5)
            self.match_var.set(0.5)
            self.exploration_var.set(1.0)
            
            # Reset neuron frequency history
            for neuron in self.neurons:
                neuron.frequency_history = []
            
            # Update visualizations
            self.update_visualization()
            messagebox.showinfo("Reset", "System has been reset")
        except Exception as e:
            logger.error(f"Error resetting system: {e}")
            messagebox.showerror("Error", f"Failed to reset system: {str(e)}")
    
    def capture_phase_space(self, buffer_data=None):
        """Capture an image of the current 3D phase space plot for the attention system to observe"""
        # If no buffer data is provided, use the current wave history
        if buffer_data is None and len(self.wave_history) >= 3:
            w1 = self.wave_history[-3]
            w2 = self.wave_history[-2]
            w3 = self.wave_history[-1]
        else:
            # Not enough data yet
            return None
            
        # Downsample for efficiency by taking every 8th point
        stride = 8
        w1 = w1[::stride]
        w2 = w2[::stride]
        w3 = w3[::stride]
        
        # Clear the 3D plot
        self.ax3d.clear()
        
        # Create a colormap based on energy
        energy = np.sqrt(w1**2 + w2**2 + w3**2)
        colors = plt.cm.viridis(energy / (energy.max() + 1e-8))  # Add epsilon to prevent division by zero
        
        # Plot the 3D phase space
        self.ax3d.scatter(w1, w2, w3, c=colors, s=5, alpha=0.7)
        
        # Set labels and limits
        self.ax3d.set_title("3D Phase Space")
        self.ax3d.set_xlabel("Wave(n-2)")
        self.ax3d.set_ylabel("Wave(n-1)")
        self.ax3d.set_zlabel("Wave(n)")
        
        # Set consistent limits for better visualization
        limit = max(np.max(np.abs(w1)), np.max(np.abs(w2)), np.max(np.abs(w3)), 0.1)
        self.ax3d.set_xlim([-limit, limit])
        self.ax3d.set_ylim([-limit, limit])
        self.ax3d.set_zlim([-limit, limit])
        
        # Draw the figure
        self.canvas_3d.draw()
        
        # Capture the plot as an image
        width, height = self.fig_3d.get_size_inches() * self.fig_3d.dpi
        width, height = int(width), int(height)
        
        try:
            # Get the renderer
            renderer = self.fig_3d.canvas.renderer
            # Render the canvas to an array
            buf = renderer.buffer_rgba()
            # Convert to image array
            phase_img = np.frombuffer(buf, dtype=np.uint8).reshape(height, width, 4)
            # Convert RGBA to RGB
            phase_img = phase_img[:, :, :3]
            
            # Store the image
            self.phase_image = phase_img
            return phase_img
        except Exception as e:
            logger.error(f"Failed to capture phase space: {e}")
            return None
    
    def processing_loop(self):
        """Main processing loop that coordinates speaker neurons and dendritic attention"""
        last_adaptation_time = time.time()
        adaptation_interval = 0.2  # seconds between adaptations
        
        while self.running:
            try:
                # 1. Generate waveforms from each neuron
                combined_wave = np.zeros(self.buffer_size, dtype=np.float32)
                for neuron in self.neurons:
                    wave = neuron.generate_wave()
                    combined_wave += wave
                
                # Normalize to prevent clipping
                max_val = np.max(np.abs(combined_wave))
                if max_val > 1.0:
                    combined_wave /= max_val
                
                # 2. Play the audio if we have an output device
                if self.neurons and self.neurons[0].output_stream:
                    self.neurons[0].output_stream.write(combined_wave.tobytes())
                
                # 3. Store for visualization
                self.wave_history.append(combined_wave)
                if len(self.wave_history) > 50:
                    self.wave_history.pop(0)
                
                # 4. Update phase space visualization
                self.root.after(0, self.update_visualization)
                
                # 5. Capture phase space for attention system
                phase_img = self.capture_phase_space()
                
                # 6. Feed to dendritic attention system and get adjustments
                current_time = time.time()
                if phase_img is not None and (current_time - last_adaptation_time) >= adaptation_interval:
                    # Get current frequencies for all neurons
                    current_freqs = [neuron.base_freq for neuron in self.neurons]
                    
                    # Process the image through attention system
                    attention_field = self.attention_system.update(phase_img)
                    
                    # Calculate match score
                    current_match = np.mean(self.attention_system.match_history[-5:]) if len(self.attention_system.match_history) >= 5 else 0.5
                    match_delta = current_match - self.last_match_score
                    
                    # Get frequency adjustments from attention system
                    adjustments = self.attention_system.get_frequency_adjustments(
                        num_neurons=self.num_neurons,
                        current_freqs=current_freqs
                    )
                    
                    # Apply adjustments to neurons with more controlled scaling
                    scale_factor = 50  # Increased from 20 to 50 Hz per adjustment unit
                    for i in range(min(len(adjustments), len(self.neurons))):
                        # Apply adjustment to current frequency
                        curr_freq = self.neurons[i].base_freq
                        new_freq = curr_freq + adjustments[i] * scale_factor
                        
                        # Ensure new frequency is within range and not NaN
                        if not np.isnan(new_freq):
                            # Update neuron and UI
                            self.update_neuron_frequency(i, new_freq)
                    
                    # Learn from the results if we have valid data
                    if not np.isnan(match_delta) and not np.any(np.isnan(self.last_adjustments)):
                        self.attention_system.learn_response(self.last_adjustments, match_delta)
                    
                    # Update state for next iteration
                    self.last_adjustments = adjustments
                    self.last_match_score = current_match
                    
                    # Update UI indicators
                    self.root.after(0, self.update_indicators)
                    
                    # Reset timer
                    last_adaptation_time = current_time
                    
                    # Increment adjustment counter
                    self.adjustment_counter += 1
                
                # Short sleep to keep CPU usage reasonable
                time.sleep(0.02)
                
            except Exception as e:
                logger.error(f"Processing error: {e}")
                time.sleep(0.1)
    
    def update_indicators(self):
        """Update UI indicators with current system state"""
        try:
            # Update attention width indicator
            self.attn_width_var.set(self.attention_system.attention_width)
            
            # Update stability indicator
            self.stability_var.set(self.attention_system.stability_measure)
            
            # Update match score indicator
            current_match = np.mean(self.attention_system.match_history[-5:]) if len(self.attention_system.match_history) >= 5 else 0.5
            self.match_var.set(current_match)
            
            # Update exploration rate
            self.exploration_var.set(self.attention_system.exploration_rate)
            
            # Update attention visualization
            attention_img = self.attention_system.get_visualization()
            if attention_img is not None:
                pil_img = Image.fromarray(attention_img)
                # Resize to fit canvas
                canvas_width = self.attn_canvas.winfo_width()
                canvas_height = self.attn_canvas.winfo_height()
                if canvas_width > 100 and canvas_height > 100:  # Only if canvas has been drawn
                    pil_img = pil_img.resize((canvas_width, canvas_height), Image.LANCZOS)
                    # Convert to PhotoImage
                    photo = ImageTk.PhotoImage(pil_img)
                    # Update canvas
                    self.attn_canvas.delete("all")  # Clear canvas first
                    self.attn_canvas.create_image(0, 0, anchor=tk.NW, image=photo)
                    self.attn_canvas.image = photo  # Keep reference
            
            # Update activity plot
            if len(self.attention_system.activity_history) > 1:
                self.ax_activity.clear()
                self.ax_activity.set_title("System Activity")
                
                # Plot match history
                if len(self.attention_system.match_history) > 1:
                    x = np.arange(len(self.attention_system.match_history))
                    self.ax_activity.plot(x, self.attention_system.match_history, 'b-', label='Pattern Match')
                
                # Plot activity history 
                if len(self.attention_system.activity_history) > 1:
                    x = np.arange(len(self.attention_system.activity_history))
                    self.ax_activity.plot(x, self.attention_system.activity_history, 'g-', label='Activity')
                    
                # Plot exploration rate
                x = np.arange(len(self.attention_system.match_history))
                exploration_data = [self.attention_system.exploration_rate] * len(x)
                self.ax_activity.plot(x, exploration_data, 'r--', label='Exploration Rate')
                
                # Add adjustment markers
                if self.adjustment_counter > 0:
                    self.ax_activity.axvline(x=self.adjustment_counter, color='r', linestyle='--', alpha=0.3)
                    
                self.ax_activity.legend()
                self.ax_activity.set_ylim([0, 1])
                self.canvas_activity.draw()
                
            # Update frequency history plot
            self.ax_freq.clear()
            self.ax_freq.set_title("Neuron Frequency History")
            self.ax_freq.set_ylim([0, 2100])  # Just above max frequency
            self.ax_freq.set_xlabel("Time")
            self.ax_freq.set_ylabel("Frequency (Hz)")
            
            # Add individual frequency histories
            for i, neuron in enumerate(self.neurons):
                if len(neuron.frequency_history) > 0:
                    x = np.arange(len(neuron.frequency_history))
                    self.ax_freq.plot(x, neuron.frequency_history, 
                                     label=f"Neuron {i+1} ({neuron.waveform})")
            
            self.ax_freq.legend()
            self.canvas_freq.draw()
                
        except Exception as e:
            logger.error(f"Error updating indicators: {e}")
    
    def update_visualization(self):
        """Update 3D phase space visualization"""
        if len(self.wave_history) >= 3:
            self.capture_phase_space()
    
    def cleanup(self):
        """Cleanup resources when closing"""
        self.running = False
        for neuron in self.neurons:
            neuron.cleanup()
        self.p.terminate()
    
    def quit(self):
        """Handle application shutdown"""
        self.cleanup()
        self.root.quit()
        self.root.destroy()

def main():
    # Set up exception handling
    def handle_exception(exc_type, exc_value, exc_traceback):
        logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
        messagebox.showerror("Error", str(exc_value))
    
    import sys
    sys.excepthook = handle_exception
    
    # Create main window
    root = tk.Tk()
    root.geometry("1200x800")
    root.title("Adaptive Dendritic System with Speaker Neurons")
    
    # Create and run application
    app = AdaptiveDendriticNeuroUI(root)
    
    # Set up clean shutdown
    root.protocol("WM_DELETE_WINDOW", app.quit)
    
    # Start main loop
    root.mainloop()

if __name__=="__main__":
    main()