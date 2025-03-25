#!/usr/bin/env python3
"""
Dendritic Attention System - Input Source Comparison Test

This script modifies the original adaptive dendritic system to test whether 
the fractal patterns in the attention field emerge regardless of input source.

It provides multiple input modes to compare:
1. Audio-driven 3D phase space (original)
2. Webcam input
3. Static patterns
4. Pure random noise
5. Prerecorded sequence

The system tracks fractal dimension of the attention field over time for each input mode.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import time
import numpy as np
import cv2
import pyaudio
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D  # noqa
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import logging
from PIL import Image, ImageTk
from scipy import stats

# Configure logging
logging.basicConfig(
    filename='dendritic_test.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.DEBUG
)
logger = logging.getLogger(__name__)

# ---------------------------
# Fractal Analysis Functions
# ---------------------------
def box_count(data, box_size):
    """Count how many boxes of size box_size contain any part of the pattern."""
    S = np.add.reduceat(
            np.add.reduceat(data, np.arange(0, data.shape[0], box_size), axis=0),
                               np.arange(0, data.shape[1], box_size), axis=1)
    # Count boxes where sum is greater than zero
    return np.sum(S > 0)

def fractal_dimension(Z, min_box=2, max_box=None, step=2):
    """
    Compute the fractal dimension of a 2D array Z using the box-counting method.
    The fractal dimension D is given by -slope of log(box_count) vs log(box_size).
    """
    # Ensure Z is binary
    Z = Z > Z.mean()    
    
    # Set max_box as smallest dimension if not provided
    if max_box is None:
        max_box = min(Z.shape) // 4
    
    # Ensure max_box doesn't exceed array dimensions
    max_box = min(max_box, min(Z.shape) // 2)
    
    # Ensure we have enough boxes to calculate a meaningful dimension
    min_box = max(2, min_box)
    if max_box <= min_box:
        return 1.0, [min_box], [1]  # Default to dimension 1 if range is too small
        
    sizes = np.arange(min_box, max_box, step)
    if len(sizes) < 2:
        sizes = np.array([min_box, max_box-1])
        
    counts = []
    for size in sizes:
        count = box_count(Z, size)
        if count > 0:
            counts.append(count)
        else:
            counts.append(1)

    # Fit a line to the log-log plot
    log_sizes = np.log(sizes)
    log_counts = np.log(counts)
    
    # Use linear regression to find the slope
    try:
        slope, intercept, r_value, p_value, std_err = stats.linregress(log_sizes, log_counts)
        return -slope, sizes, counts
    except:
        logger.error("Error calculating fractal dimension")
        return 1.0, sizes, counts  # Default to dimension 1 on error


# ---------------------------
# DendriticAttention Class
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
        self.response_vectors = np.random.randn(4, 5)  # 4 features to 5 frequency responses
        self.attention_width = 0.5  # Initial width (0-1 scale)
        self.stability_measure = 0.5  # How stable the system thinks patterns are
        
        # Activity history
        self.activity_history = []
        self.match_history = []
        self.fractal_dimension_history = []
        self.reset_time = time.time()
        
        # Exploration parameters
        self.exploration_rate = 0.5
        
    def normalize(self, vectors):
        """Normalize vectors to unit length"""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / (norms + 1e-8)
        
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
        numerator = np.dot(input_centered, expected_centered)
        denominator = np.sqrt(np.sum(input_centered**2) * np.sum(expected_centered**2))
        
        if denominator < 1e-8:
            return 0.0
            
        correlation = numerator / denominator
        return max(0, (correlation + 1) / 2)  # Scale to [0,1]
        
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
        
        # Calculate fractal dimension of the attention field visualization
        vis_img = self.get_visualization()
        if vis_img is not None:
            red_channel = vis_img[:, :, 0]  # Red channel contains expected pattern
            fd, _, _ = fractal_dimension(red_channel)
            self.fractal_dimension_history.append((time.time() - self.reset_time, fd))
            # Keep only the last 100 values
            if len(self.fractal_dimension_history) > 100:
                self.fractal_dimension_history.pop(0)
        
        # Calculate current exploration rate based on stability and runtime
        runtime = time.time() - self.reset_time
        base_exploration = max(0.1, 1.0 - min(1.0, runtime / 60.0))  # Decrease over first minute
        stability_factor = 1.0 - self.stability_measure
        self.exploration_rate = 0.7 * self.exploration_rate + 0.3 * (base_exploration + 0.5 * stability_factor)
        
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
        
        self.activity_history.append(total_activity)
        if len(self.activity_history) > 100:
            self.activity_history.pop(0)
    
    def get_frequency_adjustments(self, num_neurons=5):
        """Calculate frequency adjustments for speaker neurons"""
        # Ensure we have the right number of outputs
        if self.response_vectors.shape[1] != num_neurons:
            self.response_vectors = np.random.randn(4, num_neurons) * 0.1
            
        # Calculate adjustments: features dot response_matrix
        raw_adjustments = np.dot(self.activity_vector, self.response_vectors)
        
        # Scale based on exploration rate (higher exploration = bigger changes)
        scaled_adjustments = raw_adjustments * (0.5 + self.exploration_rate)
        
        # Add frequency exploration using oscillation
        time_factor = np.sin(time.time() * np.pi * 0.1)  # Slow oscillation over time
        exploration_wave = np.sin(np.linspace(0, 2*np.pi, num_neurons) + time_factor)
        scaled_adjustments += exploration_wave * self.exploration_rate * 0.2
        
        # Add more variation if stability is low
        if self.stability_measure < 0.5:
            scaled_adjustments += np.random.randn(num_neurons) * (0.5 - self.stability_measure) * 0.3
            
        return scaled_adjustments
        
    def learn_response(self, adjustments, match_delta):
        """Learn which adjustments lead to better pattern matching"""
        # If match score improved, reinforce those adjustments
        if match_delta > 0:
            for i in range(len(self.activity_vector)):
                self.response_vectors[i] += 0.01 * self.activity_vector[i] * adjustments
        else:
            # If match got worse, do the opposite next time
            for i in range(len(self.activity_vector)):
                self.response_vectors[i] -= 0.01 * self.activity_vector[i] * adjustments
                
        # Occasionally add exploration noise
        if np.random.random() < 0.05:
            self.response_vectors += np.random.randn(*self.response_vectors.shape) * 0.01
    
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
        
    def reset(self):
        """Reset the attention system"""
        # Reset pattern memory
        self.expected_pattern = None
        self.memory_strength = 0.0
        
        # Reset attention field
        self.attention_width = 0.5
        self.stability_measure = 0.5
        
        # Reset history
        self.activity_history = []
        self.match_history = []
        self.fractal_dimension_history = []
        self.reset_time = time.time()
        
        # Reset dendrites partially - keep positions but reset strengths and randomize directions
        self.strengths = np.ones(self.n_dendrites) * 0.5
        self.directions = self.normalize(np.random.randn(self.n_dendrites, 2))
        
        # Reset exploration
        self.exploration_rate = 0.5


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
            self.output_stream = None
    
    def generate_wave(self):
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
        if self.output_stream:
            try:
                self.output_stream.write(wave.tobytes())
            except Exception as e:
                logger.error(f"Audio output error: {e}")
        return wave
    
    def cleanup(self):
        try:
            if hasattr(self, 'output_stream') and self.output_stream:
                self.output_stream.stop_stream()
                self.output_stream.close()
            self.audio.terminate()
            logger.info("Cleaned up neuron audio")
        except Exception as e:
            logger.error(f"Cleanup error: {e}")


# ---------------------------
# Webcam Input Class
# ---------------------------
class WebcamInput:
    """Handles webcam input for the attention system."""
    def __init__(self, camera_index=0):
        self.camera_index = camera_index
        self.cap = None
        self.frame = None
        self.running = False
        self.thread = None
    
    def start(self):
        """Start the webcam capture thread."""
        if self.cap is not None:
            return
            
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                logger.error(f"Failed to open camera {self.camera_index}")
                self.cap = None
                return False
                
            self.running = True
            self.thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.thread.start()
            return True
        except Exception as e:
            logger.error(f"Error starting webcam: {e}")
            return False
    
    def _capture_loop(self):
        """Background thread to continuously capture frames."""
        while self.running and self.cap is not None:
            try:
                ret, frame = self.cap.read()
                if ret:
                    # Convert to grayscale for simplicity
                    self.frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                time.sleep(0.03)  # ~30 fps
            except Exception as e:
                logger.error(f"Webcam capture error: {e}")
                time.sleep(0.1)
    
    def get_frame(self):
        """Get the latest webcam frame."""
        if self.frame is None:
            # Return random noise if no frame is available
            return np.random.rand(64, 64)
        # Return a resized copy of the frame
        return cv2.resize(self.frame, (64, 64))
    
    def stop(self):
        """Stop the webcam capture."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()
            self.cap = None
            self.frame = None


# ---------------------------
# Pattern Generator Class
# ---------------------------
class PatternGenerator:
    """Generates various patterns for testing the attention system."""
    def __init__(self, size=(64, 64)):
        self.size = size
        self.pattern_type = "random"
        self.t = 0.0
        self.static_pattern = None
        self.sequence = []
        self.seq_index = 0
    
    def set_pattern_type(self, pattern_type):
        """Set the type of pattern to generate."""
        self.pattern_type = pattern_type
        self.t = 0.0
        
        # Generate a fixed random pattern for static mode
        if pattern_type == "static":
            self.static_pattern = np.random.rand(*self.size)
            
        # For sequence mode, generate a sequence of patterns
        if pattern_type == "sequence":
            self.sequence = []
            for i in range(10):  # Generate 10 frames
                if i < 5:
                    # Create a moving horizontal bar
                    frame = np.zeros(self.size)
                    bar_pos = int(i * self.size[0] / 5)
                    bar_width = max(1, self.size[0] // 10)
                    frame[bar_pos:bar_pos+bar_width, :] = 1.0
                else:
                    # Create a moving vertical bar
                    frame = np.zeros(self.size)
                    bar_pos = int((i-5) * self.size[1] / 5)
                    bar_width = max(1, self.size[1] // 10)
                    frame[:, bar_pos:bar_pos+bar_width] = 1.0
                self.sequence.append(frame)
            self.seq_index = 0
    
    def get_frame(self):
        """Get the next frame based on the current pattern type."""
        if self.pattern_type == "random":
            # Pure random noise
            return np.random.rand(*self.size)
            
        elif self.pattern_type == "static":
            # Return the fixed pattern
            return self.static_pattern
            
        elif self.pattern_type == "sequence":
            # Return the next frame in the sequence
            frame = self.sequence[self.seq_index]
            self.seq_index = (self.seq_index + 1) % len(self.sequence)
            return frame
            
        elif self.pattern_type == "dynamic":
            # Create a dynamic pattern that changes over time
            x, y = np.meshgrid(
                np.linspace(-3, 3, self.size[1]),
                np.linspace(-3, 3, self.size[0])
            )
            # Moving Gaussian + ripples
            center_x = 3 * np.sin(self.t * 0.1)
            center_y = 3 * np.cos(self.t * 0.13)
            d_squared = (x - center_x)**2 + (y - center_y)**2
            gaussian = np.exp(-d_squared / 2.0)
            ripple = 0.2 * np.sin(5.0 * np.sqrt(d_squared) - self.t)
            pattern = gaussian + ripple
            
            # Normalize to [0, 1]
            pattern = (pattern - pattern.min()) / (pattern.max() - pattern.min() + 1e-8)
            
            # Update time
            self.t += 0.1
            
            return pattern
            
        else:
            # Default to random
            return np.random.rand(*self.size)


# ---------------------------
# DendriticInputTester Class
# ---------------------------
class DendriticInputTester:
    def __init__(self, root):
        self.root = root
        self.root.title("Dendritic Attention System - Input Source Test")
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
        
        # Initialize webcam input
        self.webcam = WebcamInput()
        
        # Initialize pattern generator
        self.pattern_generator = PatternGenerator()
        
        # Flag to control system state
        self.running = False
        self.input_mode = "phase_space"  # Default input mode
        self.phase_image = None
        
        # Setup UI layout
        self.setup_layout()
        
        # Variable to track the last adjustment
        self.last_adjustments = np.zeros(self.num_neurons)
        self.last_match_score = 0.5
        
        # Counter for adjustments
        self.adjustment_counter = 0
        
        # Create the initial speaker neurons
        self.create_speaker_neurons()

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
        
        # Tab 1: Input Visualization
        self.input_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.input_tab, text="Input")
        
        self.input_canvas = tk.Canvas(self.input_tab, bg="black")
        self.input_canvas.pack(fill="both", expand=True)
        
        # Tab 2: 3D Phase Space
        self.phase_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.phase_tab, text="3D Phase Space")
        
        self.fig_3d = plt.Figure(figsize=(8, 6))
        self.ax3d = self.fig_3d.add_subplot(111, projection='3d')
        self.ax3d.set_title("Combined 3D Phase Space")
        self.canvas_3d = FigureCanvasTkAgg(self.fig_3d, self.phase_tab)
        self.canvas_3d.get_tk_widget().pack(fill="both", expand=True)
        
        # Tab 3: Attention Visualization
        self.attn_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.attn_tab, text="Attention Field")
        
        self.attn_canvas = tk.Canvas(self.attn_tab, bg="black")
        self.attn_canvas.pack(fill="both", expand=True)
        
        # Tab 4: Fractal Dimension
        self.fractal_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.fractal_tab, text="Fractal Dimension")
        
        self.fig_fractal = plt.Figure(figsize=(8, 6))
        self.ax_fractal = self.fig_fractal.add_subplot(111)
        self.ax_fractal.set_title("Fractal Dimension Over Time")
        self.ax_fractal.set_xlabel("Time (seconds)")
        self.ax_fractal.set_ylabel("Fractal Dimension")
        self.ax_fractal.grid(True)
        self.canvas_fractal = FigureCanvasTkAgg(self.fig_fractal, self.fractal_tab)
        self.canvas_fractal.get_tk_widget().pack(fill="both", expand=True)
        
        # Tab 5: Activity Metrics
        self.activity_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.activity_tab, text="Activity Metrics")
        
        self.fig_activity = plt.Figure(figsize=(8, 6))
        self.ax_activity = self.fig_activity.add_subplot(111)
        self.ax_activity.set_title("System Activity")
        self.canvas_activity = FigureCanvasTkAgg(self.fig_activity, self.activity_tab)
        self.canvas_activity.get_tk_widget().pack(fill="both", expand=True)
        
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
        
        # Input Mode selection
        input_frame = tk.LabelFrame(self.control_scrollable_frame, text="Input Source", padx=5, pady=5)
        input_frame.pack(fill="x", padx=5, pady=5)
        
        self.input_mode_var = tk.StringVar(value="phase_space")
        ttk.Radiobutton(input_frame, text="3D Phase Space (Audio)", variable=self.input_mode_var, 
                       value="phase_space", command=self.change_input_mode).pack(anchor="w", padx=5, pady=2)
        ttk.Radiobutton(input_frame, text="Webcam", variable=self.input_mode_var, 
                       value="webcam", command=self.change_input_mode).pack(anchor="w", padx=5, pady=2)
        ttk.Radiobutton(input_frame, text="Random Noise", variable=self.input_mode_var, 
                       value="random", command=self.change_input_mode).pack(anchor="w", padx=5, pady=2)
        ttk.Radiobutton(input_frame, text="Static Pattern", variable=self.input_mode_var, 
                       value="static", command=self.change_input_mode).pack(anchor="w", padx=5, pady=2)
        ttk.Radiobutton(input_frame, text="Dynamic Pattern", variable=self.input_mode_var, 
                       value="dynamic", command=self.change_input_mode).pack(anchor="w", padx=5, pady=2)
        ttk.Radiobutton(input_frame, text="Pattern Sequence", variable=self.input_mode_var, 
                       value="sequence", command=self.change_input_mode).pack(anchor="w", padx=5, pady=2)
        
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
        self.exploration_var = tk.DoubleVar(value=0.5)
        ttk.Progressbar(state_frame, variable=self.exploration_var, maximum=1.0, length=100).grid(row=3, column=1, sticky="ew", padx=2, pady=2)
        
        # Fractal dimension indicator
        tk.Label(state_frame, text="Fractal Dimension:").grid(row=4, column=0, sticky="w", padx=2, pady=2)
        self.fractal_dim_var = tk.StringVar(value="0.00")
        tk.Label(state_frame, textvariable=self.fractal_dim_var).grid(row=4, column=1, sticky="w", padx=2, pady=2)
        
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
        
        # Export results button
        self.export_button = ttk.Button(controls_frame, text="Export Results", command=self.export_results)
        self.export_button.pack(side="left", fill="x", expand=True, padx=5, pady=5)
    
    def create_speaker_neurons(self):
        # Remove old controls if any
        for widget in self.neuron_controls_frame.winfo_children():
            widget.destroy()
            
        # Clean up existing neurons
        for neuron in self.neurons:
            neuron.cleanup()
            
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
            init_freq = 200 + (i * 800) / (self.num_neurons - 1 or 1)
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
    
    def change_input_mode(self):
        """Handle changes in input mode selection."""
        new_mode = self.input_mode_var.get()
        logger.info(f"Changing input mode to: {new_mode}")
        
        # Stop webcam if we're switching away from it
        if self.input_mode == "webcam" and new_mode != "webcam":
            self.webcam.stop()
        
        # Start webcam if we're switching to it
        if new_mode == "webcam" and self.input_mode != "webcam":
            if not self.webcam.start():
                messagebox.showerror("Error", "Failed to start webcam. Falling back to random noise.")
                self.input_mode_var.set("random")
                new_mode = "random"
        
        # Set up pattern generator for the selected mode
        if new_mode in ["random", "static", "dynamic", "sequence"]:
            self.pattern_generator.set_pattern_type(new_mode)
        
        # Update the input mode
        self.input_mode = new_mode
    
    def update_neuron_frequency(self, idx, freq, adjust_slider=True):
        """Update the frequency of a speaker neuron."""
        # Ensure frequency is within reasonable bounds
        freq = max(20, min(2000, freq))
        
        # Update the neuron object
        self.neurons[idx].base_freq = freq
        
        # If requested, update the slider too
        if adjust_slider:
            self.neuron_freq_vars[idx].set(freq)
    
    def toggle_running(self):
        """Start or stop the processing loop."""
        self.running = not self.running
        if self.running:
            self.start_stop_button.config(text="Stop")
            # Start processing in separate thread
            self.processing_thread = threading.Thread(target=self.processing_loop, daemon=True)
            self.processing_thread.start()
        else:
            self.start_stop_button.config(text="Start")
    
    def reset_system(self):
        """Reset the attention system and clear history."""
        # Reset the attention system
        self.attention_system.reset()
        
        # Reset wave history
        self.wave_history = []
        
        # Reset metrics
        self.last_match_score = 0.5
        self.last_adjustments = np.zeros(self.num_neurons)
        
        # Reset UI indicators
        self.attn_width_var.set(0.5)
        self.stability_var.set(0.5)
        self.match_var.set(0.5)
        self.exploration_var.set(0.5)
        self.fractal_dim_var.set("0.00")
        
        # Update visualizations
        self.update_visualization()
    
    def capture_phase_space(self, buffer_data=None):
        """Capture an image of the current 3D phase space plot for the attention system to observe."""
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
        colors = plt.cm.viridis(energy / energy.max())
        
        # Plot the 3D phase space
        scatter = self.ax3d.scatter(w1, w2, w3, c=colors, s=5, alpha=0.7)
        
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
        try:
            self.canvas_3d.draw()
        except AttributeError as e:
            logger.warning(f"3D rendering issue skipped: {e}")
        
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
    
    def get_input_frame(self):
        """Get the current input frame based on the selected input mode."""
        if self.input_mode == "phase_space":
            # Use the captured phase space image
            if self.phase_image is not None:
                # Convert to grayscale for simplicity
                return cv2.cvtColor(self.phase_image, cv2.COLOR_RGB2GRAY)
            # Fall back to random noise if no phase image yet
            return np.random.rand(64, 64)
            
        elif self.input_mode == "webcam":
            # Use webcam input
            return self.webcam.get_frame()
            
        else:
            # Use pattern generator for all other modes
            return self.pattern_generator.get_frame()
    
    def processing_loop(self):
        """Main processing loop that coordinates speaker neurons and dendritic attention."""
        last_adaptation_time = time.time()
        adaptation_interval = 0.2  # seconds between adaptations
        fractal_update_interval = 1.0  # seconds between fractal dimension plot updates
        last_fractal_update = time.time()
        
        while self.running:
            try:
                # 1. Generate waveforms from each neuron (used in phase_space mode)
                combined_wave = np.zeros(self.buffer_size, dtype=np.float32)
                
                # Only generate audio if we're using phase_space mode
                if self.input_mode == "phase_space":
                    for neuron in self.neurons:
                        wave = neuron.generate_wave()
                        combined_wave += wave
                    
                    # Normalize to prevent clipping
                    max_val = np.max(np.abs(combined_wave))
                    if max_val > 1.0:
                        combined_wave /= max_val
                    
                    # Play the audio if we have an output device
                    if self.neurons and self.neurons[0].output_stream:
                        try:
                            self.neurons[0].output_stream.write(combined_wave.tobytes())
                        except Exception as e:
                            logger.error(f"Audio playback error: {e}")
                    
                    # Store for visualization
                    self.wave_history.append(combined_wave)
                    if len(self.wave_history) > 50:
                        self.wave_history.pop(0)
                    
                    # Update phase space visualization
                    self.root.after(0, self.capture_phase_space)
                
                # 2. Get input frame for attention system based on selected mode
                input_frame = self.get_input_frame()
                
                # 3. Update input visualization
                self.root.after(0, lambda: self.update_input_visualization(input_frame))
                
                # 4. Process input through attention system
                current_time = time.time()
                if (current_time - last_adaptation_time) >= adaptation_interval:
                    # Process the image through attention system
                    attention_field = self.attention_system.update(input_frame)
                    
                    # Calculate match score
                    current_match = np.mean(self.attention_system.match_history[-5:]) if len(self.attention_system.match_history) >= 5 else 0.5
                    match_delta = current_match - self.last_match_score
                    
                    # Only adjust neurons in phase_space mode
                    if self.input_mode == "phase_space":
                        # Get frequency adjustments from attention system
                        adjustments = self.attention_system.get_frequency_adjustments(self.num_neurons)
                        
                        # Apply adjustments to neurons with scaling factor
                        scale_factor = 20  # Hz per adjustment unit
                        for i in range(min(len(adjustments), len(self.neurons))):
                            # Apply adjustment to current frequency
                            curr_freq = self.neurons[i].base_freq
                            new_freq = curr_freq + adjustments[i] * scale_factor
                            # Update neuron and UI
                            self.update_neuron_frequency(i, new_freq)
                        
                        # Learn from the results
                        self.attention_system.learn_response(self.last_adjustments, match_delta)
                        
                        # Update state for next iteration
                        self.last_adjustments = adjustments
                    
                    self.last_match_score = current_match
                    
                    # Update UI indicators
                    self.root.after(0, self.update_indicators)
                    
                    # Reset timer
                    last_adaptation_time = current_time
                
                # 5. Update fractal dimension plot
                if (current_time - last_fractal_update) >= fractal_update_interval:
                    self.root.after(0, self.update_fractal_plot)
                    last_fractal_update = current_time
                
                # Short sleep to keep CPU usage reasonable
                time.sleep(0.02)
                
            except Exception as e:
                logger.error(f"Processing error: {e}")
                time.sleep(0.1)
    
    def update_input_visualization(self, input_frame):
        """Update the input visualization canvas."""
        if input_frame is None:
            return
            
        # Create a PIL image from the input frame
        if input_frame.max() <= 1.0:
            # Normalize to [0, 255] for PIL
            input_frame = (input_frame * 255).astype(np.uint8)
        
        pil_img = Image.fromarray(input_frame)
        
        # Resize to fit canvas
        canvas_width = self.input_canvas.winfo_width()
        canvas_height = self.input_canvas.winfo_height()
        
        if canvas_width > 100 and canvas_height > 100:  # Only if canvas has been drawn
            pil_img = pil_img.resize((canvas_width, canvas_height), Image.LANCZOS)
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(pil_img)
            # Update canvas
            self.input_canvas.delete("all")
            self.input_canvas.create_image(0, 0, anchor=tk.NW, image=photo)
            self.input_canvas.image = photo  # Keep reference
    
    def update_indicators(self):
        """Update UI indicators with current system state."""
        # Update attention width indicator
        self.attn_width_var.set(self.attention_system.attention_width)
        
        # Update stability indicator
        self.stability_var.set(self.attention_system.stability_measure)
        
        # Update match score indicator
        current_match = np.mean(self.attention_system.match_history[-5:]) if len(self.attention_system.match_history) >= 5 else 0.5
        self.match_var.set(current_match)
        
        # Update exploration rate indicator
        self.exploration_var.set(self.attention_system.exploration_rate)
        
        # Update fractal dimension indicator
        if self.attention_system.fractal_dimension_history:
            fd = self.attention_system.fractal_dimension_history[-1][1]
            self.fractal_dim_var.set(f"{fd:.3f}")
        
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
                self.attn_canvas.delete("all")
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
            
            self.ax_activity.legend()
            self.ax_activity.set_ylim([0, 1])
            try:
                self.canvas_activity.draw()
            except Exception as e:
                logger.error(f"Error drawing activity plot: {e}")
    
    def update_fractal_plot(self):
        """Update the fractal dimension plot."""
        if not self.attention_system.fractal_dimension_history:
            return
            
        self.ax_fractal.clear()
        self.ax_fractal.set_title(f"Fractal Dimension - {self.input_mode}")
        self.ax_fractal.set_xlabel("Time (seconds)")
        self.ax_fractal.set_ylabel("Fractal Dimension")
        
        # Extract time and fractal dimension values
        times, dimensions = zip(*self.attention_system.fractal_dimension_history)
        
        # Plot fractal dimension over time
        self.ax_fractal.plot(times, dimensions, 'r-')
        
        # Add a horizontal line at the current value
        if dimensions:
            self.ax_fractal.axhline(y=dimensions[-1], color='k', linestyle='--', alpha=0.5)
        
        # Set y-axis limits to reasonable fractal dimension range
        self.ax_fractal.set_ylim([1.0, 2.0])
        
        # Add grid
        self.ax_fractal.grid(True)
        
        try:
            self.canvas_fractal.draw()
        except Exception as e:
            logger.error(f"Error drawing fractal plot: {e}")
    
    def update_visualization(self):
        """Update visualizations."""
        # Update phase space visualization if in that mode
        if self.input_mode == "phase_space" and len(self.wave_history) >= 3:
            self.capture_phase_space()
    
    def export_results(self):
        """Export the current results to a file."""
        try:
            # Create a timestamp for the filename
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
                initialfile=f"fractal_dimension_{self.input_mode}_{timestamp}.csv"
            )
            
            if not filename:
                return  # User cancelled
                
            # Write fractal dimension history to CSV
            with open(filename, 'w') as f:
                f.write("Time,FractalDimension,InputMode,AttentionWidth,Stability,MatchScore\n")
                for i, (t, fd) in enumerate(self.attention_system.fractal_dimension_history):
                    # Get the matching metrics if available
                    match_score = self.attention_system.match_history[i] if i < len(self.attention_system.match_history) else 0
                    
                    f.write(f"{t:.2f},{fd:.6f},{self.input_mode},{self.attention_system.attention_width:.4f}," +
                            f"{self.attention_system.stability_measure:.4f},{match_score:.4f}\n")
                    
            messagebox.showinfo("Export Complete", f"Data exported to {filename}")
            
        except Exception as e:
            logger.error(f"Error exporting results: {e}")
            messagebox.showerror("Export Error", f"Failed to export results: {e}")
    
    def cleanup(self):
        """Cleanup resources when closing."""
        self.running = False
        
        # Stop webcam if running
        if hasattr(self, 'webcam'):
            self.webcam.stop()
            
        # Clean up audio resources
        for neuron in self.neurons:
            neuron.cleanup()
            
        if hasattr(self, 'p'):
            self.p.terminate()
            
        logger.info("System resources cleaned up")
    
    def quit(self):
        """Handle application shutdown."""
        self.cleanup()
        self.root.quit()
        self.root.destroy()


# ---------------------------
# Main function
# ---------------------------
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
    root.title("Dendritic Attention System - Input Source Test")
    
    # Create and run application
    app = DendriticInputTester(root)
    
    # Set up clean shutdown
    root.protocol("WM_DELETE_WINDOW", app.quit)
    
    # Start main loop
    root.mainloop()


# ---------------------------
# Run the application
# ---------------------------
if __name__ == "__main__":
    main()