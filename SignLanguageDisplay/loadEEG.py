import torch
import numpy as np
from scipy import signal
from scipy.stats import skew, kurtosis
import serial
import time
from collections import deque
import threading
import sys

class EEGEmotionDetector:
    def __init__(self, model_path, com_port, baudrate=57600, log_file="eeg_analysis.log"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Setup logging
        import logging
        self.logger = logging.getLogger('EEGDetector')
        self.logger.setLevel(logging.DEBUG)
        
        # File handler
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
        
        self.logger.info("Initializing EEG Emotion Detector")
        self.logger.info(f"Device: {self.device}")
        
        self.model = self.load_model(model_path)
        self.model.eval()
        
        self.com_port = com_port
        self.baudrate = baudrate
        self.serial_conn = None
        
        # Analysis stats
        self.stats = {
            'total_samples': 0,
            'valid_samples': 0,
            'emotion_predictions': {
                'Neutral': 0,
                'Happy': 0,
                'Sad': 0,
                'Angry': 0
            },
            'channel_stats': {
                'min_values': [],
                'max_values': [],
                'mean_values': [],
                'std_values': []
            },
            'feature_stats': {
                'min_values': [],
                'max_values': [],
                'mean_values': [],
                'std_values': []
            }
        }
        
        # Increased buffer size and added debug flag
        self.buffer_size = 128  # Increased to ensure enough data
        self.channels = 4
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.debug = True
        
        self.emotions = {
            0: "Neutral",
            1: "Happy",
            2: "Sad", 
            3: "Angry"
        }
        
        self.is_running = False
        
    def load_model(self, model_path):
        print(f"Loading model from {model_path}")
        try:
            # First try loading with weights_only=True
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        except Exception as e:
            print("Falling back to regular model loading...")
            # If that fails, load normally
            checkpoint = torch.load(model_path, map_location=self.device)
        
        model = ImprovedEEGNet(input_size=96).to(self.device)
        # Handle both old and new checkpoint formats
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print("Model loaded successfully")
        return model
        
    def connect(self):
        try:
            self.serial_conn = serial.Serial(
                port=self.com_port,
                baudrate=self.baudrate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=1  # Added timeout
            )
            print(f"Connected to EEG-SMT on {self.com_port}")
            time.sleep(2)  # Allow device to stabilize
            return True
        except Exception as e:
            print(f"Failed to connect: {str(e)}")
            return False
            
    def process_eeg_packet(self, packet):
        try:
            self.stats['total_samples'] += 1
            
            if len(packet) != 17:
                self.logger.debug(f"Invalid packet length: {len(packet)}")
                return None
                
            channels = []
            for i in range(4):
                value = (packet[i*2 + 1] << 8) | packet[i*2 + 2]
                # Ensure value is within expected range
                if value > 4095:  # 12-bit ADC max
                    self.logger.debug(f"Invalid channel value: {value} on channel {i}")
                    return None
                microvolts = value * (5000000 / 4096) / 2048
                channels.append(microvolts)
                
            # Basic validation
            if any(abs(v) > 500 for v in channels):  # Typical EEG < 500µV
                self.logger.debug(f"Abnormal voltage values: {channels}")
                return None
                
            # Update channel statistics
            channels_np = np.array(channels)
            self.stats['channel_stats']['min_values'].append(np.min(channels_np))
            self.stats['channel_stats']['max_values'].append(np.max(channels_np))
            self.stats['channel_stats']['mean_values'].append(np.mean(channels_np))
            self.stats['channel_stats']['std_values'].append(np.std(channels_np))
            
            self.stats['valid_samples'] += 1
            self.logger.debug(f"Valid sample processed: {channels}")
            
            return channels
        except Exception as e:
            self.logger.error(f"Error processing packet: {str(e)}")
            return None
            
    def extract_features(self, data):
        try:
            if len(data) < self.buffer_size/2:  # Ensure minimum data
                if self.debug:
                    print(f"Insufficient data: {len(data)} samples")
                return None
                
            features = []
            data = np.array(data)  # Convert to numpy array
            
            for channel in range(self.channels):
                channel_data = data[:, channel]
                
                # Handle NaN/Inf values
                channel_data = np.nan_to_num(channel_data, nan=0.0, posinf=0.0, neginf=0.0)
                
                # Time domain features
                features.extend([
                    np.mean(channel_data),
                    np.std(channel_data),
                    skew(channel_data, nan_policy='omit'),
                    kurtosis(channel_data, nan_policy='omit'),
                    np.max(channel_data),
                    np.min(channel_data),
                    np.ptp(channel_data),
                    np.sum(np.square(channel_data)),
                    np.sum(np.abs(channel_data))
                ])
                
                # Frequency domain features
                freqs, psd = signal.welch(channel_data, fs=59, nperseg=min(59, len(channel_data)))
                
                bands = {
                    'delta': (1, 4),
                    'theta': (4, 8), 
                    'alpha': (8, 13),
                    'beta': (13, 30),
                    'gamma': (30, 45)
                }
                
                total_power = np.sum(psd)
                if total_power == 0:
                    total_power = 1e-10  # Avoid division by zero
                    
                for band_name, (low, high) in bands.items():
                    mask = (freqs >= low) & (freqs <= high)
                    if not np.any(mask):
                        features.extend([0, 0, 0])  # Default values if band not found
                        continue
                        
                    band_power = np.mean(psd[mask])
                    relative_power = np.sum(psd[mask]) / total_power
                    peak_freq = freqs[mask][np.argmax(psd[mask])]
                    
                    features.extend([
                        band_power,
                        relative_power, 
                        peak_freq
                    ])
            
            features = np.array(features, dtype=np.float32)
            
            # Validate features
            if np.any(np.isnan(features)) or np.any(np.isinf(features)):
                if self.debug:
                    print("Invalid features detected")
                return None
            
            # Apply log transform only to the largest values (energy features)
            for i in range(len(features)):
                if abs(features[i]) > 1000:
                    features[i] = np.sign(features[i]) * np.log10(abs(features[i]))
                
            return features
            
        except Exception as e:
            if self.debug:
                print(f"Error extracting features: {str(e)}")
            return None

    def calibrate_neutral(self, duration=10):
        """
        Calibrate the model for neutral state detection.
        
        Args:
            duration: Duration in seconds to collect neutral state data
        """
        if not self.serial_conn or not self.serial_conn.is_open:
            print("Cannot calibrate - no connection")
            return False
            
        print(f"\nStarting neutral calibration - please relax for {duration} seconds")
        print("Keep your mind clear and maintain a neutral expression")
        
        # Clear any existing data
        self.data_buffer.clear()
        
        # Collect calibration data
        calibration_samples = []
        start_time = time.time()
        
        while time.time() - start_time < duration:
            remaining = int(duration - (time.time() - start_time))
            print(f"\rCalibrating: {remaining} seconds remaining...", end="")
            
            try:
                # Wait for sync byte with timeout
                sync_byte = self.serial_conn.read()
                if not sync_byte or sync_byte != b'\xA5':
                    continue
                    
                # Read packet
                packet = self.serial_conn.read(17)
                if len(packet) != 17:
                    continue
                    
                channels = self.process_eeg_packet(packet)
                if channels is None:
                    continue
                    
                self.data_buffer.append(channels)
                calibration_samples.append(channels)
            except Exception as e:
                print(f"\nError during calibration: {str(e)}")
        
        print("\nCalibration complete, processing baseline...")
        
        # Calculate confidence thresholds from calibration data
        if len(calibration_samples) > self.buffer_size//2:
            # Calculate features from calibration data
            features = self.extract_features(calibration_samples)
            if features is not None:
                # Get model prediction
                with torch.no_grad():
                    features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
                    outputs = self.model(features_tensor)
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)
                    
                    # Store baseline probabilities for later comparison
                    self.baseline_probabilities = probabilities[0].cpu().numpy()
                    self.baseline_confidence = {
                        self.emotions[i]: self.baseline_probabilities[i] 
                        for i in range(len(self.emotions))
                    }
                    
                    # Log baseline values
                    self.logger.info("Calibration baselines:")
                    for emotion, confidence in self.baseline_confidence.items():
                        self.logger.info(f"{emotion}: {confidence:.4f}")
                    
                    print("Calibration successful")
                    return True
        
        print("Calibration failed - insufficient data")
        return False
    def predict_emotion(self, features):
        try:
            if features is None:
                return None, None
                
            # Log feature statistics
            self.stats['feature_stats']['min_values'].append(np.min(features))
            self.stats['feature_stats']['max_values'].append(np.max(features))
            self.stats['feature_stats']['mean_values'].append(np.mean(features))
            self.stats['feature_stats']['std_values'].append(np.std(features))
            
            with torch.no_grad():
                features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
                outputs = self.model(features_tensor)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                raw_probs = probabilities[0].cpu().numpy()
                
                # Apply calibration if available
                if hasattr(self, 'baseline_probabilities'):
                    # Adjust probabilities based on baseline
                    adjusted_probs = raw_probs - (self.baseline_probabilities * [1.0, 1.0, 2.0, 1.0])
                    
                    # Apply a scaling factor to ensure valid probability distribution
                    adjusted_probs = np.clip(adjusted_probs, 0, None)
                    
                    # Normalize if sum is greater than 0
                    prob_sum = np.sum(adjusted_probs)
                    if prob_sum > 0:
                        adjusted_probs = adjusted_probs / prob_sum
                    else:
                        # If all adjustments are negative, fall back to raw probabilities
                        adjusted_probs = raw_probs
                    
                    # Apply penalties to frequently misclassified states
                    class_penalties = [0.0, 0.0, 0.3, 0.0]  # Penalty for Sad class
                    adjusted_probs = adjusted_probs - np.array(class_penalties)
                    adjusted_probs = np.clip(adjusted_probs, 0.0, None)
                    # Renormalize
                    if adjusted_probs.sum() > 0:
                        adjusted_probs = adjusted_probs / adjusted_probs.sum()
                    
                    # Apply a stronger correction for Sad class
                    class_weights = [1.0, 1.0, 0.7, 1.0]  # Reduce Sad weight
                    adjusted_probs = adjusted_probs * class_weights
                    
                    # Apply confidence threshold
                    prediction = np.argmax(adjusted_probs)
                    confidence = adjusted_probs[prediction]
                    
                    # Only declare non-neutral if confidence exceeds threshold
                    min_confidence = 0.3  # Adjust as needed
                    if prediction != 0 and confidence < min_confidence:
                        # Fall back to neutral if low confidence
                        prediction = 0
                        confidence = adjusted_probs[0]
                else:
                    # Without calibration, use raw probabilities
                    prediction = np.argmax(raw_probs)
                    confidence = raw_probs[prediction]
                
                # Validate prediction
                if prediction not in self.emotions:
                    self.logger.warning(f"Invalid prediction: {prediction}")
                    return None, None
                
                # Update emotion statistics
                emotion = self.emotions[prediction]
                self.stats['emotion_predictions'][emotion] += 1
                
                # Log detailed prediction info
                self.logger.debug(f"Raw probs: {raw_probs}")
                if hasattr(self, 'baseline_probabilities'):
                    self.logger.debug(f"Adjusted probs: {adjusted_probs}")
                self.logger.debug(f"Prediction: {emotion} (Confidence: {confidence:.2f})")
                
                return emotion, confidence
                
        except Exception as e:
            self.logger.error(f"Error in prediction: {str(e)}")
            return None, None
            
    def read_eeg_data(self):
        last_print_time = time.time()
        samples_processed = 0
        
        while self.is_running:
            if self.serial_conn is None or not self.serial_conn.is_open:
                print("Serial connection lost, attempting reconnect...")
                self.connect()
                time.sleep(1)
                continue
                
            try:
                # Wait for sync byte with timeout
                sync_byte = self.serial_conn.read()
                if not sync_byte:
                    continue
                if sync_byte != b'\xA5':
                    continue
                    
                # Read packet
                packet = self.serial_conn.read(17)
                if len(packet) != 17:
                    continue
                    
                channels = self.process_eeg_packet(packet)
                if channels is None:
                    continue
                    
                self.data_buffer.append(channels)
                samples_processed += 1
                
                # Process every second
                current_time = time.time()
                if current_time - last_print_time >= 10:
                    if len(self.data_buffer) >= self.buffer_size/2:
                        features = self.extract_features(list(self.data_buffer))
                        emotion, confidence = self.predict_emotion(features)
                        
                        if emotion is not None and confidence is not None:
                            print(f"\rEmotion: {emotion} (Confidence: {confidence:.2f}) - Processed {samples_processed} samples", end="")
                        else:
                            print(f"\rProcessing data... ({samples_processed} samples)", end="")
                            
                        sys.stdout.flush()
                        samples_processed = 0
                        last_print_time = current_time
                        
            except Exception as e:
                print(f"\nError reading data: {str(e)}")
                time.sleep(1)
                
    def start(self):
        if self.connect():
            self.is_running = True
            
            # Perform neutral calibration at startup
            if not self.calibrate_neutral(duration=20):
                print("Warning: Calibration failed, proceeding without calibration")
            
            self.read_thread = threading.Thread(target=self.read_eeg_data)
            self.read_thread.start()
            print("Started emotion detection")
        else:
            print("Failed to start - connection error")
            
    def stop(self):
        self.is_running = False
        if self.serial_conn:
            self.serial_conn.close()
        if hasattr(self, 'read_thread'):
            self.read_thread.join()
            
        # Log final statistics
        self.logger.info("\n=== Session Statistics ===")
        self.logger.info(f"Total samples processed: {self.stats['total_samples']}")
        self.logger.info(f"Valid samples: {self.stats['valid_samples']} ({self.stats['valid_samples']/max(1, self.stats['total_samples'])*100:.2f}%)")
        
        self.logger.info("\nEmotion Distribution:")
        total_predictions = sum(self.stats['emotion_predictions'].values())
        for emotion, count in self.stats['emotion_predictions'].items():
            percentage = (count / max(1, total_predictions)) * 100
            self.logger.info(f"{emotion}: {count} ({percentage:.2f}%)")
        
        self.logger.info("\nChannel Statistics:")
        for stat in ['min', 'max', 'mean', 'std']:
            values = self.stats['channel_stats'][f'{stat}_values']
            if values:
                # Replace 'μV' with 'uV' to avoid Unicode encoding issues
                self.logger.info(f"{stat.capitalize()}: {np.mean(values):.2f} uV")
        
        self.logger.info("\nFeature Statistics:")
        for stat in ['min', 'max', 'mean', 'std']:
            values = self.stats['feature_stats'][f'{stat}_values']
            if values:
                self.logger.info(f"{stat.capitalize()}: {np.mean(values):.2f}")
        
        print("\nStopped emotion detection - Check eeg_analysis.log for detailed statistics")


class ResidualBlock(torch.nn.Module):
    """Residual block used in the model."""
    def __init__(self, in_features):
        super().__init__()
        self.block = torch.nn.Sequential(
            torch.nn.Linear(in_features, in_features),
            torch.nn.LayerNorm(in_features),
            torch.nn.GELU(),
            torch.nn.Linear(in_features, in_features),
            torch.nn.LayerNorm(in_features)
        )
        
    def forward(self, x):
        return x + self.block(x)


class ImprovedEEGNet(torch.nn.Module):
    """Model architecture matching the training script."""
    def __init__(self, input_size, hidden_size=256):
        super().__init__()
        self.feature_extractor = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.LayerNorm(hidden_size),
            torch.nn.GELU(),
            ResidualBlock(hidden_size),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(hidden_size, hidden_size // 2),
            torch.nn.LayerNorm(hidden_size // 2),
            torch.nn.GELU(),
            ResidualBlock(hidden_size // 2),
            torch.nn.Dropout(0.2)
        )
        
        self.classifiers = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(hidden_size // 2, hidden_size // 4),
                torch.nn.LayerNorm(hidden_size // 4),
                torch.nn.GELU(),
                torch.nn.Linear(hidden_size // 4, 1)
            ) for _ in range(4)
        ])
        
    def forward(self, x):
        features = self.feature_extractor(x)
        outputs = torch.cat([classifier(features) for classifier in self.classifiers], dim=1)
        return outputs, features


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='EEG-SMT Emotion Detection')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model (.pth file)')
    parser.add_argument('--port', type=str, required=True, help='COM port for EEG-SMT')
    parser.add_argument('--baud', type=int, default=57600, help='Baud rate (default: 57600)')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    
    args = parser.parse_args()
    
    detector = EEGEmotionDetector(args.model, args.port, args.baud)
    detector.debug = args.debug
    
    try:
        detector.start()
        input("Press Enter to stop...\n")
    except KeyboardInterrupt:
        pass
    finally:
        detector.stop()