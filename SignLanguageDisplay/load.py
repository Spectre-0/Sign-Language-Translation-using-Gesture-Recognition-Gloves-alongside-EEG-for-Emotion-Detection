import asyncio
import numpy as np
import pandas as pd
import joblib
import time
import os
import sys
from collections import deque
import tkinter as tk
from tkinter import ttk, scrolledtext
import threading
import datetime
import pickle

# Import your ArduinoInterpreter class
from utils.ArduinoInterpreter import ArduinoInterpreter

import torch
import torch.nn as nn
from pytorch.model79Base import SignLanguageModel

class BSLRecognizer:
    def __init__(self, model_path='bsl_model.joblib', scaler_path='bsl_scaler.joblib', feature_names_path='feature_names.joblib'):
        """Initialize the BSL Sign Language Recognizer"""
        print("Initializing BSL Sign Language Recognizer...")
        
        # Load the model
        try:
            self.model = joblib.load(model_path)
            print(f"Model loaded from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            sys.exit(1)
        
        # Load the scaler
        try:
            self.scaler = joblib.load(scaler_path)
            print(f"Scaler loaded from {scaler_path}")
        except Exception as e:
            print(f"Error loading scaler: {e}")
            sys.exit(1)
        
        # Load feature names
        try:
            self.feature_names = joblib.load(feature_names_path)
            print(f"Feature names loaded from {feature_names_path}")
        except Exception as e:
            print(f"Error loading feature names: {e}")
            sys.exit(1)
        
        # Recent predictions for smoothing
        self.recent_predictions = deque(maxlen=5)
        self.prediction_counter = {}
        self.cooldown_period = 1.0  # Seconds between predictions
        self.last_prediction_time = 0
        
    def set_cooldown(self, seconds):
        """Set the cooldown period between predictions"""
        self.cooldown_period = seconds
        
    def preprocess_data(self, raw_data):
        """Preprocess raw sensor data to match model input format"""
        # Create features dictionary from raw data
        features = {
            'hand': raw_data[0],
            'flex1': raw_data[1],
            'flex2': raw_data[2],
            'flex3': raw_data[3],
            'flex4': raw_data[4],
            'flex5': raw_data[5],
            'accelX': raw_data[6],
            'accelY': raw_data[7],
            'accelZ': raw_data[8],
            'gyroX': raw_data[9],
            'gyroY': raw_data[10],
            'gyroZ': raw_data[11],
            'roll': raw_data[12],
            'pitch': raw_data[13]
        }
        
        # Create derived features (match training preprocessing)
        features['flex_ratio_1_2'] = features['flex1'] / (features['flex2'] + 1e-6)
        features['flex_ratio_2_3'] = features['flex2'] / (features['flex3'] + 1e-6)
        features['flex_ratio_3_4'] = features['flex3'] / (features['flex4'] + 1e-6)
        features['flex_ratio_4_5'] = features['flex4'] / (features['flex5'] + 1e-6)
        
        features['index_curve'] = features['flex1'] / 1000.0
        features['middle_curve'] = features['flex2'] / 1000.0
        features['ring_curve'] = features['flex3'] / 1000.0
        features['pinky_curve'] = features['flex4'] / 1000.0
        features['thumb_curve'] = features['flex5'] / 1000.0
        
        features['hand_orientation'] = np.sqrt(features['roll']**2 + features['pitch']**2)
        features['accel_magnitude'] = np.sqrt(features['accelX']**2 + features['accelY']**2 + features['accelZ']**2)
        features['gyro_magnitude'] = np.sqrt(features['gyroX']**2 + features['gyroY']**2 + features['gyroZ']**2)
        
        features['is_right_hand'] = features['hand']
        
        # Create DataFrame with all required features
        df = pd.DataFrame([features])
        
        # Ensure all expected features are present
        for feature in self.feature_names:
            if feature not in df.columns:
                df[feature] = 0
        
        # Get features in correct order
        df = df[self.feature_names]
        
        return df
    
    def predict(self, raw_data):
        """Predict the sign from raw sensor data with confidence score"""
        # Check if we're in cooldown period
        current_time = time.time()
        if current_time - self.last_prediction_time < self.cooldown_period:
            return None, None
            
        if len(raw_data) < 14:
            return None, None
            
        # Preprocess the data
        df = self.preprocess_data(raw_data)
        
        # Scale the data
        X = self.scaler.transform(df)
        
        # Get prediction and confidence scores
        prediction = None
        confidence = None
        
        # Check if the model has predict_proba method (Random Forest does)
        if hasattr(self.model, 'predict_proba'):
            # Get class probabilities
            proba = self.model.predict_proba(X)
            # Get the predicted class
            pred_class = self.model.predict(X)[0]
            # Find the index of the predicted class
            if hasattr(self.model, 'classes_'):
                class_index = np.where(self.model.classes_ == pred_class)[0][0]
                # Get the confidence score for this prediction
                confidence = proba[0][class_index]
            else:
                # If classes_ is not available, just use the max probability
                confidence = np.max(proba)
            prediction = pred_class
        else:
            # Fallback if predict_proba is not available
            prediction = self.model.predict(X)[0]
            confidence = 1.0  # Default confidence
        
        # Add to recent predictions
        self.recent_predictions.append(prediction)
        
        # Smooth prediction using a voting mechanism
        if len(self.recent_predictions) >= 3:
            prediction_counts = {}
            for pred in self.recent_predictions:
                prediction_counts[pred] = prediction_counts.get(pred, 0) + 1
            
            smoothed_prediction = max(prediction_counts, key=prediction_counts.get)
            smoothed_confidence = prediction_counts[smoothed_prediction] / len(self.recent_predictions)
            
            # Only return a prediction if we're confident
            if prediction_counts[smoothed_prediction] >= 3:
                self.prediction_counter[smoothed_prediction] = self.prediction_counter.get(smoothed_prediction, 0) + 1
                
                if self.prediction_counter[smoothed_prediction] >= 3:
                    # Reset counter after reporting to avoid repetition
                    self.prediction_counter[smoothed_prediction] = 0
                    self.last_prediction_time = current_time
                    return smoothed_prediction, confidence * smoothed_confidence
        
        return None, None

class BSLRecognizerTorch:
    def __init__(self, model_path='pytorch/79_model_params.pth', encoder_path='../../Model_Specific/PreProcessing/labelEncoder.pkl'):
        """Initialize the BSL Sign Language Recognizer"""
        print("Initializing BSL Sign Language Recognizer...")

        self.data_store = np.empty((1,14))
        self.window_size = 19
        self.step_size = 10

        self.cooldown = 5
        
        # Load the model
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            self.model = SignLanguageModel()
            # Load weights
            state_dict = torch.load(model_path, map_location=device)
            self.model.load_state_dict(state_dict)
            # self.model.load_state_dict(torch.load(model_path, map_location=device))
            self.model.eval()
            
            print(f"Model loaded from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            sys.exit(1)
        
        # Load the scaler
        try:
            with open(encoder_path, "rb") as file:
                self.encoder = pickle.load(file)
            print(f"Scaler loaded from {encoder_path}")
        except Exception as e:
            print(f"Error loading scaler: {e}")
            sys.exit(1)

        # Recent predictions for smoothing
        self.recent_predictions = deque(maxlen=5)
        self.prediction_counter = {}

    def set_cooldown(self, amt):
        """Set the cooldown period between predictions"""
        self.cooldown = int(amt*10)
        self.recent_predictions = deque(maxlen=self.cooldown)
        
    def preprocess_data(self, raw_data):
        """Preprocess raw sensor data to match model input format"""
        self.data_store = np.vstack([np.array(raw_data[:14]), self.data_store])
        self.data_store = self.data_store[:self.window_size]

    def predict(self, raw_data):
        """Predict the sign from raw sensor data with confidence score"""
        if len(raw_data) < 14:
            return None, None
            
        # Preprocess the data
        self.preprocess_data(raw_data)

        if self.data_store.shape[0] < self.window_size:
            return None, None
        
        X = torch.tensor(np.expand_dims(self.data_store, axis=0), dtype=torch.float32)
        
        # Get prediction and confidence scores
        prediction = None
        confidence = None

        # Get the predicted class
        outputs = self.model(X)
        # Get predictions
        confidence, pred_class = torch.max(outputs, 1)
        confidence = confidence.item()
        confidence = (confidence if confidence != float('nan') else 0)/10

        prediction = self.encoder.inverse_transform(pred_class)[0]

        # Add to recent predictions
        self.recent_predictions.append(prediction)

        # # Smooth prediction using a voting mechanism
        # if len(self.recent_predictions) >= self.cooldown:
        #     prediction_counts = {}
        #     for pred in self.recent_predictions:
        #         prediction_counts[pred] = prediction_counts.get(pred, 0) + 1
            
        #     smoothed_prediction = max(prediction_counts, key=prediction_counts.get)
        #     smoothed_confidence = prediction_counts[smoothed_prediction] / len(self.recent_predictions)

        #     if smoothed_confidence > 0.5:
                # return smoothed_prediction, confidence * smoothed_confidence
        return prediction, confidence
        
        # return None, None

class BSLSignLanguageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("BSL Sign Language Recognition")
        self.root.geometry("800x600")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        
        # Set up variables
        self.recognizer = BSLRecognizer()
        self.isConnected = [False, False]
        self.handData = []
        self.hands = [True, True]  # Default to both hands
        self.running = False
        self.loop = None
        self.task = None
        self.shutdown_flag = False
        
        # Create main frame
        main_frame = ttk.Frame(root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create status frame
        status_frame = ttk.LabelFrame(main_frame, text="Connection Status", padding="10")
        status_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Add connection indicators
        self.left_status = ttk.Label(status_frame, text="Left Glove: Disconnected", foreground="red")
        self.left_status.grid(row=0, column=0, padx=5, pady=5, sticky="w")
        
        self.right_status = ttk.Label(status_frame, text="Right Glove: Disconnected", foreground="red")
        self.right_status.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        
        # Create prediction display
        prediction_frame = ttk.LabelFrame(main_frame, text="Current Sign", padding="10")
        prediction_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.prediction_label = ttk.Label(prediction_frame, text="", font=("Arial", 72))
        self.prediction_label.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Add confidence score indicator
        confidence_frame = ttk.Frame(prediction_frame)
        confidence_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(confidence_frame, text="Confidence:", font=("Arial", 14)).pack(side=tk.LEFT, padx=5)
        self.confidence_label = ttk.Label(confidence_frame, text="N/A", font=("Arial", 14, "bold"))
        self.confidence_label.pack(side=tk.LEFT, padx=5)
        
        # Add confidence progress bar
        self.confidence_bar = ttk.Progressbar(prediction_frame, orient="horizontal", length=300, mode="determinate")
        self.confidence_bar.pack(padx=20, pady=10)
        
        # Create settings frame
        settings_frame = ttk.LabelFrame(main_frame, text="Settings", padding="10")
        settings_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Add sensitivity slider
        ttk.Label(settings_frame, text="Prediction Sensitivity:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.sensitivity = tk.DoubleVar(value=1.0)
        sensitivity_slider = ttk.Scale(settings_frame, from_=0.1, to=3.0, variable=self.sensitivity, 
                                       orient=tk.HORIZONTAL, length=200, command=self.update_sensitivity)
        sensitivity_slider.grid(row=0, column=1, padx=5, pady=5)
        self.sensitivity_label = ttk.Label(settings_frame, text="1.0 seconds")
        self.sensitivity_label.grid(row=0, column=2, padx=5, pady=5)
        
        # Add confidence threshold slider
        ttk.Label(settings_frame, text="Confidence Threshold:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.conf_threshold = tk.DoubleVar(value=0.6)
        conf_slider = ttk.Scale(settings_frame, from_=0.1, to=1.0, variable=self.conf_threshold, 
                                orient=tk.HORIZONTAL, length=200, command=self.update_conf_threshold)
        conf_slider.grid(row=1, column=1, padx=5, pady=5)
        self.conf_threshold_label = ttk.Label(settings_frame, text="0.6")
        self.conf_threshold_label.grid(row=1, column=2, padx=5, pady=5)
        
        # Add glove selection
        ttk.Label(settings_frame, text="Gloves:").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        glove_frame = ttk.Frame(settings_frame)
        glove_frame.grid(row=2, column=1, padx=5, pady=5, sticky="w")
        
        self.left_var = tk.BooleanVar(value=True)
        self.right_var = tk.BooleanVar(value=True)
        
        left_check = ttk.Checkbutton(glove_frame, text="Left", variable=self.left_var)
        left_check.pack(side=tk.LEFT, padx=5)
        
        right_check = ttk.Checkbutton(glove_frame, text="Right", variable=self.right_var)
        right_check.pack(side=tk.LEFT, padx=5)
        
        # Create log frame
        log_frame = ttk.LabelFrame(main_frame, text="Recognition Log", padding="10")
        log_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=8)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        self.log_text.config(state=tk.DISABLED)
        
        # Create button frame
        button_frame = ttk.Frame(main_frame, padding="10")
        button_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.start_button = ttk.Button(button_frame, text="Start Recognition", command=self.start_recognition)
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = ttk.Button(button_frame, text="Stop Recognition", command=self.stop_recognition)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        self.stop_button.config(state=tk.DISABLED)
        
        ttk.Button(button_frame, text="Clear Log", command=self.clear_log).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Exit", command=self.on_close).pack(side=tk.RIGHT, padx=5)
        
        # Initialize log
        self.log("BSL Sign Language Recognition started")
        self.log("Please press 'Start Recognition' to begin")
    
    def update_conf_threshold(self, value):
        """Update the confidence threshold label"""
        value = float(value)
        self.conf_threshold_label.config(text=f"{value:.1f}")
    
    def update_sensitivity(self, value):
        """Update the sensitivity label and recognizer cooldown"""
        value = float(value)
        self.sensitivity_label.config(text=f"{value:.1f} seconds")
        if hasattr(self, 'recognizer'):
            self.recognizer.set_cooldown(value)
    
    def log(self, message):
        """Add a message to the log"""
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        log_message = f"[{timestamp}] {message}\n"
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, log_message)
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)
    
    def update_status(self):
        """Update the connection status display"""
        self.left_status.config(
            text=f"Left Glove: {'Connected' if self.isConnected[0] else 'Disconnected'}", 
            foreground="green" if self.isConnected[0] else "red"
        )
        self.right_status.config(
            text=f"Right Glove: {'Connected' if self.isConnected[1] else 'Disconnected'}", 
            foreground="green" if self.isConnected[1] else "red"
        )
    
    def start_recognition(self):
        """Start the recognition process"""
        self.running = True
        self.hands = [self.left_var.get(), self.right_var.get()]
        
        # Disable start button, enable stop button
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        
        # Log the action
        self.log(f"Starting recognition with Left={self.hands[0]}, Right={self.hands[1]}")
        
        # Start the recognition thread
        recognition_thread = threading.Thread(target=self.run_recognition)
        recognition_thread.daemon = True
        recognition_thread.start()
    
    def stop_recognition(self):
        """Stop the recognition process"""
        self.running = False
        self.shutdown_flag = True
        
        # Enable start button, disable stop button
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        
        # Log the action
        self.log("Recognition stopped")
    
    def clear_log(self):
        """Clear the log text"""
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete(1.0, tk.END)
        self.log_text.config(state=tk.DISABLED)
    
    def on_close(self):
        """Handle window closing"""
        self.shutdown_flag = True
        self.running = False
        self.root.destroy()
    
    def run_recognition(self):
        """Run the recognition process in a separate thread"""
        # Create new event loop for the thread
        asyncio.set_event_loop(asyncio.new_event_loop())
        loop = asyncio.get_event_loop()
        
        # Store the loop for later cleanup
        self.loop = loop
        
        try:
            # Run the main coroutine
            loop.run_until_complete(self.recognition_coroutine())
        except Exception as e:
            self.log(f"Error in recognition: {e}")
        finally:
            # Clean up the loop
            loop.close()
    
    async def recognition_coroutine(self):
        """Main coroutine for recognition"""
        # Reset connection and data
        self.isConnected = [False, False]
        self.handData = []
        
        # Create Arduino interpreter
        arduino_interpreter = ArduinoInterpreter(self.isConnected, self.handData)
        
        # Start the Arduino connection task
        arduino_task = asyncio.create_task(arduino_interpreter.run(self.hands))
        self.task = arduino_task
        
        # Last prediction for tracking changes
        last_prediction = ""
        
        # Process data and make predictions
        try:
            while self.running:
                # Update connection status in UI
                self.root.after(0, self.update_status)
                
                # Only attempt prediction if we have data
                if self.handData:
                    # Get the latest data
                    latest_data = self.handData[-1]
                    
                    # Make prediction with confidence score
                    prediction, confidence = self.recognizer.predict(latest_data)
                    
                    # If we got a prediction with sufficient confidence
                    if prediction and confidence is not None:
                        # Check if confidence meets threshold
                        threshold = self.conf_threshold.get()
                        
                        if confidence >= threshold:
                            if prediction != last_prediction:
                                last_prediction = prediction
                                
                                # Update the UI with prediction and confidence
                                self.root.after(0, lambda p=prediction, c=confidence: self.update_prediction(p, c))
                                
                                # Log the prediction with confidence
                                self.root.after(0, lambda p=prediction, c=confidence: 
                                               self.log(f"Recognized: {p} (Confidence: {c:.2f})"))
                        else:
                            # Update confidence but keep last prediction
                            self.root.after(0, lambda c=confidence: self.update_confidence(c))
                
                # Brief pause to reduce CPU usage
                await asyncio.sleep(0.1)
                
                # Check for shutdown flag
                if self.shutdown_flag:
                    break
        
        except Exception as e:
            self.log(f"Error: {e}")
        finally:
            # Cancel the Arduino task
            arduino_task.cancel()
            try:
                await arduino_task
            except asyncio.CancelledError:
                pass
            
            self.log("Recognition coroutine stopped")
    
    def update_prediction(self, prediction, confidence=None):
        """Update the prediction display with confidence"""
        self.prediction_label.config(text=prediction)
        self.update_confidence(confidence)
    
    def update_confidence(self, confidence):
        """Update only the confidence display"""
        if confidence is not None:
            # Update confidence label
            percentage = int(confidence * 100)
            self.confidence_label.config(text=f"{percentage}%")
            
            # Update confidence bar
            self.confidence_bar['value'] = percentage
            
            # Update color based on confidence level
            if confidence >= 0.8:
                self.confidence_label.config(foreground="green")
            elif confidence >= 0.6:
                self.confidence_label.config(foreground="orange")
            else:
                self.confidence_label.config(foreground="red")
        else:
            self.confidence_label.config(text="N/A", foreground="black")
            self.confidence_bar['value'] = 0


def main():
    """Main function to start the application"""
    root = tk.Tk()
    app = BSLSignLanguageApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()