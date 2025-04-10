import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import threading
import asyncio
import time
import datetime
import json
import os
import sys
import pygame
from io import BytesIO
import numpy as np
import boto3
from PIL import Image, ImageTk

# Import our custom modules
from loadEEG import EEGEmotionDetector
from load import BSLRecognizer
# from load import BSLRecognizerTorch as BSLRecognizer
from utils.ArduinoInterpreter import ArduinoInterpreter

class IntegratedCommunicationApp:
    """Main application that integrates BSL sign language recognition with EEG emotion detection"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("BSL & EEG Integrated Communication System")
        self.root.geometry("1200x800")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        
        # Initialize pygame for audio playback
        pygame.mixer.init()
        
        # Application state
        self.bsl_running = False
        self.eeg_running = False
        self.shutdown_flag = False
        self.bsl_recognizer = None
        self.eeg_detector = None
        self.bsl_task = None
        self.bsl_loop = None
        self.bsl_is_connected = [False, False]  # [left, right]
        self.bsl_hand_data = []
        self.bsl_hands = [True, True]  # [left, right]
        self.current_sign = ""
        self.current_emotion = "Neutral"
        self.emotion_confidence = 0.0
        self.sign_confidence = 0.0
        self.speech_gender = "female"
        self.use_selected_emotion_var = tk.BooleanVar(value=False)
        self.selected_emotion_var = tk.StringVar(value="")
        self.available_emotions = ["Happy", "Sad", "Angry", "Neutral"]
        
        # Abbreviation expansion feature
        self.auto_expand_enabled = True
        self.recent_signs = []
        self.max_recent_signs = 10  # Maximum number of signs to remember
        
        # Dictionary of common abbreviations and their expansions
        self.abbreviations = {
            "BRB": "Be right back",
            "LOL": "Laughing out loud",
            "OMG": "Oh my goodness",
            "IDK": "I don't know",
            "TY": "Thank you",
            "PLZ": "Please",
            "IMO": "In my opinion",
            "BTW": "By the way",
            "THX": "Thanks",
            "GG": "Good game",
            "WYD": "What are you doing",
            "HRU": "How are you",
            "GBU": "God bless you",
            "HBD": "Happy birthday",
            "GM": "Good morning",
            "GN": "Good night",
            "TTYL": "Talk to you later",
            "ILY": "I love you",
            "HI": "Hello",
            "OK": "Okay"
        }
        
        # Settings variables
        self.bsl_config = {
            "model_path": "models/bsl_model.joblib",
            "scaler_path": "models/bsl_scaler.joblib",
            "feature_names_path": "models/feature_names.joblib",
            "cooldown": 1.0,
            "confidence_threshold": 0.6
        }
        
        self.eeg_config = {
            "model_path": "models/eeg_model.pth",
            "com_port": "COM3",
            "baudrate": 57600
        }
        
        self.tts_config = {
            "enabled": True,
            "aws_region": "eu-west-2",
            "voice_male": "Brian",
            "voice_female": "Amy",
            "gender": "female"
        }
        
        # Create the main UI (this must come before using log or other UI elements)
        self.create_ui()
        
        # After UI creation, now we can setup Polly and log messages
        self.setup_polly()
        
        # Try to load saved settings
        self.load_settings()
        
        # Log startup
        self.log("Application started")
        self.log("Use the tabs to configure and start the BSL and EEG systems")
    
    def create_ui(self):
        """Create the main user interface"""
        # Make the window resizable
        self.root.resizable(True, True)
        
        # Create a main canvas with scrollbar for smaller screens
        self.main_canvas = tk.Canvas(self.root)
        self.main_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Add a scrollbar to the canvas
        main_scrollbar = ttk.Scrollbar(self.root, orient=tk.VERTICAL, command=self.main_canvas.yview)
        main_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Configure the canvas
        self.main_canvas.configure(yscrollcommand=main_scrollbar.set)
        self.main_canvas.bind('<Configure>', lambda e: self.main_canvas.configure(scrollregion=self.main_canvas.bbox("all")))
        
        # Create a frame inside the canvas for all content
        self.main_frame = ttk.Frame(self.main_canvas)
        self.main_canvas.create_window((0, 0), window=self.main_frame, anchor="nw", width=self.root.winfo_width())
        
        # Create the main notebook (tabbed interface)
        self.notebook = ttk.Notebook(self.main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create tabs
        self.main_tab = ttk.Frame(self.notebook)
        self.bsl_tab = ttk.Frame(self.notebook)
        self.eeg_tab = ttk.Frame(self.notebook)
        self.settings_tab = ttk.Frame(self.notebook)
        
        self.notebook.add(self.main_tab, text="Main Dashboard")
        self.notebook.add(self.bsl_tab, text="Sign Language")
        self.notebook.add(self.eeg_tab, text="Emotion Detection")
        self.notebook.add(self.settings_tab, text="Settings")
        
        # Set up each tab
        self.setup_main_tab()
        self.setup_bsl_tab()
        self.setup_eeg_tab()
        self.setup_settings_tab()
        
        # Add mouse wheel bindings
        self.bind_mousewheel_to_widgets()
        
        # Configure window to dynamically adjust size
        self.configure_dynamic_sizing()
    
    def configure_dynamic_sizing(self):
        """Configure window to adjust to screen size"""
        # Get screen dimensions
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        
        # Set window size to 90% of screen size
        window_width = int(screen_width * 0.9)
        window_height = int(screen_height * 0.9)
        
        # For very small displays, ensure minimum size
        min_width = min(800, screen_width - 50)
        min_height = min(600, screen_height - 50)
        
        window_width = max(window_width, min_width)
        window_height = max(window_height, min_height)
        
        # Set geometry
        self.root.geometry(f"{window_width}x{window_height}")
        
        # Set minsize to ensure window can't be made too small
        self.root.minsize(min_width, min_height)
        
        # Update canvas size
        self.main_canvas.configure(width=window_width, height=window_height)
    
    def bind_mousewheel_to_widgets(self):
        """Bind mousewheel scrolling to all widgets"""
        def _on_mousewheel(event):
            self.main_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        # Bind to canvas
        self.main_canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        # Bind to common OS X mouse wheel event
        self.main_canvas.bind_all("<Button-4>", lambda e: self.main_canvas.yview_scroll(-1, "units"))
        self.main_canvas.bind_all("<Button-5>", lambda e: self.main_canvas.yview_scroll(1, "units"))
        
        # Update canvas scroll region when notebook tab changes
        self.notebook.bind("<<NotebookTabChanged>>", lambda e: self.root.after(100, self.update_scroll_region))
    
    def update_scroll_region(self):
        """Update the canvas scroll region"""
        self.main_canvas.update_idletasks()
        self.main_canvas.configure(scrollregion=self.main_canvas.bbox("all"))
        
        # Ensure canvas width matches window width
        self.main_canvas.itemconfig(self.main_canvas.find_withtag("all")[0], width=self.root.winfo_width() - 20)
    
    def setup_main_tab(self):
        """Set up the main dashboard tab"""
        # Create a canvas with scrollbar for the main tab
        main_canvas = tk.Canvas(self.main_tab)
        main_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Add a scrollbar to the canvas
        #tab_scrollbar = ttk.Scrollbar(self.main_tab, orient=tk.VERTICAL, command=main_canvas.yview)
        #tab_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Configure the canvas
        #main_canvas.configure(yscrollcommand=tab_scrollbar.set)
        main_canvas.bind('<Configure>', lambda e: main_canvas.configure(scrollregion=main_canvas.bbox("all")))
        
        # Create a frame inside the canvas for all content
        content_frame = ttk.Frame(main_canvas)
        main_canvas.create_window((0, 0), window=content_frame, anchor="nw", width=1000)
        
        
        # Create frames
        top_frame = ttk.Frame(content_frame, padding="10")
        top_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # System status 
        status_frame = ttk.LabelFrame(top_frame, text="System Status", padding="10")
        status_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # BSL Status indicator
        self.bsl_status_frame = ttk.Frame(status_frame)
        self.bsl_status_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(self.bsl_status_frame, text="Sign Language:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.bsl_status_indicator = ttk.Label(self.bsl_status_frame, text="Inactive", foreground="red")
        self.bsl_status_indicator.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        
        # Left/Right glove indicators
        self.left_glove_indicator = ttk.Label(self.bsl_status_frame, text="Left Glove: Disconnected", foreground="red")
        self.left_glove_indicator.grid(row=0, column=2, padx=5, pady=5, sticky="w")
        
        self.right_glove_indicator = ttk.Label(self.bsl_status_frame, text="Right Glove: Disconnected", foreground="red")
        self.right_glove_indicator.grid(row=0, column=3, padx=5, pady=5, sticky="w")
        
        # EEG Status indicator
        self.eeg_status_frame = ttk.Frame(status_frame)
        self.eeg_status_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(self.eeg_status_frame, text="Emotion Detection:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.eeg_status_indicator = ttk.Label(self.eeg_status_frame, text="Inactive", foreground="red")
        self.eeg_status_indicator.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        
        # TTS Status indicator
        self.tts_status_frame = ttk.Frame(status_frame)
        self.tts_status_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(self.tts_status_frame, text="Text-to-Speech:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.tts_status_indicator = ttk.Label(self.tts_status_frame, text="Ready (Female Voice)", foreground="green")
        self.tts_status_indicator.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        
        # Toggle buttons
        toggle_frame = ttk.Frame(top_frame)
        toggle_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.bsl_toggle_button = ttk.Button(toggle_frame, text="Start Sign Language", command=self.toggle_bsl)
        self.bsl_toggle_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.eeg_toggle_button = ttk.Button(toggle_frame, text="Start Emotion Detection", command=self.toggle_eeg)
        self.eeg_toggle_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Output frame - shows the recognized sign and detected emotion
        output_frame = ttk.LabelFrame(content_frame, text="Recognition Output", padding="10")
        output_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Sign language output
        sign_output_frame = ttk.Frame(output_frame)
        sign_output_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        ttk.Label(sign_output_frame, text="Recognized Sign:", font=("Arial", 14)).pack(anchor=tk.W, padx=5, pady=5)
        self.sign_label = ttk.Label(sign_output_frame, text="Waiting...", font=("Arial", 48, "bold"))
        self.sign_label.pack(padx=20, pady=10)
        
        # Sign confidence
        sign_conf_frame = ttk.Frame(sign_output_frame)
        sign_conf_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(sign_conf_frame, text="Confidence:", font=("Arial", 12)).pack(side=tk.LEFT, padx=5)
        self.sign_confidence_label = ttk.Label(sign_conf_frame, text="N/A", font=("Arial", 12, "bold"))
        self.sign_confidence_label.pack(side=tk.LEFT, padx=5)
        
        self.sign_confidence_bar = ttk.Progressbar(sign_output_frame, orient="horizontal", length=300, mode="determinate")
        self.sign_confidence_bar.pack(padx=20, pady=5)
        
        # Emotion output
        emotion_output_frame = ttk.Frame(output_frame)
        emotion_output_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        ttk.Label(emotion_output_frame, text="Detected Emotion:", font=("Arial", 14)).pack(anchor=tk.W, padx=5, pady=5)
        self.emotion_label = ttk.Label(emotion_output_frame, text="Neutral", font=("Arial", 48, "bold"))
        self.emotion_label.pack(padx=20, pady=10)
        
        # Emotion confidence
        emotion_conf_frame = ttk.Frame(emotion_output_frame)
        emotion_conf_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(emotion_conf_frame, text="Confidence:", font=("Arial", 12)).pack(side=tk.LEFT, padx=5)
        self.emotion_confidence_label = ttk.Label(emotion_conf_frame, text="N/A", font=("Arial", 12, "bold"))
        self.emotion_confidence_label.pack(side=tk.LEFT, padx=5)
        
        self.emotion_confidence_bar = ttk.Progressbar(emotion_output_frame, orient="horizontal", length=300, mode="determinate")
        self.emotion_confidence_bar.pack(padx=20, pady=5)
        
        # Speak button
        speak_frame = ttk.Frame(output_frame)
        speak_frame.pack(fill=tk.X, padx=5, pady=15)
        
        self.speak_button = ttk.Button(speak_frame, text="Speak Current Sign with Emotion", command=self.speak_current_sign)
        self.speak_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Text input field for custom speech
        ttk.Label(speak_frame, text="Or enter custom text:").pack(side=tk.LEFT, padx=10)
        self.custom_text = tk.StringVar()
        emotion_frame = ttk.Frame(speak_frame)
        emotion_frame.pack(side=tk.LEFT, padx=5, pady=5)

        self.use_selected_emotion_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(emotion_frame, text="Use selected emotion:", variable=self.use_selected_emotion_var).pack(side=tk.LEFT)

        self.selected_emotion_var = tk.StringVar(value="Happy")
        emotion_dropdown = ttk.Combobox(emotion_frame, textvariable=self.selected_emotion_var, 
                                    values=self.available_emotions, width=10, state="readonly")
        emotion_dropdown.pack(side=tk.LEFT, padx=5)
        custom_entry = ttk.Entry(speak_frame, textvariable=self.custom_text, width=40)
        custom_entry.pack(side=tk.LEFT, padx=5, pady=5)
        
        ttk.Button(speak_frame, text="Speak Custom Text", command=self.speak_custom_text).pack(side=tk.LEFT, padx=5, pady=5)
        
        # Log frame
        log_frame = ttk.LabelFrame(content_frame, text="Activity Log", padding="10")
        log_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=8)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        self.log_text.config(state=tk.DISABLED)
        
        # Create button frame at the bottom
        button_frame = ttk.Frame(content_frame, padding="10")
        button_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(button_frame, text="Clear Log", command=self.clear_log).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Save Log", command=self.save_log).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Exit", command=self.on_close).pack(side=tk.RIGHT, padx=5)

        self.root.update_idletasks()
        
        # Bind mousewheel to this canvas
        # def _on_mousewheel(event):
        #     main_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        # main_canvas.bind_all("<MouseWheel>", _on_mousewheel)
        # main_canvas.bind_all("<Button-4>", lambda e: main_canvas.yview_scroll(-1, "units"))
        # main_canvas.bind_all("<Button-5>", lambda e: main_canvas.yview_scroll(1, "units"))
        
        # # Update scroll region after window sizing
        # self.root.after(100, lambda: main_canvas.configure(scrollregion=main_canvas.bbox("all")))
    
    def setup_bsl_tab(self):
        """Set up the BSL sign language tab"""
        # Create model settings frame
        model_frame = ttk.LabelFrame(self.bsl_tab, text="BSL Model Settings", padding="10")
        model_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Model path
        ttk.Label(model_frame, text="Model Path:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.bsl_model_path_var = tk.StringVar(value=self.bsl_config["model_path"])
        ttk.Entry(model_frame, textvariable=self.bsl_model_path_var, width=50).grid(row=0, column=1, padx=5, pady=5, sticky="w")
        ttk.Button(model_frame, text="Browse...", command=lambda: self.browse_file(self.bsl_model_path_var)).grid(row=0, column=2, padx=5, pady=5)
        
        # Scaler path
        ttk.Label(model_frame, text="Scaler Path:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.bsl_scaler_path_var = tk.StringVar(value=self.bsl_config["scaler_path"])
        ttk.Entry(model_frame, textvariable=self.bsl_scaler_path_var, width=50).grid(row=1, column=1, padx=5, pady=5, sticky="w")
        ttk.Button(model_frame, text="Browse...", command=lambda: self.browse_file(self.bsl_scaler_path_var)).grid(row=1, column=2, padx=5, pady=5)
        
        # Feature names path
        ttk.Label(model_frame, text="Feature Names Path:").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.bsl_feature_names_path_var = tk.StringVar(value=self.bsl_config["feature_names_path"])
        ttk.Entry(model_frame, textvariable=self.bsl_feature_names_path_var, width=50).grid(row=2, column=1, padx=5, pady=5, sticky="w")
        ttk.Button(model_frame, text="Browse...", command=lambda: self.browse_file(self.bsl_feature_names_path_var)).grid(row=2, column=2, padx=5, pady=5)
        
        # BSL Recognition settings
        settings_frame = ttk.LabelFrame(self.bsl_tab, text="Recognition Settings", padding="10")
        settings_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Prediction cooldown slider
        ttk.Label(settings_frame, text="Prediction Cooldown:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.bsl_cooldown_var = tk.DoubleVar(value=self.bsl_config["cooldown"])
        cooldown_slider = ttk.Scale(settings_frame, from_=0.1, to=3.0, variable=self.bsl_cooldown_var, 
                                   orient=tk.HORIZONTAL, length=200, command=self.update_bsl_cooldown)
        cooldown_slider.grid(row=0, column=1, padx=5, pady=5)
        self.bsl_cooldown_label = ttk.Label(settings_frame, text=f"{self.bsl_config['cooldown']:.1f} seconds")
        self.bsl_cooldown_label.grid(row=0, column=2, padx=5, pady=5)
        
        # Confidence threshold slider
        ttk.Label(settings_frame, text="Confidence Threshold:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.bsl_confidence_threshold_var = tk.DoubleVar(value=self.bsl_config["confidence_threshold"])
        conf_slider = ttk.Scale(settings_frame, from_=0.1, to=1.0, variable=self.bsl_confidence_threshold_var, 
                              orient=tk.HORIZONTAL, length=200, command=self.update_bsl_confidence_threshold)
        conf_slider.grid(row=1, column=1, padx=5, pady=5)
        self.bsl_confidence_threshold_label = ttk.Label(settings_frame, text=f"{self.bsl_config['confidence_threshold']:.1f}")
        self.bsl_confidence_threshold_label.grid(row=1, column=2, padx=5, pady=5)
        
        # Glove selection
        ttk.Label(settings_frame, text="Gloves:").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        glove_frame = ttk.Frame(settings_frame)
        glove_frame.grid(row=2, column=1, padx=5, pady=5, sticky="w")
        
        self.left_glove_var = tk.BooleanVar(value=True)
        self.right_glove_var = tk.BooleanVar(value=True)
        
        left_check = ttk.Checkbutton(glove_frame, text="Left", variable=self.left_glove_var)
        left_check.pack(side=tk.LEFT, padx=5)
        
        right_check = ttk.Checkbutton(glove_frame, text="Right", variable=self.right_glove_var)
        right_check.pack(side=tk.LEFT, padx=5)
        
        # Control buttons
        control_frame = ttk.Frame(self.bsl_tab, padding="10")
        control_frame.pack(fill=tk.X, padx=5, pady=15)
        
        self.bsl_start_button = ttk.Button(control_frame, text="Start Sign Language Recognition", command=self.start_bsl)
        self.bsl_start_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.bsl_stop_button = ttk.Button(control_frame, text="Stop Sign Language Recognition", command=self.stop_bsl)
        self.bsl_stop_button.pack(side=tk.LEFT, padx=5, pady=5)
        self.bsl_stop_button.config(state=tk.DISABLED)
        
        # Status display
        status_display_frame = ttk.LabelFrame(self.bsl_tab, text="Status", padding="10")
        status_display_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.bsl_status_text = scrolledtext.ScrolledText(status_display_frame, height=10)
        self.bsl_status_text.pack(fill=tk.BOTH, expand=True)
        self.bsl_status_text.config(state=tk.DISABLED)
    
    def setup_eeg_tab(self):
        """Set up the EEG emotion detection tab"""
        # Create model settings frame
        model_frame = ttk.LabelFrame(self.eeg_tab, text="EEG Model Settings", padding="10")
        model_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Model path
        ttk.Label(model_frame, text="Model Path:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.eeg_model_path_var = tk.StringVar(value=self.eeg_config["model_path"])
        ttk.Entry(model_frame, textvariable=self.eeg_model_path_var, width=50).grid(row=0, column=1, padx=5, pady=5, sticky="w")
        ttk.Button(model_frame, text="Browse...", command=lambda: self.browse_file(self.eeg_model_path_var)).grid(row=0, column=2, padx=5, pady=5)
        
        # Connection settings
        connection_frame = ttk.LabelFrame(self.eeg_tab, text="Connection Settings", padding="10")
        connection_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # COM port
        ttk.Label(connection_frame, text="COM Port:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.eeg_com_port_var = tk.StringVar(value=self.eeg_config["com_port"])
        ttk.Entry(connection_frame, textvariable=self.eeg_com_port_var, width=10).grid(row=0, column=1, padx=5, pady=5, sticky="w")
        
        # Baud rate
        ttk.Label(connection_frame, text="Baud Rate:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.eeg_baud_rate_var = tk.IntVar(value=self.eeg_config["baudrate"])
        ttk.Combobox(connection_frame, textvariable=self.eeg_baud_rate_var, values=[9600, 19200, 38400, 57600, 115200], width=10).grid(row=1, column=1, padx=5, pady=5, sticky="w")
        
        # Calibration frame
        calibration_frame = ttk.LabelFrame(self.eeg_tab, text="Calibration", padding="10")
        calibration_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(calibration_frame, text="Calibration ensures accurate emotion detection. Sit still and relax during calibration.").pack(pady=5)
        
        self.calibrate_button = ttk.Button(calibration_frame, text="Calibrate Neutral State (20 seconds)", command=self.calibrate_eeg)
        self.calibrate_button.pack(padx=5, pady=10)
        
        # Control buttons
        control_frame = ttk.Frame(self.eeg_tab, padding="10")
        control_frame.pack(fill=tk.X, padx=5, pady=15)
        
        self.eeg_start_button = ttk.Button(control_frame, text="Start Emotion Detection", command=self.start_eeg)
        self.eeg_start_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.eeg_stop_button = ttk.Button(control_frame, text="Stop Emotion Detection", command=self.stop_eeg)
        self.eeg_stop_button.pack(side=tk.LEFT, padx=5, pady=5)
        self.eeg_stop_button.config(state=tk.DISABLED)
        
        # Status display
        status_display_frame = ttk.LabelFrame(self.eeg_tab, text="Status", padding="10")
        status_display_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.eeg_status_text = scrolledtext.ScrolledText(status_display_frame, height=10)
        self.eeg_status_text.pack(fill=tk.BOTH, expand=True)
        self.eeg_status_text.config(state=tk.DISABLED)
    
    def setup_settings_tab(self):
        """Set up the general settings tab"""
        # Text-to-Speech settings
        tts_frame = ttk.LabelFrame(self.settings_tab, text="Text-to-Speech Settings", padding="10")
        tts_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Enable TTS - using grid for all children of tts_frame for consistency
        self.tts_enabled_var = tk.BooleanVar(value=self.tts_config["enabled"])
        ttk.Checkbutton(tts_frame, text="Enable Text-to-Speech", variable=self.tts_enabled_var, 
                       command=self.update_tts_status).grid(row=0, column=0, columnspan=2, padx=5, pady=5, sticky="w")
        
        # AWS Region
        ttk.Label(tts_frame, text="AWS Region:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.aws_region_var = tk.StringVar(value=self.tts_config["aws_region"])
        regions = ["us-east-1", "us-east-2", "us-west-1", "us-west-2", "eu-west-1", "eu-west-2", "eu-central-1"]
        ttk.Combobox(tts_frame, textvariable=self.aws_region_var, values=regions, width=15).grid(row=1, column=1, padx=5, pady=5, sticky="w")
        
        # Voice selection
        ttk.Label(tts_frame, text="Voice Gender:").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        
        voice_frame = ttk.Frame(tts_frame)
        voice_frame.grid(row=2, column=1, padx=5, pady=5, sticky="w")
        
        self.voice_gender_var = tk.StringVar(value=self.tts_config["gender"])
        ttk.Radiobutton(voice_frame, text="Female", variable=self.voice_gender_var, value="female").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(voice_frame, text="Male", variable=self.voice_gender_var, value="male").pack(side=tk.LEFT, padx=5)
        
        # Female voice selection
        ttk.Label(tts_frame, text="Female Voice:").grid(row=3, column=0, padx=5, pady=5, sticky="w")
        self.female_voice_var = tk.StringVar(value=self.tts_config["voice_female"])
        female_voices = ["Amy", "Emma", "Joanna", "Kimberly", "Salli"]
        ttk.Combobox(tts_frame, textvariable=self.female_voice_var, values=female_voices, width=15).grid(row=3, column=1, padx=5, pady=5, sticky="w")
        
        # Male voice selection
        ttk.Label(tts_frame, text="Male Voice:").grid(row=4, column=0, padx=5, pady=5, sticky="w")
        self.male_voice_var = tk.StringVar(value=self.tts_config["voice_male"])
        male_voices = ["Brian", "Matthew", "Joey", "Justin"]
        ttk.Combobox(tts_frame, textvariable=self.male_voice_var, values=male_voices, width=15).grid(row=4, column=1, padx=5, pady=5, sticky="w")
        
        # Test TTS
        ttk.Button(tts_frame, text="Test Voice", command=self.test_tts).grid(row=5, column=0, columnspan=2, padx=5, pady=10)
        
        # AWS credentials
        aws_frame = ttk.LabelFrame(self.settings_tab, text="AWS Credentials", padding="10")
        aws_frame.pack(fill=tk.X, padx=5, pady=10)
        
        ttk.Label(aws_frame, text="Access Key:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.aws_access_key_var = tk.StringVar()
        ttk.Entry(aws_frame, textvariable=self.aws_access_key_var, width=40, show="*").grid(row=0, column=1, padx=5, pady=5, sticky="w")
        
        ttk.Label(aws_frame, text="Secret Key:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.aws_secret_key_var = tk.StringVar()
        ttk.Entry(aws_frame, textvariable=self.aws_secret_key_var, width=40, show="*").grid(row=1, column=1, padx=5, pady=5, sticky="w")
        
        ttk.Button(aws_frame, text="Save Credentials", command=self.save_aws_credentials).grid(row=2, column=0, columnspan=2, padx=5, pady=10)
        
        # Abbreviation Settings Frame
        abbrev_frame = ttk.LabelFrame(self.settings_tab, text="Abbreviation Expansion Settings", padding="10")
        abbrev_frame.pack(fill=tk.X, padx=5, pady=10)
        
        # Enable Auto-Expand
        self.auto_expand_var = tk.BooleanVar(value=self.auto_expand_enabled)
        ttk.Checkbutton(abbrev_frame, text="Enable Abbreviation Auto-Expansion", 
                        variable=self.auto_expand_var,
                        command=self.toggle_auto_expand).grid(row=0, column=0, columnspan=2, padx=5, pady=5, sticky="w")
        
        ttk.Label(abbrev_frame, text="Current Abbreviations:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        
        # Abbreviation list
        abbrev_list_frame = ttk.Frame(abbrev_frame)
        abbrev_list_frame.grid(row=2, column=0, columnspan=2, padx=5, pady=5, sticky="nsew")
        
        # Create a scrolled text area to show abbreviations
        self.abbrev_text = scrolledtext.ScrolledText(abbrev_list_frame, height=6, width=50)
        self.abbrev_text.pack(fill=tk.BOTH, expand=True)
        self.abbrev_text.config(state=tk.NORMAL)
        
        # Update the abbreviation text area with current abbreviations
        self.update_abbreviation_display()
        
        # Add new abbreviation
        add_frame = ttk.Frame(abbrev_frame)
        add_frame.grid(row=3, column=0, columnspan=2, padx=5, pady=5, sticky="w")
        
        ttk.Label(add_frame, text="Add New:").pack(side=tk.LEFT, padx=5, pady=5)
        
        ttk.Label(add_frame, text="Abbreviation:").pack(side=tk.LEFT, padx=5, pady=5)
        self.new_abbrev_var = tk.StringVar()
        ttk.Entry(add_frame, textvariable=self.new_abbrev_var, width=10).pack(side=tk.LEFT, padx=5, pady=5)
        
        ttk.Label(add_frame, text="Expansion:").pack(side=tk.LEFT, padx=5, pady=5)
        self.new_expansion_var = tk.StringVar()
        ttk.Entry(add_frame, textvariable=self.new_expansion_var, width=30).pack(side=tk.LEFT, padx=5, pady=5)
        
        ttk.Button(add_frame, text="Add", command=self.add_abbreviation).pack(side=tk.LEFT, padx=5, pady=5)
        
        # Application settings
        app_frame = ttk.LabelFrame(self.settings_tab, text="Application Settings", padding="10")
        app_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Theme selection
        ttk.Label(app_frame, text="Theme:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.theme_var = tk.StringVar(value="Default")
        themes = ["Default", "Light", "Dark"]
        ttk.Combobox(app_frame, textvariable=self.theme_var, values=themes, width=15, state="readonly").grid(row=0, column=1, padx=5, pady=5, sticky="w")
        
        # Action buttons
        button_frame = ttk.Frame(self.settings_tab, padding="10")
        button_frame.pack(fill=tk.X, padx=5, pady=15)
        
        ttk.Button(button_frame, text="Save Settings", command=self.save_settings).pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Button(button_frame, text="Reset to Defaults", command=self.reset_settings).pack(side=tk.LEFT, padx=5, pady=5)
    
    def setup_polly(self):
        """Setup the AWS Polly client"""
        # Define a simple console print function to use before UI is set up
        def console_log(message):
            print(f"[AWS Polly] {message}")
            
        # Try to get credentials from environment or credentials file first
        try:
            # Create a session to check if credentials are available
            session = boto3.Session()
            credentials = session.get_credentials()
            
            if credentials and credentials.access_key and credentials.secret_key:
                if hasattr(self, 'log_text'):
                    self.log("AWS credentials found in environment or credentials file")
                else:
                    console_log("AWS credentials found in environment or credentials file")
                self.polly = boto3.client('polly', region_name=self.tts_config["aws_region"])
                return
        except Exception as e:
            error_msg = f"AWS session initialization error: {e}"
            if hasattr(self, 'log_text'):
                self.log(error_msg)
            else:
                console_log(error_msg)
        
        # Try to read from our own credentials file
        try:
            if os.path.exists('aws_credentials.json'):
                with open('aws_credentials.json', 'r') as file:
                    creds = json.load(file)
                    
                    if 'access_key' in creds and 'secret_key' in creds:
                        if hasattr(self, 'log_text'):
                            self.log("AWS credentials loaded from aws_credentials.json")
                        else:
                            console_log("AWS credentials loaded from aws_credentials.json")
                        self.polly = boto3.client(
                            'polly', 
                            region_name=self.tts_config["aws_region"],
                            aws_access_key_id=creds['access_key'],
                            aws_secret_access_key=creds['secret_key']
                        )
                        return
        except Exception as e:
            error_msg = f"Error loading AWS credentials file: {e}"
            if hasattr(self, 'log_text'):
                self.log(error_msg)
            else:
                console_log(error_msg)
        
        # If still no credentials, create client without them (will fail unless credentials are provided later)
        info_msg = "No AWS credentials found. Please enter them in Settings tab."
        if hasattr(self, 'log_text'):
            self.log(info_msg)
        else:
            console_log(info_msg)
            
        try:
            self.polly = boto3.client('polly', region_name=self.tts_config["aws_region"])
        except Exception as e:
            error_msg = f"Error creating Polly client: {e}"
            if hasattr(self, 'log_text'):
                self.log(error_msg)
            else:
                console_log(error_msg)
                
            self.polly = None
    
    def log(self, message):
        """Add a message to the log"""
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        log_message = f"[{timestamp}] {message}\n"
        
        # Update the main log
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, log_message)
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)
    
    def bsl_log(self, message):
        """Add a message to the BSL log"""
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        log_message = f"[{timestamp}] {message}\n"
        
        # Update the BSL log
        self.bsl_status_text.config(state=tk.NORMAL)
        self.bsl_status_text.insert(tk.END, log_message)
        self.bsl_status_text.see(tk.END)
        self.bsl_status_text.config(state=tk.DISABLED)
        
        # Also add to main log
        self.log(f"BSL: {message}")
    
    def eeg_log(self, message):
        """Add a message to the EEG log"""
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        log_message = f"[{timestamp}] {message}\n"
        
        # Update the EEG log
        self.eeg_status_text.config(state=tk.NORMAL)
        self.eeg_status_text.insert(tk.END, log_message)
        self.eeg_status_text.see(tk.END)
        self.eeg_status_text.config(state=tk.DISABLED)
        
        # Also add to main log
        self.log(f"EEG: {message}")
    
    def clear_log(self):
        """Clear the log text"""
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete(1.0, tk.END)
        self.log_text.config(state=tk.DISABLED)
    
    def save_log(self):
        """Save the log to a file"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"log_{timestamp}.txt"
        
        try:
            with open(filename, 'w') as file:
                file.write(self.log_text.get(1.0, tk.END))
            self.log(f"Log saved to {filename}")
        except Exception as e:
            self.log(f"Error saving log: {e}")
    
    def update_bsl_cooldown(self, value):
        """Update the BSL cooldown value and label"""
        value = float(value)
        self.bsl_cooldown_label.config(text=f"{value:.1f} seconds")
        self.bsl_config["cooldown"] = value
        
        # Update the recognizer if it exists
        if hasattr(self, 'bsl_recognizer') and self.bsl_recognizer is not None:
            self.bsl_recognizer.set_cooldown(value)
    
    def update_bsl_confidence_threshold(self, value):
        """Update the BSL confidence threshold value and label"""
        value = float(value)
        self.bsl_confidence_threshold_label.config(text=f"{value:.1f}")
        self.bsl_config["confidence_threshold"] = value
    
    def update_tts_status(self):
        """Update the TTS status indicator based on the enabled state"""
        if self.tts_enabled_var.get():
            gender = self.voice_gender_var.get().capitalize()
            self.tts_status_indicator.config(text=f"Ready ({gender} Voice)", foreground="green")
        else:
            self.tts_status_indicator.config(text="Disabled", foreground="red")
    
    def browse_file(self, string_var):
        """Open a file browser and set the selected file path to the StringVar"""
        filename = filedialog.askopenfilename()
        if filename:
            string_var.set(filename)
    
    def save_aws_credentials(self):
        """Save AWS credentials to a file and update the Polly client"""
        access_key = self.aws_access_key_var.get().strip()
        secret_key = self.aws_secret_key_var.get().strip()
        
        if not access_key or not secret_key:
            messagebox.showerror("Error", "Both Access Key and Secret Key are required")
            return
            
        credentials = {
            "access_key": access_key,
            "secret_key": secret_key
        }
        
        try:
            # Save to file
            with open('aws_credentials.json', 'w') as file:
                json.dump(credentials, file)
            
            # Re-initialize the Polly client with explicit credentials
            try:
                self.polly = boto3.client(
                    'polly', 
                    region_name=self.aws_region_var.get(),
                    aws_access_key_id=credentials['access_key'],
                    aws_secret_access_key=credentials['secret_key']
                )
                # Test the credentials right away
                self.polly.describe_voices(LanguageCode="en-GB", Engine="standard")
                
                messagebox.showinfo("Success", "AWS credentials saved and verified successfully")
                self.log("AWS credentials updated and verified")
            except Exception as e:
                messagebox.showerror("AWS Error", f"Credentials saved but AWS reported an error: {str(e)}\n\nPlease check if the credentials are valid.")
                self.log(f"Error using AWS credentials: {e}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save credentials: {e}")
            self.log(f"Error saving AWS credentials: {e}")
    
    def save_settings(self):
        """Save all settings to a file"""
        # Update config dictionaries from UI values
        self.bsl_config["model_path"] = self.bsl_model_path_var.get()
        self.bsl_config["scaler_path"] = self.bsl_scaler_path_var.get()
        self.bsl_config["feature_names_path"] = self.bsl_feature_names_path_var.get()
        self.bsl_config["cooldown"] = self.bsl_cooldown_var.get()
        self.bsl_config["confidence_threshold"] = self.bsl_confidence_threshold_var.get()
        
        self.eeg_config["model_path"] = self.eeg_model_path_var.get()
        self.eeg_config["com_port"] = self.eeg_com_port_var.get()
        self.eeg_config["baudrate"] = self.eeg_baud_rate_var.get()
        
        self.tts_config["enabled"] = self.tts_enabled_var.get()
        self.tts_config["aws_region"] = self.aws_region_var.get()
        self.tts_config["voice_male"] = self.male_voice_var.get()
        self.tts_config["voice_female"] = self.female_voice_var.get()
        self.tts_config["gender"] = self.voice_gender_var.get()
        
        # Update abbreviation setting
        self.auto_expand_enabled = self.auto_expand_var.get()
        
        # Combine all settings
        settings = {
            "bsl_config": self.bsl_config,
            "eeg_config": self.eeg_config,
            "tts_config": self.tts_config,
            "theme": self.theme_var.get(),
            "auto_expand_enabled": self.auto_expand_enabled,
            "abbreviations": self.abbreviations
        }
        
        try:
            with open('settings.json', 'w') as file:
                json.dump(settings, file, indent=4)
            
            messagebox.showinfo("Success", "Settings saved successfully")
            self.log("Settings saved to settings.json")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save settings: {e}")
            self.log(f"Error saving settings: {e}")
    
    def load_settings(self):
        """Load settings from a file"""
        try:
            if os.path.exists('settings.json'):
                with open('settings.json', 'r') as file:
                    settings = json.load(file)
                
                # Update config dictionaries
                if "bsl_config" in settings:
                    self.bsl_config.update(settings["bsl_config"])
                    self.bsl_model_path_var.set(self.bsl_config["model_path"])
                    self.bsl_scaler_path_var.set(self.bsl_config["scaler_path"])
                    self.bsl_feature_names_path_var.set(self.bsl_config["feature_names_path"])
                    self.bsl_cooldown_var.set(self.bsl_config["cooldown"])
                    self.bsl_confidence_threshold_var.set(self.bsl_config["confidence_threshold"])
                    self.bsl_cooldown_label.config(text=f"{self.bsl_config['cooldown']:.1f} seconds")
                    self.bsl_confidence_threshold_label.config(text=f"{self.bsl_config['confidence_threshold']:.1f}")
                
                if "eeg_config" in settings:
                    self.eeg_config.update(settings["eeg_config"])
                    self.eeg_model_path_var.set(self.eeg_config["model_path"])
                    self.eeg_com_port_var.set(self.eeg_config["com_port"])
                    self.eeg_baud_rate_var.set(self.eeg_config["baudrate"])
                
                if "tts_config" in settings:
                    self.tts_config.update(settings["tts_config"])
                    self.tts_enabled_var.set(self.tts_config["enabled"])
                    self.aws_region_var.set(self.tts_config["aws_region"])
                    self.male_voice_var.set(self.tts_config["voice_male"])
                    self.female_voice_var.set(self.tts_config["voice_female"])
                    self.voice_gender_var.set(self.tts_config["gender"])
                    self.update_tts_status()
                
                if "theme" in settings:
                    self.theme_var.set(settings["theme"])
                
                # Load abbreviation settings
                if "auto_expand_enabled" in settings:
                    self.auto_expand_enabled = settings["auto_expand_enabled"]
                    if hasattr(self, 'auto_expand_var'):
                        self.auto_expand_var.set(self.auto_expand_enabled)
                
                if "abbreviations" in settings:
                    self.abbreviations = settings["abbreviations"]
                    if hasattr(self, 'update_abbreviation_display'):
                        self.update_abbreviation_display()
                
                self.log("Settings loaded from settings.json")
                return True
        except Exception as e:
            self.log(f"Error loading settings: {e}")
        
        return False
    
    def reset_settings(self):
        """Reset all settings to defaults"""
        if messagebox.askyesno("Confirm Reset", "Are you sure you want to reset all settings to defaults?"):
            # Default BSL settings
            self.bsl_config = {
                "model_path": "models/bsl_model.joblib",
                "scaler_path": "models/bsl_scaler.joblib",
                "feature_names_path": "models/feature_names.joblib",
                "cooldown": 1.0,
                "confidence_threshold": 0.6
            }
            
            # Default EEG settings
            self.eeg_config = {
                "model_path": "models/eeg_model.pth",
                "com_port": "COM3",
                "baudrate": 57600
            }
            
            # Default TTS settings
            self.tts_config = {
                "enabled": True,
                "aws_region": "eu-west-2",
                "voice_male": "Brian",
                "voice_female": "Amy",
                "gender": "female"
            }
            
            # Reset abbreviation settings
            self.auto_expand_enabled = True
            self.abbreviations = {
                "BRB": "Be right back",
                "LOL": "Laughing out loud",
                "OMG": "Oh my goodness",
                "IDK": "I don't know",
                "TY": "Thank you",
                "PLZ": "Please",
                "IMO": "In my opinion",
                "BTW": "By the way",
                "THX": "Thanks",
                "GG": "Good game",
                "WYD": "What are you doing",
                "HRU": "How are you",
                "GBU": "God bless you",
                "HBD": "Happy birthday",
                "GM": "Good morning",
                "GN": "Good night",
                "TTYL": "Talk to you later",
                "ILY": "I love you",
                "HI": "Hello",
                "OK": "Okay"
            }
            
            # Update UI elements
            self.bsl_model_path_var.set(self.bsl_config["model_path"])
            self.bsl_scaler_path_var.set(self.bsl_config["scaler_path"])
            self.bsl_feature_names_path_var.set(self.bsl_config["feature_names_path"])
            self.bsl_cooldown_var.set(self.bsl_config["cooldown"])
            self.bsl_confidence_threshold_var.set(self.bsl_config["confidence_threshold"])
            self.bsl_cooldown_label.config(text=f"{self.bsl_config['cooldown']:.1f} seconds")
            self.bsl_confidence_threshold_label.config(text=f"{self.bsl_config['confidence_threshold']:.1f}")
            
            self.eeg_model_path_var.set(self.eeg_config["model_path"])
            self.eeg_com_port_var.set(self.eeg_config["com_port"])
            self.eeg_baud_rate_var.set(self.eeg_config["baudrate"])
            
            self.tts_enabled_var.set(self.tts_config["enabled"])
            self.aws_region_var.set(self.tts_config["aws_region"])
            self.male_voice_var.set(self.tts_config["voice_male"])
            self.female_voice_var.set(self.tts_config["voice_female"])
            self.voice_gender_var.set(self.tts_config["gender"])
            
            self.theme_var.set("Default")
            
            self.auto_expand_var.set(self.auto_expand_enabled)
            self.update_abbreviation_display()
            
            self.update_tts_status()
            
            self.log("All settings reset to defaults")
    
    def toggle_bsl(self):
        """Toggle BSL sign language recognition on/off"""
        if self.bsl_running:
            self.stop_bsl()
        else:
            self.start_bsl()
    
    def toggle_eeg(self):
        """Toggle EEG emotion detection on/off"""
        if self.eeg_running:
            self.stop_eeg()
        else:
            self.start_eeg()
    
    def update_bsl_status(self):
        """Update the BSL status indicators"""
        if self.bsl_running:
            self.bsl_status_indicator.config(text="Active", foreground="green")
            self.bsl_toggle_button.config(text="Stop Sign Language")
        else:
            self.bsl_status_indicator.config(text="Inactive", foreground="red")
            self.bsl_toggle_button.config(text="Start Sign Language")
        
        # Update glove indicators
        self.left_glove_indicator.config(
            text=f"Left Glove: {'Connected' if self.bsl_is_connected[0] else 'Disconnected'}", 
            foreground="green" if self.bsl_is_connected[0] else "red"
        )
        self.right_glove_indicator.config(
            text=f"Right Glove: {'Connected' if self.bsl_is_connected[1] else 'Disconnected'}", 
            foreground="green" if self.bsl_is_connected[1] else "red"
        )
    
    def update_eeg_status(self):
        """Update the EEG status indicator"""
        if self.eeg_running:
            self.eeg_status_indicator.config(text="Active", foreground="green")
            self.eeg_toggle_button.config(text="Stop Emotion Detection")
        else:
            self.eeg_status_indicator.config(text="Inactive", foreground="red")
            self.eeg_toggle_button.config(text="Start Emotion Detection")
    
    def start_bsl(self):
        """Start BSL sign language recognition"""
        # Update button states
        self.bsl_start_button.config(state=tk.DISABLED)
        self.bsl_stop_button.config(state=tk.NORMAL)
        self.bsl_running = True
        
        # Update BSL hands based on checkboxes
        self.bsl_hands = [self.left_glove_var.get(), self.right_glove_var.get()]
        
        # Log
        self.bsl_log(f"Starting sign language recognition with Left={self.bsl_hands[0]}, Right={self.bsl_hands[1]}")
        
        # Create recognizer if not exists
        if self.bsl_recognizer is None:
            try:
                self.bsl_recognizer = BSLRecognizer(
                    model_path=self.bsl_config["model_path"],
                    scaler_path=self.bsl_config["scaler_path"],
                    feature_names_path=self.bsl_config["feature_names_path"]
                )
                # self.bsl_recognizer = BSLRecognizer()
                self.bsl_recognizer.set_cooldown(self.bsl_config["cooldown"])
                self.bsl_log("BSL Recognizer initialized successfully")
            except Exception as e:
                self.bsl_log(f"Error initializing BSL Recognizer: {e}")
                self.stop_bsl()
                return
        
        # Start the recognition thread
        self.bsl_recognition_thread = threading.Thread(target=self.run_bsl_recognition)
        self.bsl_recognition_thread.daemon = True
        self.bsl_recognition_thread.start()
        
        # Update status indicators
        self.update_bsl_status()
    
    def stop_bsl(self):
        """Stop BSL sign language recognition"""
        self.bsl_running = False
        self.bsl_log("Stopping sign language recognition")
        
        # Update button states
        self.bsl_start_button.config(state=tk.NORMAL)
        self.bsl_stop_button.config(state=tk.DISABLED)
        
        # Clean up
        if hasattr(self, 'bsl_task') and self.bsl_task is not None:
            self.bsl_task.cancel()
        
        # Update status indicators
        self.update_bsl_status()
    
    def run_bsl_recognition(self):
        """Run the BSL recognition process in a separate thread"""
        # Create new event loop for the thread
        asyncio.set_event_loop(asyncio.new_event_loop())
        loop = asyncio.get_event_loop()
        
        # Store the loop for later cleanup
        self.bsl_loop = loop
        
        try:
            # Run the main coroutine
            loop.run_until_complete(self.bsl_recognition_coroutine())
        except Exception as e:
            self.bsl_log(f"Error in recognition: {e}")
        finally:
            # Clean up the loop
            loop.close()
            self.bsl_log("BSL recognition thread stopped")
    
    async def bsl_recognition_coroutine(self):
        """Main coroutine for BSL recognition"""
        # Reset connection and data
        self.bsl_is_connected = [False, False]
        self.bsl_hand_data = []
        
        # Create Arduino interpreter
        arduino_interpreter = ArduinoInterpreter(self.bsl_is_connected, self.bsl_hand_data)
        
        # Start the Arduino connection task
        arduino_task = asyncio.create_task(arduino_interpreter.run(self.bsl_hands))
        self.bsl_task = arduino_task
        
        # Process data and make predictions
        try:
            while self.bsl_running:
                # Update connection status in UI
                self.root.after(0, self.update_bsl_status)
                
                # Only attempt prediction if we have data
                if self.bsl_hand_data:
                    # Get the latest data
                    latest_data = self.bsl_hand_data[-1]
                    
                    # Make prediction with confidence score
                    prediction, confidence = self.bsl_recognizer.predict(latest_data)
                    
                    # If we got a prediction with sufficient confidence
                    if prediction is not None and confidence is not None:
                        # Check if confidence meets threshold
                        threshold = self.bsl_config["confidence_threshold"]
                        
                        if confidence >= threshold:
                            # Update current sign and confidence
                            self.current_sign = prediction
                            self.sign_confidence = confidence
                            
                            # Update the UI
                            self.root.after(0, lambda: self.update_sign_display(prediction, confidence))
                            
                            # Log the prediction
                            self.bsl_log(f"Recognized: {prediction} (Confidence: {confidence:.2f})")
                
                # Brief pause to reduce CPU usage
                await asyncio.sleep(0.1)
                
        except Exception as e:
            self.bsl_log(f"Error: {e}")
        finally:
            # Cancel the Arduino task
            arduino_task.cancel()
            try:
                await arduino_task
            except asyncio.CancelledError:
                pass
    
    def update_sign_display(self, sign, confidence):
        """Update the sign display with the latest prediction and check for abbreviations"""
        # Update sign label
        self.sign_label.config(text=sign)
        
        # Update confidence
        percentage = int(confidence * 100)
        self.sign_confidence_label.config(text=f"{percentage}%")
        self.sign_confidence_bar['value'] = percentage
        
        # Update color based on confidence level
        if confidence >= 0.8:
            self.sign_confidence_label.config(foreground="green")
        elif confidence >= 0.6:
            self.sign_confidence_label.config(foreground="orange")
        else:
            self.sign_confidence_label.config(foreground="red")
        
        # Process the sign for abbreviation expansion
        self.process_sign_for_abbreviations(sign)
    
    
    def process_sign_for_abbreviations(self, sign):
        """Process a sign and check if it forms an abbreviation"""
        if not self.auto_expand_enabled:
            return
        
        # Add the sign to the recent signs list (convert to uppercase to match abbreviation keys)
        self.recent_signs.append(sign.upper())
        
        # Keep only the most recent signs
        if len(self.recent_signs) > self.max_recent_signs:
            self.recent_signs.pop(0)
        
        # Log current sequence for debugging
        self.log(f"Current sign sequence: {''.join(self.recent_signs[-5:])}")
        
        # Check for abbreviations in the recent signs
        for i in range(2, min(len(self.recent_signs) + 1, 6)):  # Start from 2 to avoid single letters
            # Get the last i signs and join them
            potential_abbrev = ''.join(self.recent_signs[-i:])
            
            # Log what we're checking
            self.log(f"Checking potential abbreviation: {potential_abbrev}")
            
            # Check if it matches any known abbreviation
            if potential_abbrev in self.abbreviations:
                expansion = self.abbreviations[potential_abbrev]
                self.log(f"Detected abbreviation: {potential_abbrev} → {expansion}")
                
                # If TTS is enabled, speak the expanded abbreviation
                if self.tts_enabled_var.get():
                    self.speak_text(expansion)
                
                # Display a notification about the abbreviation expansion
                self.show_abbreviation_notification(potential_abbrev, expansion)
                
                # Clear the recent signs to avoid double detection
                self.recent_signs = []
                break
    
    def show_abbreviation_notification(self, abbreviation, expansion):
        """Show a notification when an abbreviation is expanded"""
        # Create a toplevel window that automatically closes after a few seconds
        notification = tk.Toplevel(self.root)
        notification.title("Abbreviation Expanded")
        
        # Center the window
        window_width = 400
        window_height = 120
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x = int((screen_width - window_width) / 2)
        y = int((screen_height - window_height) / 2)
        notification.geometry(f"{window_width}x{window_height}+{x}+{y}")
        
        # Make the window stay on top
        notification.attributes('-topmost', True)
        
        # Add content
        frame = ttk.Frame(notification, padding="20")
        frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(frame, text=f"Expanded Abbreviation:", font=("Arial", 12, "bold")).pack(pady=(0, 5))
        ttk.Label(frame, text=f"{abbreviation} → {expansion}", font=("Arial", 14)).pack(pady=5)
        
        # Close button
        ttk.Button(frame, text="OK", command=notification.destroy).pack(pady=10)
        
        # Auto-close after 5 seconds
        notification.after(5000, notification.destroy)
    
    def toggle_auto_expand(self):
        """Toggle the auto-expand feature"""
        self.auto_expand_enabled = self.auto_expand_var.get()
        if self.auto_expand_enabled:
            self.log("Abbreviation auto-expansion enabled")
        else:
            self.log("Abbreviation auto-expansion disabled")
    
    def update_abbreviation_display(self):
        """Update the abbreviation text area with current abbreviations"""
        self.abbrev_text.config(state=tk.NORMAL)
        self.abbrev_text.delete(1.0, tk.END)
        
        # Sort abbreviations alphabetically
        sorted_abbrevs = sorted(self.abbreviations.items())
        
        for abbrev, expansion in sorted_abbrevs:
            self.abbrev_text.insert(tk.END, f"{abbrev}: {expansion}\n")
        
        self.abbrev_text.config(state=tk.DISABLED)
    
    def add_abbreviation(self):
        """Add a new abbreviation to the dictionary"""
        abbrev = self.new_abbrev_var.get().strip().upper()
        expansion = self.new_expansion_var.get().strip()
        
        if not abbrev or not expansion:
            messagebox.showerror("Error", "Both abbreviation and expansion must be provided")
            return
        
        # Add to the dictionary
        self.abbreviations[abbrev] = expansion
        
        # Update the display
        self.update_abbreviation_display()
        
        # Clear the entry fields
        self.new_abbrev_var.set("")
        self.new_expansion_var.set("")
        
        self.log(f"Added abbreviation: {abbrev} → {expansion}")
        
        # Save abbreviations to settings
        self.save_settings()
    
    def start_eeg(self):
        """Start EEG emotion detection"""
        # Update button states
        self.eeg_start_button.config(state=tk.DISABLED)
        self.eeg_stop_button.config(state=tk.NORMAL)
        self.calibrate_button.config(state=tk.DISABLED)
        self.eeg_running = True
        
        # Log
        self.eeg_log("Starting emotion detection")
        
        # Create detector if not exists
        if self.eeg_detector is None:
            try:
                self.eeg_detector = EEGEmotionDetector(
                    model_path=self.eeg_config["model_path"],
                    com_port=self.eeg_config["com_port"],
                    baudrate=self.eeg_config["baudrate"]
                )
                self.eeg_log("EEG Detector initialized successfully")
            except Exception as e:
                self.eeg_log(f"Error initializing EEG Detector: {e}")
                self.stop_eeg()
                return
        
        # Start emotion detection in a thread
        self.eeg_thread = threading.Thread(target=self.run_eeg_detection)
        self.eeg_thread.daemon = True
        self.eeg_thread.start()
        
        # Update status indicators
        self.update_eeg_status()
    
    def stop_eeg(self):
        """Stop EEG emotion detection"""
        self.eeg_running = False
        self.eeg_log("Stopping emotion detection")
        
        # Update button states
        self.eeg_start_button.config(state=tk.NORMAL)
        self.eeg_stop_button.config(state=tk.DISABLED)
        self.calibrate_button.config(state=tk.NORMAL)
        
        # Stop the detector
        if self.eeg_detector is not None:
            self.eeg_detector.stop()
        
        # Update status indicators
        self.update_eeg_status()
    
    def calibrate_eeg(self):
        """Run EEG calibration"""
        if self.eeg_running:
            messagebox.showwarning("Warning", "Please stop emotion detection before calibrating")
            return
        
        # Create detector if not exists
        if self.eeg_detector is None:
            try:
                self.eeg_detector = EEGEmotionDetector(
                    model_path=self.eeg_config["model_path"],
                    com_port=self.eeg_config["com_port"],
                    baudrate=self.eeg_config["baudrate"]
                )
                self.eeg_log("EEG Detector initialized successfully")
            except Exception as e:
                self.eeg_log(f"Error initializing EEG Detector: {e}")
                return
        
        # Disable the button during calibration
        self.calibrate_button.config(state=tk.DISABLED)
        
        # Start calibration in a thread
        threading.Thread(target=self.run_calibration).start()
    
    def run_calibration(self):
        """Run the calibration process in a thread"""
        try:
            # Connect to the device
            if self.eeg_detector.connect():
                self.eeg_log("Connected to EEG device for calibration")
                
                # Run calibration
                self.eeg_log("Starting calibration - please relax and remain still...")
                if self.eeg_detector.calibrate_neutral(duration=20):
                    self.eeg_log("Calibration completed successfully")
                else:
                    self.eeg_log("Calibration failed - please try again")
                
                # Close the connection
                self.eeg_detector.serial_conn.close()
            else:
                self.eeg_log(f"Failed to connect to EEG device on {self.eeg_config['com_port']}")
        except Exception as e:
            self.eeg_log(f"Error during calibration: {e}")
        finally:
            # Re-enable the button
            self.root.after(0, lambda: self.calibrate_button.config(state=tk.NORMAL))
    
    def run_eeg_detection(self):
        """Run the EEG detection process in a thread"""
        try:
            self.eeg_detector.start()
            
            last_print_time = time.time()
            
            # Create a buffer for stabilizing emotion detection
            emotion_buffer = []
            max_buffer_size = 5
            
            while self.eeg_running:
                # Get the latest data buffer from the detector
                if hasattr(self.eeg_detector, 'data_buffer') and len(self.eeg_detector.data_buffer) > 0:
                    # Process every second
                    current_time = time.time()
                    if current_time - last_print_time >= 1.0:
                        if len(self.eeg_detector.data_buffer) >= self.eeg_detector.buffer_size/2:
                            features = self.eeg_detector.extract_features(list(self.eeg_detector.data_buffer))
                            emotion, confidence = self.eeg_detector.predict_emotion(features)
                            
                            if emotion is not None and confidence is not None:
                                # Add to buffer for stabilization
                                emotion_buffer.append((emotion, confidence))
                                if len(emotion_buffer) > max_buffer_size:
                                    emotion_buffer.pop(0)
                                
                                # Get most common emotion
                                if len(emotion_buffer) >= 3:
                                    emotion_counts = {}
                                    for e, _ in emotion_buffer:
                                        emotion_counts[e] = emotion_counts.get(e, 0) + 1
                                    
                                    stable_emotion = max(emotion_counts, key=emotion_counts.get)
                                    
                                    # Get average confidence for this emotion
                                    confidences = [c for e, c in emotion_buffer if e == stable_emotion]
                                    avg_confidence = sum(confidences) / len(confidences) if confidences else 0
                                    
                                    # Update if emotion has changed
                                    if stable_emotion != self.current_emotion or abs(avg_confidence - self.emotion_confidence) > 0.1:
                                        self.current_emotion = stable_emotion
                                        self.emotion_confidence = avg_confidence
                                        
                                        # Update the UI
                                        self.root.after(0, lambda e=stable_emotion, c=avg_confidence: 
                                                        self.update_emotion_display(e, c))
                                        
                                        # Log the prediction
                                        self.eeg_log(f"Detected emotion: {stable_emotion} (Confidence: {avg_confidence:.2f})")
                            
                        last_print_time = current_time
                
                time.sleep(0.1)
                
        except Exception as e:
            self.eeg_log(f"Error in EEG detection: {e}")
        finally:
            if self.eeg_detector is not None:
                self.eeg_detector.stop()
    
    def update_emotion_display(self, emotion, confidence):
        """Update the emotion display with the latest prediction"""
        # Update emotion label
        self.emotion_label.config(text=emotion)
        
        # Update confidence
        percentage = int(confidence * 100)
        self.emotion_confidence_label.config(text=f"{percentage}%")
        self.emotion_confidence_bar['value'] = percentage
        
        # Update color based on confidence level
        if confidence >= 0.8:
            self.emotion_confidence_label.config(foreground="green")
        elif confidence >= 0.6:
            self.emotion_confidence_label.config(foreground="orange")
        else:
            self.emotion_confidence_label.config(foreground="red")
    
    def speak_current_sign(self):
        """Speak the current sign with the detected emotion"""
        if not self.tts_enabled_var.get():
            messagebox.showinfo("Text-to-Speech Disabled", "Please enable Text-to-Speech in the settings tab.")
            return
        
        if not self.current_sign:
            messagebox.showinfo("No Sign Detected", "Please sign something first.")
            return
        
        # Use selected emotion if the option is enabled
        if self.use_selected_emotion_var.get() and self.selected_emotion_var.get():
            emotion_to_use = self.selected_emotion_var.get()
            self.log(f"Using selected emotion: {emotion_to_use}")
        else:
            emotion_to_use = self.current_emotion
            self.log(f"Using detected emotion: {emotion_to_use}")
        
        self.speak_text(self.current_sign, emotion_to_use)
    
    def speak_custom_text(self):
        """Speak the custom text with selected or detected emotion"""
        if not self.tts_enabled_var.get():
            messagebox.showinfo("Text-to-Speech Disabled", "Please enable Text-to-Speech in the settings tab.")
            return
        
        custom_text = self.custom_text.get().strip()
        if not custom_text:
            messagebox.showinfo("Empty Text", "Please enter some text to speak.")
            return
        
        # Get the selected emotion (if any) or use the detected emotion
        if self.use_selected_emotion_var.get() and self.selected_emotion_var.get():
            emotion_to_use = self.selected_emotion_var.get()
            self.log(f"Using selected emotion: {emotion_to_use}")
        else:
            emotion_to_use = self.current_emotion
            self.log(f"Using detected emotion: {emotion_to_use}")
        
        # Check if the custom text is an abbreviation
        if self.auto_expand_enabled and custom_text.upper() in self.abbreviations:
            expanded_text = self.abbreviations[custom_text.upper()]
            self.log(f"Expanded abbreviation: {custom_text} → {expanded_text}")
            self.show_abbreviation_notification(custom_text.upper(), expanded_text)
            self.speak_text(expanded_text, emotion_to_use)
        else:
            # Process the text for any included abbreviations
            if self.auto_expand_enabled:
                # Split text into words and check each word for abbreviations
                words = custom_text.split()
                expanded_words = []
                expansion_occurred = False
                
                for word in words:
                    if word.upper() in self.abbreviations:
                        expanded_words.append(self.abbreviations[word.upper()])
                        self.log(f"Expanded abbreviation in text: {word} → {self.abbreviations[word.upper()]}")
                        expansion_occurred = True
                    else:
                        expanded_words.append(word)
                
                # If any expansions occurred, show notification and use expanded text
                if expansion_occurred:
                    expanded_text = ' '.join(expanded_words)
                    self.show_abbreviation_notification(custom_text, expanded_text)
                    self.speak_text(expanded_text, emotion_to_use)
                else:
                    self.speak_text(custom_text, emotion_to_use)
            else:
                self.speak_text(custom_text, emotion_to_use)

    def speak_text(self, text, emotion=None):
        """Use Amazon Polly to speak the text with emotion"""
        if self.polly is None:
            messagebox.showerror("Error", "Amazon Polly client not initialized. Please check AWS credentials.")
            return
        
        try:
            # Determine which voice to use based on gender setting
            if self.voice_gender_var.get() == "female":
                voice_id = self.female_voice_var.get()
            else:
                voice_id = self.male_voice_var.get()
            
            # Use specified emotion or fall back to detected emotion
            emotion_to_use = emotion if emotion else self.current_emotion
            
            # Create SSML for the text with emotion
            ssml_text = self.create_emotional_ssml(text, emotion_to_use)
            
            self.log(f"Speaking: '{text}' with {emotion_to_use} emotion using {voice_id} voice")
            
            # Try to load credentials from file again to ensure we're using the latest
            try:
                if os.path.exists('aws_credentials.json'):
                    with open('aws_credentials.json', 'r') as file:
                        creds = json.load(file)
                    
                    if 'access_key' in creds and 'secret_key' in creds and creds['access_key'] and creds['secret_key']:
                        # Create a fresh client with the loaded credentials
                        polly_client = boto3.client(
                            'polly', 
                            region_name=self.aws_region_var.get(),
                            aws_access_key_id=creds['access_key'],
                            aws_secret_access_key=creds['secret_key']
                        )
                        
                        # Use this client for this operation
                        response = polly_client.synthesize_speech(
                            Text=ssml_text,
                            VoiceId=voice_id,
                            OutputFormat="mp3",
                            Engine="standard",
                            LanguageCode="en-GB",
                            TextType="ssml"
                        )
                    else:
                        # Fall back to default client
                        response = self.polly.synthesize_speech(
                            Text=ssml_text,
                            VoiceId=voice_id,
                            OutputFormat="mp3",
                            Engine="standard",
                            LanguageCode="en-GB",
                            TextType="ssml"
                        )
                else:
                    # No credential file, use default client
                    response = self.polly.synthesize_speech(
                        Text=ssml_text,
                        VoiceId=voice_id,
                        OutputFormat="mp3",
                        Engine="standard",
                        LanguageCode="en-GB",
                        TextType="ssml"
                    )
            except Exception as file_error:
                self.log(f"Error loading credentials file, using default client: {file_error}")
                # Fall back to default client
                response = self.polly.synthesize_speech(
                    Text=ssml_text,
                    VoiceId=voice_id,
                    OutputFormat="mp3",
                    Engine="standard",
                    LanguageCode="en-GB",
                    TextType="ssml"
                )
            
            # Get the audio stream and play it using pygame
            audio_stream = response['AudioStream'].read()
            audio_file = BytesIO(audio_stream)
            pygame.mixer.music.load(audio_file)
            pygame.mixer.music.play()
            
        except Exception as e:
            if "Unable to locate credentials" in str(e):
                messagebox.showerror("AWS Credentials Error", 
                                "AWS SDK cannot find valid credentials. Please enter your AWS credentials in the Settings tab and click 'Save Credentials'.")
            else:
                messagebox.showerror("Speech Error", f"Error generating speech: {e}")
            self.log(f"Error generating speech: {e}")
    
    def create_emotional_ssml(self, text, emotion):
        """Create SSML with emotional parameters based on the detected emotion"""
        # Define emotional speech parameters
        emotion_params = {
            "Happy": {
                "rate": "fast",
                "pitch": "high",
                "volume": "loud"
            },
            "Sad": {
                "rate": "slow",
                "pitch": "low",
                "volume": "soft"
            },
            "Angry": {
                "rate": "fast",
                "pitch": "low",
                "volume": "loud"
            },
            "Neutral": {
                "rate": "medium",
                "pitch": "medium",
                "volume": "medium"
            }
        }
        
        # Get parameters for the emotion (default to Neutral if not found)
        params = emotion_params.get(emotion, emotion_params["Neutral"])
        
        # Create SSML
        ssml = f'''<speak>
                    <prosody rate="{params['rate']}" pitch="{params['pitch']}" volume="{params['volume']}">
                        {text}
                    </prosody>
                 </speak>'''
        
        return ssml
    
    def test_tts(self):
        """Test the text-to-speech with current settings"""
        if not self.tts_enabled_var.get():
            messagebox.showinfo("Text-to-Speech Disabled", "Please enable Text-to-Speech in the settings tab.")
            return
        
        # Check if we have valid AWS credentials
        try:
            # Make sure we're using the current region setting
            region = self.aws_region_var.get()
            
            # First try to load credentials from aws_credentials.json
            if os.path.exists('aws_credentials.json'):
                try:
                    with open('aws_credentials.json', 'r') as file:
                        creds = json.load(file)
                    
                    if 'access_key' in creds and 'secret_key' in creds and creds['access_key'] and creds['secret_key']:
                        self.log("Using explicit credentials from aws_credentials.json")
                        # Create a new client with explicit credentials
                        self.polly = boto3.client(
                            'polly', 
                            region_name=region,
                            aws_access_key_id=creds['access_key'],
                            aws_secret_access_key=creds['secret_key']
                        )
                except Exception as e:
                    self.log(f"Error loading credentials from file: {e}")
            
            # Verify we can use Polly
            self.polly.describe_voices(LanguageCode="en-GB", Engine="standard")
        except Exception as e:
            error_msg = str(e)
            if "Unable to locate credentials" in error_msg:
                messagebox.showerror("AWS Credentials Error", 
                                   "AWS SDK cannot find valid credentials. Please enter your AWS credentials in the Settings tab and click 'Save Credentials'.")
            else:
                messagebox.showerror("AWS Error", f"Error connecting to AWS Polly: {error_msg}")
            self.log(f"AWS Polly error: {e}")
            return
        
        # Ask which emotion to test
        emotions = ["Happy", "Sad", "Angry", "Neutral"]
        test_dialog = TestVoiceDialog(self.root, emotions)
        if test_dialog.result:
            emotion, gender = test_dialog.result
            
            # Set the voice gender for the test
            old_gender = self.voice_gender_var.get()
            self.voice_gender_var.set(gender)
            
            # Test text for each emotion
            test_texts = {
                "Happy": "This is how I sound when I'm happy!",
                "Sad": "This is how I sound when I'm sad.",
                "Angry": "This is how I sound when I'm angry!",
                "Neutral": "This is how I sound in a neutral state."
            }
            
            # Speak the test text with the selected emotion
            self.current_emotion = emotion
            self.speak_text(test_texts[emotion])
            
            # Restore the original gender setting
            self.voice_gender_var.set(old_gender)
    
    def on_close(self):
        """Handle window closing"""
        if messagebox.askyesno("Exit", "Are you sure you want to exit?"):
            self.shutdown_flag = True
            
            # Stop BSL if running
            if self.bsl_running:
                self.stop_bsl()
            
            # Stop EEG if running
            if self.eeg_running:
                self.stop_eeg()
            
            # Save settings before exit
            try:
                self.save_settings()
            except:
                pass
            
            self.root.destroy()


class TestVoiceDialog:
    """Dialog for testing voice with different emotions and genders"""
    def __init__(self, parent, emotions):
        self.result = None
        
        # Create the dialog window
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Test Voice")
        self.dialog.geometry("300x200")
        self.dialog.resizable(False, False)
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        # Center the dialog
        window_width = 300
        window_height = 200
        screen_width = parent.winfo_screenwidth()
        screen_height = parent.winfo_screenheight()
        x = int((screen_width - window_width) / 2)
        y = int((screen_height - window_height) / 2)
        self.dialog.geometry(f"{window_width}x{window_height}+{x}+{y}")
        
        # Create the dialog content
        frame = ttk.Frame(self.dialog, padding="10")
        frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(frame, text="Select Emotion:").pack(pady=5)
        
        # Emotion selection
        self.emotion_var = tk.StringVar(value=emotions[0])
        emotion_combo = ttk.Combobox(frame, textvariable=self.emotion_var, values=emotions, state="readonly")
        emotion_combo.pack(pady=5)
        
        ttk.Label(frame, text="Select Voice Gender:").pack(pady=5)
        
        # Gender selection
        self.gender_var = tk.StringVar(value="female")
        gender_frame = ttk.Frame(frame)
        gender_frame.pack(pady=5)
        
        ttk.Radiobutton(gender_frame, text="Female", variable=self.gender_var, value="female").pack(side=tk.LEFT, padx=10)
        ttk.Radiobutton(gender_frame, text="Male", variable=self.gender_var, value="male").pack(side=tk.LEFT, padx=10)
        
        # Buttons
        button_frame = ttk.Frame(frame)
        button_frame.pack(pady=15)
        
        ttk.Button(button_frame, text="Test", command=self.on_ok).pack(side=tk.LEFT, padx=10)
        ttk.Button(button_frame, text="Cancel", command=self.on_cancel).pack(side=tk.LEFT, padx=10)
        
        # Wait for the dialog to close
        self.dialog.wait_window()
    
    def on_ok(self):
        """Handle OK button click"""
        self.result = (self.emotion_var.get(), self.gender_var.get())
        self.dialog.destroy()
    
    def on_cancel(self):
        """Handle Cancel button click"""
        self.dialog.destroy()


def main():
    """Main function to start the application"""
    root = tk.Tk()
    app = IntegratedCommunicationApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
