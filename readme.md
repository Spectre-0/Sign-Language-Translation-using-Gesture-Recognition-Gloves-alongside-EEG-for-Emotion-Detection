# BSL Sign Language Translation System with Emotion Detection

## Table of Contents
- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Installation](#installation)
- [Usage](#usage)
- [System Architecture](#system-architecture)
  - [Gesture Recognition System](#gesture-recognition-system)
  - [EEG Emotion Detection System](#eeg-emotion-detection-system)
  - [Voice Output System](#voice-output-system)
  - [Integration System](#integration-system)
- [Technical Implementation Details](#technical-implementation-details)
  - [Gesture Recognition](#gesture-recognition)
  - [EEG Emotion Detection](#eeg-emotion-detection)
  - [Voice Synthesis](#voice-synthesis)
- [Performance Metrics](#performance-metrics)
- [Team Contributions](#team-contributions)
- [Challenges and Solutions](#challenges-and-solutions)
- [Future Work](#future-work)

## Project Overview

This system creates a bridge between British Sign Language (BSL) users and spoken language by translating hand gestures into speech while incorporating the user's emotional state for more expressive communication. The project combines custom-designed sensor gloves for gesture recognition with an EEG-based emotion detection system that modulates speech output according to the user's emotional state, creating a more natural and expressive communication experience.

## Key Features

- **Real-time BSL Translation**: Recognizes and translates British Sign Language alphabet (A-Z) and common phrases
- **Emotion-Aware Speech**: Modifies speech output tone based on detected emotional state
- **Wireless Operation**: Bluetooth-connected gloves for natural movement without restrictions
- **Abbreviation Expansion**: Automatically expands common abbreviations (e.g., "BRB" → "Be right back")
- **Personalized Calibration**: Neural calibration system adapts to individual EEG patterns
- **Multi-Modal Integration**: Seamlessly combines gesture recognition and emotion detection

## Installation

### Prerequisites
- Python 3.8 or higher
- Arduino IDE (for glove firmware)
- AWS account (for Amazon Polly TTS)

### Glove Hardware Requirements
- Seeed XIAO nRF52840 microcontroller
- 5× Adafruit flex sensors
- 3.7V LiPo battery
- Stripboard for circuit
- Ansell HyFlex 11-840 gloves or equivalent

### EEG Hardware Requirements
- Olimex EEG-SMT board
- 4× active electrodes
- 1× passive reference electrode
- Conductive gel
- Headband for mounting

### Software Installation

1. Clone the repository:
```bash
git clone https://github.com/Spectre-0/Sign-Language-Translation-using-Gesture-Recognition-Gloves-alongside-EEG-for-Emotion-Detection.git
cd bsl-emotion-translation
```

2. Install required Python packages:
```bash
pip install -r requirements.txt
```

3. Configure AWS credentials for Amazon Polly:
```bash
# Create aws_credentials.json in the project directory with your credentials:
{
  "access_key": "YOUR_ACCESS_KEY",
  "secret_key": "YOUR_SECRET_KEY"
}
```

4. Upload the glove firmware:
   - Open `glove_firmware/glove_firmware.ino` in Arduino IDE
   - Select the Seeed XIAO nRF52840 board
   - Upload the firmware to both gloves

## Usage

### Setup
1. Power on the gloves and ensure they are charged
2. Prepare the EEG headset:
   - Apply conductive gel to electrodes
   - Position electrodes at Fp1, Fp2, F7, F8 locations
   - Place reference electrode on earlobe
   - Secure with headband

### Running the System
1. Start the main application:
```bash
python main.py
```

2. Connect to gloves:
   - The system will automatically scan for and connect to nearby gloves
   - Verify connection status in the interface

3. Calibrate the EEG:
   - Click "Calibrate Neutral State" button
   - Maintain a neutral expression for 20 seconds during calibration

4. Begin signing:
   - Start signing letters or words with the gloves
   - The system will recognize signs and generate speech output
   - Speech will be modulated based on detected emotions

### Interface Controls
- **Toggle Voice Gender**: Switch between male/female voice
- **Sensitivity Slider**: Adjust gesture recognition sensitivity
- **Confidence Threshold**: Set minimum confidence for recognition
- **Add Abbreviation**: Create custom abbreviation expansions

## System Architecture

### Gesture Recognition System

#### Hardware Components
- **Microcontroller**: Seeed XIAO nRF52840 with built-in Bluetooth, accelerometer, and gyroscope
- **Sensors**: 5 Adafruit flex sensors (one per finger) to capture finger bending
- **Power**: 3.7V 1800mAh LiPo battery for extended wireless operation
- **Circuit Design**: Custom stripboard circuit with voltage dividers for flex sensors
- **Gloves**: Ansell HyFlex 11-840 professional work gloves with sewn sensor pockets

#### Software Components
- **Arduino Firmware**: C++ code for sensor data collection and filtering
- **Bluetooth Communication**: Bluefruit library for wireless data transmission
- **Signal Processing**: Kalman filtering to reduce noise in sensor readings
- **Data Preprocessing**: Feature engineering including flex ratios and orientation features
- **Machine Learning**: Optimized Random Forest classifier with 85% accuracy across all letters

### EEG Emotion Detection System

#### Hardware Components
- **EEG Device**: Olimex EEG-SMT open-source EEG board
- **Electrodes**: 4 active electrodes + 1 passive reference electrode
- **Electrode Placement**: Fp1, Fp2, F7, F8 (frontal lobe) with earlobe reference
- **Mounting**: Custom headband system for stable electrode positioning

#### Software Components
- **Signal Acquisition**: Serial communication with the EEG-SMT device (57600 baud)
- **Feature Extraction**: Time-domain, frequency-domain, and non-linear features
- **Machine Learning**: Custom neural network with residual connections
- **Neural Calibration**: Personalized baseline adjustment system
- **Real-Time Processing**: Continuous emotion classification with temporal smoothing

### Voice Output System

#### Components
- **TTS Engine**: Amazon Polly cloud-based service
- **Voice Options**: British male (Brian) and female (Amy) voices
- **Emotion Modulation**: SSML prosody tags for emotion-specific speech patterns
- **Audio Playback**: PyGame for smooth audio rendering
- **AWS Integration**: Boto3 for Amazon Web Services connectivity

### Integration System

#### Components
- **GUI Application**: Intuitive interface for system control and monitoring
- **Multi-Threading**: Parallel processing of gesture and emotion data
- **Data Fusion**: Integration of gesture recognition and emotion detection results
- **User Profiles**: Calibration data storage for different users
- **Configuration Options**: Adjustable parameters for performance optimization

## Technical Implementation Details

### Gesture Recognition

The gloves use a combination of flex sensors and an IMU to capture hand movements and gestures. Data is sent via Bluetooth to a computer for processing.

#### Data Flow:
1. Flex sensors measure finger bending (0-90 degrees)
2. IMU captures hand orientation (accelerometer + gyroscope)
3. Arduino code samples sensors at 100ms intervals
4. Kalman filtering removes noise from readings
5. Bluetooth transmits data packets to computer
6. Feature extraction creates derived metrics (flex ratios, orientation features)
7. Random Forest model classifies gestures with 85% accuracy
8. Temporal smoothing prevents output jitter

#### Machine Learning:
- **Model**: Random Forest with 100 decision trees
- **Features**: 21 features per channel (84 total)
- **Training Data**: 2,600 samples (100 per letter)
- **Performance**: 91% accuracy for best letters (W, C, S), 78% for challenging letters (N, M)

### EEG Emotion Detection

The EEG system uses brain wave patterns to classify four emotional states: Neutral, Happy, Sad, and Angry.

#### Data Flow:
1. EEG-SMT captures brain activity at 59Hz sampling rate
2. Frontal lobe electrodes (Fp1, Fp2, F7, F8) focus on emotion centers
3. Signal preprocessing removes artifacts and filters noise
4. Feature extraction creates 96-dimensional feature vector
5. Neural network classifies emotional state
6. Neural calibration adjusts for individual differences
7. Temporal integration smooths classifications

#### Machine Learning:
- **Model**: Custom neural network with residual connections
- **Training Data**: DEAP dataset + custom recordings
- **Performance**: 75.23% overall validation accuracy
- **Emotion-Specific Accuracy**: Neutral (58.7%), Happy (83.0%), Sad (87.0%), Angry (90.6%)

### Voice Synthesis

The speech output system uses Amazon Polly to generate natural-sounding speech with emotional modulation.

#### Features:
- **Emotion Mapping**: Speech parameters adjusted based on detected emotion
- **Happy**: Faster rate, higher pitch, louder volume
- **Sad**: Slower rate, lower pitch, softer volume
- **Angry**: Faster rate, lower pitch, louder volume
- **Neutral**: Medium values for all parameters

#### Implementation:
- **SSML Tags**: Speech Synthesis Markup Language for prosody control
- **Voice Selection**: Gender choice between male and female voices
- **Abbreviation Expansion**: Custom dictionary for common phrases

## Performance Metrics

### Sign Recognition Accuracy
- **Overall Accuracy**: 83.7% across all participants and letters
- **Best Performing Letters**: S (92.0%), W (91.4%), C (91.8%)
- **Challenging Letters**: N (77.8%), M (79.2%), J (80.6%)

### Emotion Detection Accuracy
- **Overall Accuracy**: 72.5% across all emotions
- **Best Performing Emotions**: Sad (86.0%), Angry (86.0%)
- **Challenging Emotions**: Neutral (56.0%)

### System Latency
- **Gesture Recognition**: ~150-200ms
- **Emotion Detection**: ~500ms
- **Speech Synthesis**: 200-500ms
- **Total End-to-End**: 1-2 seconds

## Team Contributions

### Adnan Uddin
- **EEG System**: Developed complete EEG emotion detection pipeline
  - Designed and implemented the neural network architecture
  - Created the feature extraction system for EEG signals
  - Implemented neural calibration methodology
  - Developed real-time processing pipeline

- **Machine Learning**:
  - Contributed to the Random Forest model for glove gesture recognition
  - Implemented data augmentation techniques for emotion detection
  - Optimized model performance through hyperparameter tuning

- **Integration**:
  - Assisted with component integration between systems
  - Helped with EEG-TTS integration for emotion-modulated speech

### Wiktor Szczepan
- **Gesture Recognition Model**:
  - Led development of the Random Forest gesture recognition model
  - Designed feature engineering pipeline for gesture data
  - Implemented cross-validation methodology
  - Created confusion matrix analysis tools
  - Developed model performance visualization

- **Data Processing**:
  - Created preprocessing pipelines for gesture data
  - Implemented data cleaning and normalization
  - Developed feature importance analysis
  - Optimized classification algorithms

- **Hardware Integration**:
  - Contributed to glove component selection and testing
  - Assisted with sensor positioning and circuit design

### Muhammad Syam Atif
- **Hardware Development**:
  - Led the hardware design and implementation for gesture recognition gloves
  - Designed and constructed the custom circuit on stripboard
  - Implemented sensor integration and wiring
  - Created protection mechanisms for components

- **Firmware Development**:
  - Developed the Arduino C/C++ code for sensor data collection
  - Implemented Kalman filtering for signal processing
  - Created the Bluetooth communication protocol
  - Optimized sampling rate and power consumption

- **Physical Construction**:
  - Designed sensor mounting system for flexibility and durability
  - Created protective pouches for electronic components
  - Implemented battery housing and management

### Bazil Saleem
- **Speech System**:
  - Developed the complete voice output system using Amazon Polly
  - Implemented emotion-modulated speech synthesis
  - Created the amazon.py module for AWS connectivity
  - Developed audio playback system using PyGame

- **TTS Integration**:
  - Implemented SSML prosody modifications for emotional expression
  - Created voice selection interface
  - Developed audio stream handling and playback
  - Optimized latency in speech generation

- **Abbreviation System**:
  - Designed and implemented the abbreviation expansion feature
  - Created the dictionary management system
  - Implemented real-time abbreviation detection and expansion

### Amanda Sophie Betton
- **Project Management**:
  - Led overall project coordination and planning
  - Managed timeline and milestone tracking
  - Facilitated team communication and task allocation
  - Ensured project scope alignment with requirements

- **Hardware Construction**:
  - Contributed significantly to glove assembly and construction
  - Participated in hardware testing and validation

- **Integration Testing**:
  - Led system integration and testing
  - Developed test protocols for component validation
  - Coordinated cross-component functionality
  - Managed project documentation and reporting

---

This project was developed as a final year group project in Computer Science, demonstrating the successful integration of multiple technologies to create a comprehensive communication tool for the hearing impaired, combining hardware engineering, signal processing, machine learning, and user interface design into a cohesive system.
