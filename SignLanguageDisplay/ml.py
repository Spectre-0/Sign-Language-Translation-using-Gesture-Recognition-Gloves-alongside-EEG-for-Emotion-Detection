import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import pickle
import os
import joblib
import time
import xgboost as xgb

def load_and_prepare_data(file_path='combined_data.csv'):
    """Load and prepare the BSL sign language data"""
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    print(f"Data loaded successfully with shape: {df.shape}")
    
    return preprocess_data(df)

def preprocess_data(df):
    """Preprocess the BSL sign language data"""
    print("Preprocessing data...")
    
    # 1. Handle outliers
    for col in df.select_dtypes(include=[np.number]).columns:
        if col not in ['run_no', 'hand', 'timestamp']:
            mean, std = df[col].mean(), df[col].std()
            df[col] = df[col].clip(mean - 3*std, mean + 3*std)
    
    # 2. Create derived features
    df['flex_ratio_1_2'] = df['flex1'] / (df['flex2'] + 1e-6)
    df['flex_ratio_2_3'] = df['flex2'] / (df['flex3'] + 1e-6)
    df['flex_ratio_3_4'] = df['flex3'] / (df['flex4'] + 1e-6)
    df['flex_ratio_4_5'] = df['flex4'] / (df['flex5'] + 1e-6)
    
    # Calculate finger curvature (approximation)
    df['index_curve'] = df['flex1'] / 1000.0
    df['middle_curve'] = df['flex2'] / 1000.0
    df['ring_curve'] = df['flex3'] / 1000.0
    df['pinky_curve'] = df['flex4'] / 1000.0
    df['thumb_curve'] = df['flex5'] / 1000.0
    
    # 3. Handle zeros in flex sensors (potential sensor errors)
    for col in ['flex1', 'flex2', 'flex3', 'flex4', 'flex5']:
        zero_mask = df[col] == 0
        if zero_mask.sum() > 0:
            letter_medians = df.groupby('word')[col].median()
            for letter in df['word'].unique():
                letter_mask = (df['word'] == letter) & zero_mask
                df.loc[letter_mask, col] = letter_medians[letter]
    
    # 4. Add orientation-based features
    df['hand_orientation'] = np.sqrt(df['roll']**2 + df['pitch']**2)
    df['accel_magnitude'] = np.sqrt(df['accelX']**2 + df['accelY']**2 + df['accelZ']**2)
    df['gyro_magnitude'] = np.sqrt(df['gyroX']**2 + df['gyroY']**2 + df['gyroZ']**2)
    
    # 5. Create hand position features based on both hand and sensor values
    # Assuming 0 = left hand, 1 = right hand
    df['is_right_hand'] = df['hand']
    
    print("Preprocessing complete.")
    return df

def train_and_save_model(df, model_file='bsl_model.joblib'):
    """Train machine learning models and save the best one"""
    # Split features and target
    X = df.drop(['word', 'timestamp', 'run_no'], axis=1)
    y = df['word']
    
    # Store feature names for later use
    feature_names = X.columns.tolist()
    joblib.dump(feature_names, 'feature_names.joblib')
    
    # Save the scaler separately for real-time use
    scaler = StandardScaler()
    scaler.fit(X)
    joblib.dump(scaler, 'bsl_scaler.joblib')
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training data shape: {X_train.shape}, Testing data shape: {X_test.shape}")
    
    # First, try a simple Random Forest to get a baseline
    print("\nTraining baseline Random Forest model...")
    rf_baseline = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_baseline.fit(scaler.transform(X_train), y_train)
    
    # Evaluate baseline model
    y_pred_baseline = rf_baseline.predict(scaler.transform(X_test))
    print("\nBaseline Random Forest results:")
    print(classification_report(y_test, y_pred_baseline))
    
    # Save baseline model
    joblib.dump(rf_baseline, 'bsl_model_baseline.joblib')
    
    # Get the baseline F1 score
    baseline_f1 = classification_report(y_test, y_pred_baseline, output_dict=True)['macro avg']['f1-score']
    
    # Keep track of the best model
    best_model = rf_baseline
    best_f1 = baseline_f1
    best_model_name = "Baseline Random Forest"
    
    # Try XGBoost with GPU acceleration for faster training
    print("\nTraining XGBoost model with GPU acceleration...")
    try:
        # Check if GPU is available
        try:
            gpu_available = len(xgb.get_gpu_info()) > 0
            tree_method = 'gpu_hist' if gpu_available else 'hist'
        except:
            gpu_available = False
            tree_method = 'hist'
            
        print(f"GPU acceleration {'available' if gpu_available else 'not available'}, using {tree_method}")
        
        # Create DMatrix for faster processing
        dtrain = xgb.DMatrix(scaler.transform(X_train), label=pd.factorize(y_train)[0])
        dtest = xgb.DMatrix(scaler.transform(X_test), label=pd.factorize(y_test)[0])
        
        # Get the label mapping
        labels, _ = pd.factorize(y_train)
        unique_labels = np.unique(labels)
        label_to_word = {i: word for i, word in zip(unique_labels, y_train.iloc[np.where(labels == unique_labels)[0]].values)}
        
        # Set XGBoost parameters
        xgb_params = {
            'objective': 'multi:softprob',
            'num_class': len(y.unique()),
            'max_depth': 5,
            'eta': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'tree_method': tree_method,
            'eval_metric': 'mlogloss'
        }
        
        # Train XGBoost model
        num_rounds = 100
        print(f"Training XGBoost for {num_rounds} rounds...")
        xgb_model = xgb.train(xgb_params, dtrain, num_rounds)
        
        # Evaluate XGBoost model
        y_pred_proba_xgb = xgb_model.predict(dtest)
        y_pred_xgb = np.argmax(y_pred_proba_xgb, axis=1)
        
        # Convert numeric predictions back to original letter labels
        y_pred_xgb_labels = [label_to_word.get(pred, "unknown") for pred in y_pred_xgb]
        
        print("\nXGBoost Classifier results:")
        print(classification_report(y_test, y_pred_xgb_labels))
        
        # Calculate F1 score
        xgb_f1 = classification_report(y_test, y_pred_xgb_labels, output_dict=True)['macro avg']['f1-score']
        
        # Include XGBoost in model comparison
        print(f"XGBoost F1 Score: {xgb_f1:.4f}")
        
        # Check if XGBoost is better
        if xgb_f1 > best_f1:
            best_f1 = xgb_f1
            best_model_name = "XGBoost"
            
            # Save XGBoost model specifically
            xgb_model.save_model('xgb_model.json')
            print("\nSaving XGBoost model as it performed best.")
            
            # Create XGBoost wrapper for compatibility
            class XGBoostWrapper:
                def __init__(self, model_path='xgb_model.json'):
                    self.model = None
                    self.model_path = model_path
                    self.label_map = label_to_word
                
                def predict(self, X):
                    if self.model is None:
                        self.model = xgb.Booster()
                        self.model.load_model(self.model_path)
                    dtest = xgb.DMatrix(X)
                    y_pred_proba = self.model.predict(dtest)
                    y_pred = np.argmax(y_pred_proba, axis=1)
                    return np.array([self.label_map.get(pred, "unknown") for pred in y_pred])
                
                def predict_proba(self, X):
                    if self.model is None:
                        self.model = xgb.Booster()
                        self.model.load_model(self.model_path)
                    dtest = xgb.DMatrix(X)
                    return self.model.predict(dtest)
            
            # Create and save wrapper
            best_model = XGBoostWrapper()
            
    except Exception as e:
        print(f"Error training XGBoost model: {e}")
        print("XGBoost training failed, continuing with Random Forest.")
    
    # Save the best model
    joblib.dump(best_model, model_file)
    print(f"Best model ({best_model_name}) saved to {model_file}")
    
    # Create a metadata file with information about the model
    metadata = {
        'feature_names': feature_names,
        'model_type': best_model_name,
        'training_date': time.strftime("%Y-%m-%d %H:%M:%S"),
        'accuracy': float(best_f1),
        'num_features': len(feature_names),
        'num_classes': len(y.unique()),
        'classes': sorted(y.unique().tolist())
    }
    
    # Save metadata as a simple JSON file
    import json
    with open('model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=4)
    
    print("Model training complete!")
    
    return best_model

if __name__ == "__main__":
    print("BSL Sign Language Recognition - Model Training")
    print("=============================================")
    
    # Check if data file exists
    default_file = 'combined_data.csv'
    data_file = input(f"Enter path to data file (press Enter for default '{default_file}'): ").strip()
    
    if not data_file:
        data_file = default_file
    
    if not os.path.exists(data_file):
        print(f"Error: File '{data_file}' not found.")
        exit(1)
    
    # Load and preprocess the data
    processed_data = load_and_prepare_data(data_file)
    
    # Ask for model save file
    model_file = input("Enter filename to save model (press Enter for default 'bsl_model.joblib'): ").strip()
    if not model_file:
        model_file = 'bsl_model.joblib'
    
    # Train and save the model
    model = train_and_save_model(processed_data, model_file)
    
    print("\nDone! The model is ready for real-time sign language recognition.")