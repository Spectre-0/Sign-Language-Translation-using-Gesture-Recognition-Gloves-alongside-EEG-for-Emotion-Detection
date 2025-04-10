import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import os

def load_data(file_path):
    """Load and return the sign language dataset"""
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    print(f"Data loaded successfully with shape: {df.shape}")
    return df

def analyze_word_distributions(df):
    """Analyze distribution of signs and features by word"""
    plt.figure(figsize=(15, 6))
    word_counts = df['word'].value_counts().sort_index()
    sns.barplot(x=word_counts.index, y=word_counts.values)
    plt.title('Number of Samples per Letter')
    plt.xlabel('Letter')
    plt.ylabel('Count')
    plt.savefig('letter_distribution.png')
    plt.close()
    
    print("\n=== Samples per letter ===")
    print(word_counts)

def visualize_features_by_letter(df):
    """Create visualizations of feature distributions by letter"""
    # Sample 200 rows per letter for visualization (to keep plots manageable)
    sample_df = pd.DataFrame()
    for letter in df['word'].unique():
        letter_data = df[df['word'] == letter]
        sampled = letter_data.sample(min(len(letter_data), 200))
        sample_df = pd.concat([sample_df, sampled], ignore_index=True)
    
    # Select key features for visualization
    for feature in ['flex1', 'flex3', 'flex5', 'roll', 'pitch']:
        plt.figure(figsize=(15, 8))
        sns.boxplot(x='word', y=feature, data=sample_df)
        plt.title(f'Distribution of {feature} by Letter')
        plt.xticks(rotation=0)
        plt.savefig(f'boxplot_{feature}_by_letter.png')
        plt.close()

def analyze_feature_correlations(df):
    """Analyze correlations between features"""
    # Select only numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col not in ['run_no', 'hand', 'timestamp']]
    
    # Compute correlation matrix
    corr = df[numeric_cols].corr()
    
    # Plot correlation matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig('feature_correlations.png')
    plt.close()
    
    # Find highly correlated features
    print("\n=== Highly correlated features (|corr| > 0.7) ===")
    high_corr_list = []
    for i, x in enumerate(corr.columns):
        for j, y in enumerate(corr.columns):
            if i != j and abs(corr.iloc[i, j]) > 0.7:
                high_corr_list.append({
                    'Feature1': x, 
                    'Feature2': y, 
                    'Correlation': corr.iloc[i, j]
                })
    
    if high_corr_list:
        high_corr = pd.DataFrame(high_corr_list)
        high_corr = high_corr.sort_values('Correlation', ascending=False)
        print(high_corr)
    else:
        print("No highly correlated features found.")

def perform_pca_visualization(df):
    """Perform PCA to visualize letter separation"""
    # Sample data for PCA (using more data makes the plot too crowded)
    sample_df = pd.DataFrame()
    for letter in df['word'].unique():
        letter_data = df[df['word'] == letter]
        sampled = letter_data.sample(min(len(letter_data), 200))
        sample_df = pd.concat([sample_df, sampled], ignore_index=True)
    
    # Select features for PCA
    features = [col for col in df.columns if col not in ['run_no', 'word', 'hand', 'timestamp']]
    
    # Standardize features
    X = sample_df[features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # Create DataFrame with PCA results
    pca_df = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
    pca_df['word'] = sample_df['word'].values
    
    # Plot PCA results
    plt.figure(figsize=(12, 10))
    
    # Plot for each letter
    for letter in sorted(pca_df['word'].unique()):
        letter_data = pca_df[pca_df['word'] == letter]
        plt.scatter(letter_data['PC1'], letter_data['PC2'], label=letter, alpha=0.7)
    
    plt.title('PCA of BSL Sign Language Data')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('pca_letters.png')
    plt.close()
    
    print(f"\n=== PCA Explained Variance ===")
    print(f"PC1: {pca.explained_variance_ratio_[0]:.2%}")
    print(f"PC2: {pca.explained_variance_ratio_[1]:.2%}")

def analyze_letter_separation(df):
    """Evaluate how well we can separate letters with a simple model"""
    # Prepare features and target
    X = df.drop(['word', 'timestamp', 'run_no'], axis=1)
    y = df['word']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("\n=== Training a Random Forest classifier to assess letter separation ===")
    # Train a simple model
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    
    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred))
    
    # Plot feature importances
    feature_importances = pd.DataFrame(
        model.feature_importances_,
        index=X.columns,
        columns=['importance']
    ).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x=feature_importances.index, y=feature_importances['importance'])
    plt.title('Feature Importances for Letter Classification')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('feature_importances.png')
    plt.close()
    
    print("\n=== Top 5 most important features ===")
    print(feature_importances.head(5))
    
    # Compute and plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(20, 16))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=sorted(y.unique()), 
                yticklabels=sorted(y.unique()))
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()

def analyze_run_consistency(df):
    """Analyze consistency across different runs"""
    # Get average feature values for each run and letter
    run_letter_avg = df.groupby(['run_no', 'word']).mean().reset_index()
    
    # Choose a specific letter to analyze
    for letter in ['a', 'm', 'z']:  # Beginning, middle, and end of alphabet
        letter_data = run_letter_avg[run_letter_avg['word'] == letter]
        
        plt.figure(figsize=(15, 8))
        for feature in ['flex1', 'flex5', 'roll']:
            # Normalize to 0-1 range for comparison
            feature_vals = letter_data[feature].values
            if feature_vals.max() != feature_vals.min():  # Avoid division by zero
                normalized = (feature_vals - feature_vals.min()) / (feature_vals.max() - feature_vals.min())
                plt.plot(letter_data['run_no'], normalized, label=feature)
        
        plt.title(f'Feature Consistency Across Runs for Letter "{letter}"')
        plt.xlabel('Run Number')
        plt.ylabel('Normalized Feature Value')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'run_consistency_{letter}.png')
        plt.close()

def main():
    # Check if the user provided a file path
    file_path = "combined_data.csv"
    
    if not file_path:
        file_path = 'combined_data.csv'
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return
    
    # Load the data
    df = load_data(file_path)
    
    # Fix duplicate letters if present
    # Count unique values to avoid processing if data is clean
    duplicate_check = df['word'].value_counts()
    print(f"Found {len(duplicate_check)} unique letters in dataset")
    
    # Analyze word distributions
    analyze_word_distributions(df)
    
    # Visualize features by letter
    visualize_features_by_letter(df)
    
    # Analyze feature correlations
    analyze_feature_correlations(df)
    
    # PCA visualization
    perform_pca_visualization(df)
    
    # Analyze letter separation
    analyze_letter_separation(df)
    
    # Analyze run consistency
    analyze_run_consistency(df)
    
    print("\nAnalysis complete! Check the current directory for generated plots.")

if __name__ == "__main__":
    main()