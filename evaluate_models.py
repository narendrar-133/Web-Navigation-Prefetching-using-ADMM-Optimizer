import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import time
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

# --- Data Processing Functions ---

def load_and_process_data(file_path='bu_sessions.csv'):
    """Loads and preprocesses the sessionized Boston University dataset."""
    print(f"Loading and preprocessing data from {file_path}...")
    try:
        df = pd.read_csv(file_path, low_memory=False)
    except FileNotFoundError:
        print(f"ERROR: The file {file_path} was not found. Please ensure it exists.")
        return None, None
        
    df = df.dropna(subset=['session_id', 'url', 'timestamp'])
    
    # Encode the URLs into integer IDs
    item_encoder = LabelEncoder()
    df['item_encoded'] = item_encoder.fit_transform(df['url'])
    print(f"Data loaded with {len(df)} events from {df['session_id'].nunique()} sessions.")
    return df, item_encoder

def create_sequences_from_sessions(df, seq_length=10):
    """Creates sequences from the sessionized data."""
    print(f"Creating sequences of length {seq_length}...")
    df.sort_values(by=['session_id', 'timestamp'], inplace=True)
    
    sequences, labels = [], []
    for _, group in df.groupby('session_id'):
        items = group['item_encoded'].tolist()
        if len(items) > seq_length:
            for i in range(len(items) - seq_length):
                sequences.append(items[i:i + seq_length])
                labels.append(items[i + seq_length])
                
    print(f"Created {len(sequences)} sequences.")
    return np.array(sequences), np.array(labels)

# --- Comprehensive Evaluation Functions ---

def evaluate_top_k_metrics_memory_safe(model, X_test, y_test, chunk_size=500):
    """
    Calculates comprehensive metrics including Top-1, Top-3 accuracy, and latency.
    """
    print(f"\nEvaluating model on the test set in chunks of {chunk_size}...")
    
    total_samples = len(X_test)
    top1_hits = 0
    top3_hits = 0
    top5_hits = 0
    total_loss = 0
    
    # For loss calculation
    criterion = tf.keras.losses.SparseCategoricalCrossentropy()
    
    # Latency measurements
    prediction_times = []
    
    # Process data in smaller chunks
    for i in range(0, total_samples, chunk_size):
        end_idx = min(i + chunk_size, total_samples)
        
        print(f"Processing chunk {i//chunk_size + 1}/{(total_samples + chunk_size - 1)//chunk_size} "
              f"(samples {i}-{end_idx-1})")
        
        # Get current chunk
        X_chunk = X_test[i:end_idx]
        y_chunk = y_test[i:end_idx]
        
        # Measure prediction latency
        start_time = time.time()
        predictions = model.predict(X_chunk, batch_size=16, verbose=0)
        end_time = time.time()
        
        # Calculate latency for this chunk (per sample)
        chunk_latency = (end_time - start_time) / len(X_chunk)
        prediction_times.extend([chunk_latency] * len(X_chunk))
        
        # Calculate loss for this chunk
        chunk_loss = criterion(y_chunk, predictions).numpy()
        total_loss += chunk_loss * len(X_chunk)
        
        # Calculate top-k metrics for this chunk
        top_5_preds = np.argsort(predictions, axis=1)[:, -5:]
        
        # Top-1 accuracy for chunk
        top1_hits += np.sum(y_chunk == top_5_preds[:, -1])
        
        # Top-3 hit rate for chunk
        top3_hits += (y_chunk[:, None] == top_5_preds[:, -3:]).any(axis=1).sum()
        
        # Top-5 hit rate for chunk
        top5_hits += (y_chunk[:, None] == top_5_preds).any(axis=1).sum()
    
    # Calculate final metrics
    top1_accuracy = top1_hits / total_samples
    top3_hit_rate = top3_hits / total_samples
    top5_hit_rate = top5_hits / total_samples
    average_loss = total_loss / total_samples
    
    # Latency statistics
    avg_latency = np.mean(prediction_times) * 1000  # Convert to milliseconds
    p95_latency = np.percentile(prediction_times, 95) * 1000
    p99_latency = np.percentile(prediction_times, 99) * 1000
    
    return {
        'top1_accuracy': top1_accuracy,
        'top3_hit_rate': top3_hit_rate,
        'top5_hit_rate': top5_hit_rate,
        'average_loss': average_loss,
        'avg_latency_ms': avg_latency,
        'p95_latency_ms': p95_latency,
        'p99_latency_ms': p99_latency,
        'prediction_times': prediction_times
    }

def plot_training_history(history):
    """Plots training and validation loss curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    ax1.plot(history.history['loss'], label='Training Loss', linewidth=2)
    if 'val_loss' in history.history:
        ax1.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    ax1.set_title('Model Loss During Training')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot accuracy
    if 'accuracy' in history.history:
        ax2.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
        if 'val_accuracy' in history.history:
            ax2.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        ax2.set_title('Model Accuracy During Training')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_history_lstm.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_latency_distribution(prediction_times, metrics_dict):
    """Plots the latency distribution of predictions."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Convert to milliseconds for better readability
    latencies_ms = np.array(prediction_times) * 1000
    
    # Histogram
    ax1.hist(latencies_ms, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.axvline(metrics_dict['avg_latency_ms'], color='red', linestyle='--', 
                label=f"Average: {metrics_dict['avg_latency_ms']:.2f}ms")
    ax1.axvline(metrics_dict['p95_latency_ms'], color='orange', linestyle='--', 
                label=f"P95: {metrics_dict['p95_latency_ms']:.2f}ms")
    ax1.set_xlabel('Latency (milliseconds)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Prediction Latency Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Box plot
    ax2.boxplot(latencies_ms, vert=True, patch_artist=True,
               boxprops=dict(facecolor='lightgreen', color='darkgreen'),
               medianprops=dict(color='red'))
    ax2.set_ylabel('Latency (milliseconds)')
    ax2.set_title('Prediction Latency Box Plot')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('latency_distribution_lstm.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_performance_metrics(metrics_dict):
    """Plots a comprehensive summary of performance metrics."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Accuracy Metrics
    accuracy_metrics = ['Top-1 Accuracy', 'Top-3 Hit Rate', 'Top-5 Hit Rate']
    accuracy_values = [
        metrics_dict['top1_accuracy'],
        metrics_dict['top3_hit_rate'], 
        metrics_dict['top5_hit_rate']
    ]
    
    bars = ax1.bar(accuracy_metrics, accuracy_values, color=['#2E86AB', '#A23B72', '#F18F01'])
    ax1.set_title('Accuracy Metrics', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Score')
    ax1.set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, value in zip(bars, accuracy_values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Loss and Latency Overview
    metrics_overview = ['Average Loss', 'Avg Latency (ms)']
    overview_values = [
        metrics_dict['average_loss'],
        metrics_dict['avg_latency_ms']
    ]
    
    bars2 = ax2.bar(metrics_overview, overview_values, color=['#C73E1D', '#6A8EAE'])
    ax2.set_title('Loss and Latency Overview', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Value')
    
    # Add value labels on bars
    for bar, value in zip(bars2, overview_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + max(overview_values)*0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Latency Percentiles
    latency_metrics = ['Average', 'P95', 'P99']
    latency_values = [
        metrics_dict['avg_latency_ms'],
        metrics_dict['p95_latency_ms'],
        metrics_dict['p99_latency_ms']
    ]
    
    bars3 = ax3.bar(latency_metrics, latency_values, color=['#4CB963', '#E6AF2E', '#A31621'])
    ax3.set_title('Latency Statistics (milliseconds)', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Latency (ms)')
    
    # Add value labels on bars
    for bar, value in zip(bars3, latency_values):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + max(latency_values)*0.01,
                f'{value:.2f}ms', ha='center', va='bottom', fontweight='bold')
    
    # Metrics Summary Table
    ax4.axis('off')
    summary_data = [
        ['Metric', 'Value'],
        ['Top-1 Accuracy', f"{metrics_dict['top1_accuracy']:.4f}"],
        ['Top-3 Hit Rate', f"{metrics_dict['top3_hit_rate']:.4f}"],
        ['Top-5 Hit Rate', f"{metrics_dict['top5_hit_rate']:.4f}"],
        ['Average Loss', f"{metrics_dict['average_loss']:.4f}"],
        ['Avg Latency', f"{metrics_dict['avg_latency_ms']:.2f} ms"],
        ['P95 Latency', f"{metrics_dict['p95_latency_ms']:.2f} ms"],
        ['P99 Latency', f"{metrics_dict['p99_latency_ms']:.2f} ms"]
    ]
    
    table = ax4.table(cellText=summary_data, 
                     cellLoc='center', 
                     loc='center',
                     colWidths=[0.4, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    ax4.set_title('Performance Metrics Summary LSTM', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('performance_metrics_summary_lstm.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    # --- Parameters ---
    SEQ_LENGTH = 10
    MODEL_PATH = 'saved_models/bu_admm_constrained_lstm.h5'
    CHUNK_SIZE = 500
    
    # --- 1. Recreate the Test Set ---
    df, item_encoder = load_and_process_data()
    if df is None:
        return

    sequences, labels = create_sequences_from_sessions(df, seq_length=SEQ_LENGTH)
    if len(sequences) == 0:
        print("CRITICAL ERROR: No sequences were created.")
        return

    # Create train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        sequences, labels, test_size=0.2, random_state=42
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # --- 2. Load the Pre-trained Model ---
    print(f"\nLoading pre-trained ADMM model from '{MODEL_PATH}'...")
    if not os.path.exists(MODEL_PATH):
        print(f"CRITICAL ERROR: Model file not found at '{MODEL_PATH}'.")
        print("Please ensure you have run the training script to save the model first.")
        return
        
    try:
        model = load_model(MODEL_PATH)
        print("Model loaded successfully!")
        model.summary()
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to load model. Error: {e}")
        return

    # --- 3. Comprehensive Evaluation ---
    print(f"\nStarting comprehensive evaluation with chunk size: {CHUNK_SIZE}")
    start_eval_time = time.time()
    
    metrics_dict = evaluate_top_k_metrics_memory_safe(
        model, X_test, y_test, chunk_size=CHUNK_SIZE
    )
    
    total_eval_time = time.time() - start_eval_time
    
    # --- 4. Display Results ---
    print("\n" + "="*70)
    print("ADMM-CONSTRAINED BiLSTM MODEL - COMPREHENSIVE EVALUATION RESULTS")
    print("="*70)
    print(f"Evaluation completed in {total_eval_time:.2f} seconds")
    print(f"Test set size: {len(X_test)} samples")
    print("-"*70)
    
    print("\n📊 ACCURACY METRICS:")
    print(f"   Top-1 Accuracy:  {metrics_dict['top1_accuracy']:.4f} ({metrics_dict['top1_accuracy']:.2%})")
    print(f"   Top-3 Hit Rate:  {metrics_dict['top3_hit_rate']:.4f} ({metrics_dict['top3_hit_rate']:.2%})")
    print(f"   Top-5 Hit Rate:  {metrics_dict['top5_hit_rate']:.4f} ({metrics_dict['top5_hit_rate']:.2%})")
    
    print("\n📈 LOSS METRICS:")
    print(f"   Average Loss:    {metrics_dict['average_loss']:.4f}")
    
    print("\n⏱️  LATENCY METRICS:")
    print(f"   Average Latency: {metrics_dict['avg_latency_ms']:.2f} ms")
    print(f"   P95 Latency:     {metrics_dict['p95_latency_ms']:.2f} ms")
    print(f"   P99 Latency:     {metrics_dict['p99_latency_ms']:.2f} ms")
    print("="*70)
    
    # --- 5. Generate Visualizations ---
    print("\n📊 Generating performance visualizations...")
    
    # Note: For loss graph, we need training history
    # If you have the training history saved, you can load and plot it here
    # plot_training_history(history)  # Uncomment if you have history
    
    # Plot latency distribution
    plot_latency_distribution(metrics_dict['prediction_times'], metrics_dict)
    
    # Plot comprehensive performance metrics
    plot_performance_metrics(metrics_dict)
    
    print("✅ All evaluations and visualizations completed!")
    print("💾 Results saved as:")
    print("   - performance_metrics_summary.png")
    print("   - latency_distribution.png")

if __name__ == '__main__':
    main()
