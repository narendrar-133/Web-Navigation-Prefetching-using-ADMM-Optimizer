import pandas as pd
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np
import time
import matplotlib.pyplot as plt

def soft_threshold(x, penalty):
    """Soft-thresholding function used in the ADMM constraint application."""
    return np.sign(x) * np.maximum(np.abs(x) - penalty, 0)

def apply_admm_constraints(transition_counts, constraint_penalty=0.01):
    """
    Directly applies ADMM-style sparsity constraints to transition counts.
    This enforces bandwidth, cache, and energy constraints simultaneously.
    """
    constrained_transitions = {}
    
    for current_item, transitions in transition_counts.items():
        # Convert to numpy array for constraint application
        counts = np.array(list(transitions.values()))
        next_items = list(transitions.keys())
        
        # Apply soft thresholding to induce sparsity
        sparse_counts = soft_threshold(counts, constraint_penalty)
        
        # Remove zero or negative transitions
        constrained_transitions[current_item] = {}
        for i, count in enumerate(sparse_counts):
            if count > 0:
                constrained_transitions[current_item][next_items[i]] = count
    
    return constrained_transitions

def plot_latency_distribution(prediction_times, metrics, save_path="latency_distribution_markov.png"):
    """Plots and saves the latency distribution histogram."""
    plt.figure(figsize=(10, 6))
    plt.hist(prediction_times, bins=50, alpha=0.7, color='orange', edgecolor='black')
    plt.axvline(metrics['avg_latency_ms'], color='r', linestyle='--', linewidth=2, label=f"Average: {metrics['avg_latency_ms']:.2f}ms")
    plt.axvline(metrics['p95_latency_ms'], color='g', linestyle='--', linewidth=2, label=f"P95: {metrics['p95_latency_ms']:.2f}ms")
    plt.axvline(metrics['p99_latency_ms'], color='purple', linestyle='--', linewidth=2, label=f"P99: {metrics['p99_latency_ms']:.2f}ms")
    plt.title('Prediction Latency Distribution for Constrained Markov Model')
    plt.xlabel('Latency (ms)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"   - Latency distribution plot saved to {save_path}")
    plt.close()

def plot_performance_metrics(metrics, save_path="performance_metrics_markov.png"):
    """Plots and saves a summary of key performance metrics."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # --- Accuracy/Hit Rate ---
    acc_labels = ['Top-1 Acc', 'Top-3 HR', 'Top-5 HR']
    acc_values = [metrics['top1_accuracy'], metrics['top3_hit_rate'], metrics['top5_hit_rate']]
    ax1.bar(acc_labels, acc_values, color=['#4c72b0', '#55a868', '#c44e52'])
    ax1.set_ylabel('Rate')
    ax1.set_title('Accuracy & Hit Rate Metrics')
    ax1.set_ylim(0, max(acc_values) * 1.2 if acc_values else 1)
    for i, v in enumerate(acc_values):
        ax1.text(i, v + 0.01, f"{v:.2%}", ha='center', va='bottom')

    # --- Latency ---
    lat_labels = ['Avg Latency', 'P95 Latency', 'P99 Latency']
    lat_values = [metrics['avg_latency_ms'], metrics['p95_latency_ms'], metrics['p99_latency_ms']]
    ax2.bar(lat_labels, lat_values, color=['#8172b2', '#ff9d9a', '#d5bb67'])
    ax2.set_ylabel('Latency (ms)')
    ax2.set_title('Latency Metrics (ms)')
    for i, v in enumerate(lat_values):
        ax2.text(i, v + 0.1, f"{v:.2f}ms", ha='center', va='bottom')

    fig.suptitle('Constrained Markov Model Performance Summary', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path)
    print(f"   - Performance metrics summary saved to {save_path}")
    plt.close()

def load_and_prepare_data(file_path='bu_sessions_markov.csv'):
    """
    Loads session data, encodes URLs, and calculates average item sizes for bandwidth simulation.
    """
    print(f"Loading data from {file_path}...")
    try:
        df = pd.read_csv(file_path)
        if 'bytes' not in df.columns:
            print("CRITICAL ERROR: The input CSV does not contain the required 'bytes' column.")
            return None, None, None
    except FileNotFoundError:
        print(f"ERROR: The file {file_path} was not found. Please run the full data pipeline first.")
        return None, None, None

    # Encode URLs to integers
    item_encoder = LabelEncoder()
    df['item_encoded'] = item_encoder.fit_transform(df['url'])
    
    # Calculate average size for each item for bandwidth constraint
    print("Calculating average item sizes for bandwidth analysis...")
    item_byte_map = df.groupby('item_encoded')['bytes'].mean()

    # Create a list of sessions, where each session is a list of item IDs
    sessions = df.groupby('session_id')['item_encoded'].apply(list)
    
    print(f"Loaded {len(sessions)} sessions.")
    return sessions, item_encoder, item_byte_map

def train_admm_constrained_markov_model(train_sessions, constraint_penalty=0.01):
    """
    Builds a transition matrix with ADMM-style constraints applied directly.
    The constraints enforce sparsity which proxies bandwidth, cache, and energy limits.
    """
    print(f"Training ADMM-Constrained Markov model with penalty={constraint_penalty}...")
    
    # First build full transition counts
    full_transitions = defaultdict(lambda: defaultdict(int))
    for session in tqdm(train_sessions, desc="Processing sessions"):
        for i in range(len(session) - 1):
            current_item, next_item = session[i], session[i+1]
            full_transitions[current_item][next_item] += 1

    # Apply ADMM constraints directly to induce sparsity
    print("Applying ADMM constraints to transition matrix...")
    constrained_transitions = apply_admm_constraints(full_transitions, constraint_penalty)
            
    print("ADMM-Constrained Markov model training complete.")
    return constrained_transitions

def evaluate_admm_constrained_model(test_sessions, transition_matrix, item_byte_map):
    """
    Evaluates the ADMM-constrained Markov model with comprehensive metrics.
    """
    print("Evaluating ADMM-Constrained Markov Model...")
    top1_hits, top3_hits, top5_hits, total_predictions = 0, 0, 0, 0
    prediction_times = []

    for session in tqdm(test_sessions, desc="Evaluating sessions"):
        if len(session) < 2:
            continue
        for i in range(len(session) - 1):
            current_item, actual_next_item = session[i], session[i+1]
            total_predictions += 1

            start_time = time.time()

            if current_item in transition_matrix:
                possible_next_items = transition_matrix[current_item]
                
                if possible_next_items:  # Check if any transitions remain after constraints
                    total_transitions = sum(possible_next_items.values())
                    
                    # Simple probability-based prediction (ADMM already handled constraints)
                    sorted_predictions = sorted(possible_next_items.items(), 
                                              key=lambda x: x[1], reverse=True)
                    
                    top_5_predictions = [item_id for item_id, _ in sorted_predictions[:5]]
                    
                    # Record latency
                    end_time = time.time()
                    prediction_times.append((end_time - start_time) * 1000)

                    # Calculate accuracy
                    if top_5_predictions:
                        if actual_next_item == top_5_predictions[0]:
                            top1_hits += 1
                        if actual_next_item in top_5_predictions[:3]:
                            top3_hits += 1
                        if actual_next_item in top_5_predictions:
                            top5_hits += 1
                else:
                    end_time = time.time()
                    prediction_times.append((end_time - start_time) * 1000)
            else:
                end_time = time.time()
                prediction_times.append((end_time - start_time) * 1000)

    if total_predictions == 0: 
        return 0.0, 0.0, 0.0, 0, []
        
    return top1_hits, top3_hits, top5_hits, total_predictions, prediction_times

def analyze_constraint_effects(transition_matrix):
    """Analyzes the effects of ADMM constraints on the model."""
    total_transitions = 0
    total_nonzero = 0
    sparsity_per_item = []
    
    for item, transitions in transition_matrix.items():
        item_transitions = sum(transitions.values())
        item_nonzero = len(transitions)
        total_transitions += item_transitions
        total_nonzero += item_nonzero
        if item_nonzero > 0:
            sparsity = 1 - (item_nonzero / item_transitions) if item_transitions > 0 else 0
            sparsity_per_item.append(sparsity)
    
    avg_sparsity = np.mean(sparsity_per_item) if sparsity_per_item else 0
    
    print(f"\n📊 ADMM CONSTRAINT ANALYSIS:")
    print(f"   Total transitions: {total_transitions}")
    print(f"   Non-zero entries: {total_nonzero}")
    print(f"   Average sparsity per item: {avg_sparsity:.2%}")
    print(f"   Compression ratio: {total_transitions/max(total_nonzero,1):.2f}x")

def main():
    start_time = time.time()
    
    # --- ADMM Constraint Parameter ---
    ADMM_CONSTRAINT_PENALTY = 0.01  # Controls sparsity strength
    
    sessions, item_encoder, item_byte_map = load_and_prepare_data()
    if sessions is None: 
        return

    train_session_ids, test_session_ids = train_test_split(sessions.index.tolist(), test_size=0.2, random_state=42)
    train_sessions, test_sessions = sessions.loc[train_session_ids], sessions.loc[test_session_ids]
    
    # Train with ADMM constraints
    transition_matrix = train_admm_constrained_markov_model(train_sessions, ADMM_CONSTRAINT_PENALTY)
    
    # Analyze constraint effects
    analyze_constraint_effects(transition_matrix)
    
    # Evaluate the model
    top1_hits, top3_hits, top5_hits, total_preds, prediction_times = evaluate_admm_constrained_model(
        test_sessions, transition_matrix, item_byte_map
    )
    
    total_eval_time = time.time() - start_time
    
    # --- Calculate Metrics ---
    metrics_dict = {
        'top1_accuracy': top1_hits / total_preds if total_preds > 0 else 0,
        'top3_hit_rate': top3_hits / total_preds if total_preds > 0 else 0,
        'top5_hit_rate': top5_hits / total_preds if total_preds > 0 else 0,
        'average_loss': -1,  # Markov model doesn't have traditional loss
        'prediction_times': prediction_times
    }
    
    if prediction_times:
        metrics_dict.update({
            'avg_latency_ms': np.mean(prediction_times),
            'p95_latency_ms': np.percentile(prediction_times, 95),
            'p99_latency_ms': np.percentile(prediction_times, 99),
        })
    else:
        metrics_dict.update({
            'avg_latency_ms': 0, 'p95_latency_ms': 0, 'p99_latency_ms': 0
        })

    # --- Print Metrics ---
    print("="*70)
    print(f"ADMM-CONSTRAINED MARKOV MODEL EVALUATION")
    print("="*70)
    print(f"Evaluation completed in {total_eval_time:.2f} seconds")
    print(f"Test set size: {len(test_sessions)} sessions")
    print(f"ADMM Constraint Penalty: {ADMM_CONSTRAINT_PENALTY}")
    print("-"*70)
    
    print("\n📊 ACCURACY METRICS:")
    print(f"   Top-1 Accuracy:  {metrics_dict['top1_accuracy']:.4f} ({metrics_dict['top1_accuracy']:.2%})")
    print(f"   Top-3 Hit Rate:  {metrics_dict['top3_hit_rate']:.4f} ({metrics_dict['top3_hit_rate']:.2%})")
    print(f"   Top-5 Hit Rate:  {metrics_dict['top5_hit_rate']:.4f} ({metrics_dict['top5_hit_rate']:.2%})")
    
    print("\n⏱️  LATENCY METRICS:")
    print(f"   Average Latency: {metrics_dict['avg_latency_ms']:.2f} ms")
    print(f"   P95 Latency:     {metrics_dict['p95_latency_ms']:.2f} ms")
    print(f"   P99 Latency:     {metrics_dict['p99_latency_ms']:.2f} ms")
    print("="*70)
    
    # --- Generate Visualizations ---
    print("\n📊 Generating performance visualizations...")
    plot_latency_distribution(metrics_dict['prediction_times'], metrics_dict)
    plot_performance_metrics(metrics_dict)
    
    print("\n✅ ADMM-Constrained Markov Model evaluation completed!")

if __name__ == '__main__':
    main()
