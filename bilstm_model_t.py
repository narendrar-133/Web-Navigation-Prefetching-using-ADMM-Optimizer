
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import LSTM, Dense, Embedding, Input, Bidirectional
import os

# Ensure the directory for saving the model exists
if not os.path.exists('saved_models'):
    os.makedirs('saved_models')

def soft_threshold(x, penalty):
    """Soft-thresholding function used in the ADMM update step."""
    return np.sign(x) * np.maximum(np.abs(x) - penalty, 0)

def admm_optimizer_with_constraints(model, X_train, y_train, rho=1.0, max_iter=3, constraint_penalty=0.01):
    """
    Applies ADMM optimization to induce model sparsity, which serves as a proxy for
    enforcing constraints on bandwidth, cache storage, and energy consumption.
    """
    print(f"\nStarting ADMM optimizer to enforce constraints...")
    w = model.get_weights()
    z = [np.copy(v) for v in w]
    u = [np.zeros_like(v) for v in w]

    for i in range(max_iter):
        model.fit(X_train, y_train, epochs=1, batch_size=256, verbose=1, validation_split=0.1)
        w = model.get_weights()
        for j in range(len(z)):
            z[j] = soft_threshold(w[j] + u[j], constraint_penalty / rho)
        for j in range(len(u)):
            u[j] += w[j] - z[j]
        print(f"ADMM Iteration {i+1}/{max_iter} completed.")

    model.set_weights(z)
    print("ADMM optimizer finished.")
    return model

def load_and_process_data(file_path='bu_sessions.csv'):
    """Loads and preprocesses the sessionized Boston University dataset."""
    print(f"Loading and preprocessing data from {file_path}...")
    try:
        df = pd.read_csv(file_path, low_memory=False)
    except FileNotFoundError:
        print(f"ERROR: The file {file_path} was not found. Please run sessionize_bu_data.py first.")
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
    # Sort by session and time to ensure correct sequence order
    df.sort_values(by=['session_id', 'timestamp'], inplace=True)
    
    sequences, labels = [], []
    # Group by the new session_id
    for _, group in df.groupby('session_id'):
        items = group['item_encoded'].tolist()
        if len(items) > seq_length:
            for i in range(len(items) - seq_length):
                sequences.append(items[i:i + seq_length])
                labels.append(items[i + seq_length])
                
    print(f"Created {len(sequences)} sequences.")
    return np.array(sequences), np.array(labels)

def build_bilstm_model(item_vocab_size, seq_length):
    """Builds a Bidirectional LSTM (BiLSTM) model."""
    print("Building Bidirectional LSTM (BiLSTM) Model...")
    input_seq = Input(shape=(seq_length,), name='item_sequence_input')
    emb_seq = Embedding(input_dim=item_vocab_size, output_dim=100)(input_seq)
    # Wrap the LSTM layer in a Bidirectional layer
    bilstm_layer = Bidirectional(LSTM(128))(emb_seq)
    output = Dense(item_vocab_size, activation='softmax')(bilstm_layer)

    model = Model(inputs=input_seq, outputs=output)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    print("BiLSTM model built successfully.")
    model.summary()
    return model

def main():
    df, item_encoder = load_and_process_data()
    
    if df is None:
        return

    # Using a sequence length of 10, as determined from our analysis
    SEQ_LENGTH = 10
    sequences, labels = create_sequences_from_sessions(df, seq_length=SEQ_LENGTH)
    
    if len(sequences) == 0:
        print("CRITICAL ERROR: No sequences were created. Check the SEQ_LENGTH and the input data.")
        return

    X_train, X_test, y_train, y_test = train_test_split(sequences, labels, test_size=0.2, random_state=42)
    
    item_vocab_size = len(item_encoder.classes_)

    model = build_bilstm_model(item_vocab_size, SEQ_LENGTH)
    
    print("\n--- Initial Model Training on BU Dataset ---")
    model.fit(X_train, y_train, epochs=5, batch_size=256, validation_split=0.1, verbose=1)
    
    # --- APPLYING ADMM OPTIMIZATION WITH EXPLICIT CONSTRAINTS ---
    model = admm_optimizer_with_constraints(model, X_train, y_train, constraint_penalty=0.01)
    
    model_path = 'saved_models/bu_admm_constrained_bilstm.h5'
    print(f"\nSaving ADMM-optimized model to {model_path}...")
    model.save(model_path)
    
    # --- Evaluation ---
    print("\n--- Evaluating Final ADMM-Constrained Model on BU Dataset ---")
    results = model.evaluate(X_test, y_test, batch_size=512, verbose=1)
    print(f"Test Loss: {results[0]:.4f}")
    print(f"Test Accuracy (Top-1): {results[1]:.4f} ({results[1]:.2%})")
    print("\nProcess complete.")

if __name__ == '__main__':
    main()
