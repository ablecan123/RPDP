import os
import numpy as np
import pandas as pd
import shutil

def preprocess():
    # Paths
    attr_path = os.path.join('dataset', 'attributes.csv')
    label_path = os.path.join('dataset', 'label.csv')
    
    # Read CSVs
    # Assuming first row is header based on inspection
    print("Reading CSV files...")
    attributes = pd.read_csv(attr_path)
    labels = pd.read_csv(label_path)
    
    # Convert to numpy
    X = attributes.values
    y = labels.values.flatten()
    
    print(f"Data shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    
    # Verify total samples
    if len(X) != 8000:
        print(f"Warning: Expected 8000 samples, found {len(X)}. Check header assumption.")
        # If it's 8001, maybe header was treated as data?
        # attributes.csv line 1 is 0,1,2... which is definitely header.
        # label.csv line 1 is 0. If that's a label, then we missed one.
        # But if attributes has header, label likely does too.
        # Let's proceed assuming 8000 data points.

    # Create directories
    base_dir = 'npy_data'
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)
    
    train_dir = os.path.join(base_dir, 'Train_data')
    test_dir = os.path.join(base_dir, 'Test_data')
    
    os.makedirs(train_dir)
    for i in range(8):
        os.makedirs(os.path.join(test_dir, str(i)))
        
    print("Directories created.")

    # Counters for filenames
    # Train: 600 normal
    # Test 0: 400 normal
    # Test 1-7: 1000 each
    
    train_count = 0
    test_counts = {i: 0 for i in range(8)}
    
    # Processing
    # We need to be careful about the order. 
    # The paper/code implies a specific split. 
    # Usually, first N are train, next M are test.
    
    # Let's process by class
    for label_val in range(8):
        indices = np.where(y == label_val)[0]
        data_subset = X[indices]
        print(f"Processing Label {label_val}: {len(data_subset)} samples")
        
        if label_val == 0:
            # Normal data: Split into Train (600) and Test (400)
            # Assuming the first 600 are for training as per standard practice
            train_subset = data_subset[:600]
            test_subset = data_subset[600:]
            
            # Save Train
            for sample in train_subset:
                np.save(os.path.join(train_dir, f"{train_count}.npy"), sample)
                train_count += 1
                
            # Save Test
            for sample in test_subset:
                np.save(os.path.join(test_dir, '0', f"{test_counts[0]}.npy"), sample)
                test_counts[0] += 1
                
        else:
            # Fault data: All to Test
            for sample in data_subset:
                np.save(os.path.join(test_dir, str(label_val), f"{test_counts[label_val]}.npy"), sample)
                test_counts[label_val] += 1
                
    print("Preprocessing complete.")
    print(f"Train samples: {train_count}")
    print(f"Test samples per class: {test_counts}")

if __name__ == "__main__":
    preprocess()
