import numpy as np
import pandas as pd
import os

def generate_simple_dataset(num_rows=500000):  # Maintained at 500,000 rows
    print(f"[*] Initializing parallel fraud dataset matrix generation ({num_rows:,} rows)...")
    
    np.random.seed(42)
    
    # 1. Generate features with precise data types for CUDA compatibility
    amount = np.random.uniform(10.0, 500.0, size=num_rows).astype(np.float32)
    v_features = np.random.normal(loc=0.0, scale=1.0, size=(num_rows, 15)).astype(np.float32)
    
    fraud_chance = np.random.rand(num_rows)
    classification_label = np.where(fraud_chance > 0.98, 1, 0).astype(np.int32)
    
    # Mathematical shift for class boundary separation
    amount = np.where(classification_label == 1, amount * 3.0, amount).astype(np.float32)
    
    # 2. Consolidate columns safely
    all_data = np.hstack((
        amount.reshape(-1, 1), 
        v_features, 
        classification_label.reshape(-1, 1)
    ))
    
    columns = ['Amount'] + [f'V{i}' for i in range(1, 16)] + ['Class']
    df = pd.DataFrame(all_data, columns=columns)
    
    # 3. Explicit typing to prevent host-to-device parsing mismatches
    df['Class'] = df['Class'].astype(np.int32)
    for col in columns[:-1]:
        df[col] = df[col].astype(np.float32)
    
    os.makedirs('data', exist_ok=True)
    
    output_path = 'data/synthetic_credit_fraud.csv'
    print("[*] Flushing matrix to local storage path...")
    df.to_csv(output_path, index=False)
    
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print("\n=== Dataset Generation Complete ===")
    print(f"Target File: {output_path}")
    print(f"Row Count  : {len(df):,}")
    print(f"File Size  : {file_size_mb:.2f} MB")
    print(f"Fraud Rate : {(df['Class'].sum() / num_rows) * 100:.2f}% ({df['Class'].sum()} rows)")

if __name__ == "__main__":
    generate_simple_dataset()