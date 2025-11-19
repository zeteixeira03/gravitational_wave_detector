import h5py
import numpy as np

def explore_ldc_structure(file_path):
    """Explore the structure of LDC files"""
    with h5py.File(file_path, 'r') as f:
        print("Keys in file:", list(f.keys()))
        
        def print_structure(name, obj):
            print(name, type(obj))
            
        f.visititems(print_structure)

def load_ldc_data(file_path):
    """Load LDC data - you'll need to adapt this based on the actual structure"""
    with h5py.File(file_path, 'r') as f:
        # This will depend on your specific LDC data structure
        # You'll need to inspect it first
        data = f['data'][:]
        return data

if __name__ == "__main__":
    # First, explore what's in your downloaded files
    explore_ldc_structure('data/raw/LDC2a_sangria/some_file.h5')