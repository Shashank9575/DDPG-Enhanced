import torch
import numpy as np
import time
import os

def check_device_availability():
    """Check and display available physical devices."""
    print("\n=== Device Information ===")
    
    # Check CPU availability
    print(f"Physical CPUs: {torch.get_num_threads()} (Threads)")
    print(f"CPU Device: cpu")

    # Check GPU availability
    gpus = [torch.cuda.get_device_properties(i) for i in range(torch.cuda.device_count())]
    print(f"Physical GPUs: {len(gpus)}")
    for i, gpu in enumerate(gpus):
        print(f"GPU {i}: {gpu.name} (CUDA Index: {i})")
    return len(gpus) > 0

def run_matrix_multiplication(device_name, matrix_shape=(1000, 10000)):
    """
    Perform matrix multiplication on the specified device.
    
    Args:
        device_name (str): Device to run the computation on ('cpu' or 'cuda:0')
        matrix_shape (tuple): Shape of matrices for multiplication (rows, cols)
    
    Returns:
        tuple: Resulting matrix shape and execution time
    """
    try:
        # Set device
        device = torch.device(device_name)
        start_time = time.time()
        
        # Generate random matrices
        a = torch.randn(matrix_shape[0], matrix_shape[1], device=device)
        b = torch.randn(matrix_shape[1], matrix_shape[0], device=device)
        
        # Perform matrix multiplication
        result = torch.matmul(a, b)
        
        # Ensure computation is complete (important for GPU)
        if device_name.startswith('cuda'):
            torch.cuda.synchronize()
            
        exec_time = time.time() - start_time
        return result.shape, exec_time
    except Exception as e:
        print(f"Error during matrix multiplication on {device_name}: {e}")
        return None, None

def main():
    """Main function to test matrix multiplication on CPU and GPU."""
    print("PyTorch Version:", torch.__version__)
    
    # Check available devices
    has_gpu = check_device_availability()
    
    # Define matrix shapes
    matrix_shape = (20000, 10000)
    print(f"\nTesting matrix multiplication with shape: {matrix_shape}")
    
    # Run on CPU
    print("\n=== CPU Test ===")
    cpu_shape, cpu_time = run_matrix_multiplication('cpu', matrix_shape)
    if cpu_shape:
        print(f"CPU Matrix multiplication result shape: {cpu_shape}")
        print(f"CPU Execution time: {cpu_time:.4f} seconds")
    
    # Run on GPU if available
    if has_gpu:
        print("\n=== GPU Test ===")
        gpu_shape, gpu_time = run_matrix_multiplication('cuda:0', matrix_shape)
        if gpu_shape:
            print(f"GPU Matrix multiplication result shape: {gpu_shape}")
            print(f"GPU Execution time: {gpu_time:.4f} seconds")
            if cpu_time > 0:
                print(f"Speedup (CPU time / GPU time): {cpu_time/gpu_time:.2f}x")
    else:
        print("\nSkipping GPU test due to unavailability.")

clear = lambda: os.system('cls' if os.name == 'nt' else 'clear')

if __name__ == "__main__":
    clear()
    main()