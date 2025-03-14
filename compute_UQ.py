import os
# import sys
# import glob
# import time
import numpy as np
import nibabel as nib
# import torch
import multiprocessing as mp
from tqdm import tqdm

# Import the uncertainty quantification functions
from uncertain_quantification import (
    variance, standard_deviation, coefficient_of_variation, 
    entropy, interquartile_range, range_width, 
    confidence_interval_width, predictive_variance, 
    mutual_information
)

# Remove the parent directory path append
# sys.path.append("../")

# Define the model name directly without selection
model_name = "Theseus_v2_181_200_rdp1"
print(f"Using model: {model_name}")

# Setup dictionaries
test_dict = {}
test_dict["project_name"] = model_name
test_dict["save_folder"] = "project_dir/"+test_dict["project_name"]+"/"
test_dict["eval_save_folder"] = "array"
test_dict["uncertainty_save_folder"] = "uncertainty"

# Create uncertainty save folder if it doesn't exist
uncertainty_save_path = os.path.join(test_dict["save_folder"], test_dict["uncertainty_save_folder"])
os.makedirs(uncertainty_save_path, exist_ok=True)

# Define normalization parameters
min_val = -1024
max_val = 2976
norm_range = [-1, 1]

# Function to denormalize data from [-1, 1] to [min_val, max_val]
def denormalize(data, min_val=-1024, max_val=2976, norm_range=[0, 1]):
    """Denormalize data from norm_range to [min_val, max_val]"""
    norm_min, norm_max = norm_range
    norm_range_width = norm_max - norm_min
    actual_range_width = max_val - min_val
    
    return ((data - norm_min) / norm_range_width) * actual_range_width + min_val

# Define a function to process and save a single metric
def process_and_save_metric(metric_func, metric_name, data, affine, header, output_path, prefix):
    """Process a single uncertainty metric and save it as a NIFTI file"""
    try:
        # Calculate the metric
        metric_data = metric_func(data, voxel_wise=True)
        
        # Create output filename
        output_filename = os.path.join(output_path, f"{prefix}_{metric_name}.nii.gz")
        
        # Create and save nifti file
        metric_nifti = nib.Nifti1Image(metric_data, affine, header)
        nib.save(metric_nifti, output_filename)
        
        return True
    except Exception as e:
        print(f"Error processing {metric_name}: {str(e)}")
        return False

def process_single_file(file_path, test_dict):
    """Process a single file with all uncertainty metrics"""
    try:
        # Skip if file already processed
        output_array_path = os.path.join(
            test_dict["save_folder"],
            test_dict["eval_save_folder"],
            os.path.basename(file_path).replace(".nii.gz", "_array.npy")
        )
        
        if not os.path.exists(output_array_path):
            print(f"Warning: No predictions found for {file_path}")
            return False
            
        # Load and process the file
        output_array = np.load(output_array_path)
        denormalized_array = denormalize(output_array, min_val, max_val, norm_range)
        
        # Load original file for header and affine
        original_nifti = nib.load(file_path)
        header = original_nifti.header
        affine = original_nifti.affine
        
        # Create output prefix
        is_val = "val_list_X" in file_path
        prefix = f"{'val' if is_val else 'test'}_{os.path.basename(file_path).replace('.nii.gz', '')}"
        
        # Process all metrics
        metrics = [
            (variance, "variance"),
            (standard_deviation, "std"),
            (coefficient_of_variation, "cv"),
            (entropy, "entropy"),
            (interquartile_range, "iqr"),
            (range_width, "range"),
            (confidence_interval_width, "ci_width"),
            (predictive_variance, "pred_var"),
            (mutual_information, "mutual_info")
        ]
        
        for metric_func, metric_name in metrics:
            process_and_save_metric(
                metric_func, metric_name, denormalized_array,
                affine, header, uncertainty_save_path, prefix
            )
            
        return True
    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}")
        return False

def process_file_batch(file_batch):
    """Process a batch of files (for multiprocessing)"""
    results = []
    for file_path in file_batch:
        result = process_single_file(file_path, test_dict)
        results.append((file_path, result))
    return results

if __name__ == "__main__":
    # Load data division
    data_div = np.load(os.path.join(test_dict["save_folder"], "data_division.npy"), allow_pickle=True)[()]
    
    # Get all files to process
    all_files = []
    
    # Add test files
    test_list = data_div['test_list_X']
    test_list.sort()
    all_files.extend(["../" + path if not path.startswith("../") else path for path in test_list])
    
    # Add validation files if available
    if 'val_list_X' in data_div:
        val_list = data_div['val_list_X']
        val_list.sort()
        all_files.extend(["../" + path if not path.startswith("../") else path for path in val_list])
    
    print(f"\nTotal files to process: {len(all_files)}")
    
    # Split files into batches for multiprocessing
    num_processes = 32
    batch_size = len(all_files) // num_processes
    if len(all_files) % num_processes:
        batch_size += 1
    
    file_batches = [all_files[i:i + batch_size] for i in range(0, len(all_files), batch_size)]
    
    print(f"Split into {len(file_batches)} batches of approximately {batch_size} files each")
    
    # Process files using multiprocessing
    print("\nStarting multiprocessing pool with", num_processes, "processes")
    with mp.Pool(processes=num_processes) as pool:
        results = list(tqdm(
            pool.imap(process_file_batch, file_batches),
            total=len(file_batches),
            desc="Processing batches"
        ))
    
    # Flatten results and count successes/failures
    all_results = [item for sublist in results for item in sublist]
    successes = sum(1 for _, success in all_results if success)
    failures = sum(1 for _, success in all_results if not success)
    
    print("\nProcessing complete!")
    print(f"Successfully processed: {successes} files")
    print(f"Failed to process: {failures} files")
    
    if failures > 0:
        print("\nFailed files:")
        for file_path, success in all_results:
            if not success:
                print(f"- {file_path}")
