import os
# import sys
# import glob
# import time
import numpy as np
import nibabel as nib
# import torch

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

# Define the list of cases to process
to_do_cases = ["00008"]  # Add more case numbers as needed
print(f"Will process only files containing these case numbers: {to_do_cases}")

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
    print(f"\n{'='*20} Processing {metric_name} {'='*20}")
    
    # Calculate the metric
    print(f"Calculating {metric_name}...")
    start_time = np.datetime64('now')
    metric_data = metric_func(data, voxel_wise=True)
    end_time = np.datetime64('now')
    
    # Print metric statistics
    print(f"Completed in {(end_time - start_time) / np.timedelta64(1, 's'):.2f} seconds")
    print(f"{metric_name} statistics:")
    print(f"  Min: {np.min(metric_data):.6f}")
    print(f"  Max: {np.max(metric_data):.6f}")
    print(f"  Mean: {np.mean(metric_data):.6f}")
    print(f"  Std: {np.std(metric_data):.6f}")
    
    # Create output filename
    output_filename = os.path.join(output_path, f"{prefix}_{metric_name}.nii.gz")
    
    # Create and save nifti file
    print(f"Saving {metric_name} to {output_filename}...")
    metric_nifti = nib.Nifti1Image(metric_data, affine, header)
    nib.save(metric_nifti, output_filename)
    print(f"Successfully saved {metric_name}")
    
    return metric_data

# ==================== data division ====================

# Load data division - keep this path independent
data_div = np.load(os.path.join(test_dict["save_folder"], "data_division.npy"), allow_pickle=True)[()]

# Get test data list - original paths from data_div
test_list = data_div['test_list_X']
# Sort the test list
test_list.sort()

# Get validation data list if available - original paths from data_div
val_list = []
if 'val_list_X' in data_div:
    val_list = data_div['val_list_X']
    val_list.sort()  # Sort the validation list

# Filter lists to only include files with case numbers in to_do_cases
test_list = [f for f in test_list if any(case in f for case in to_do_cases)]
val_list = [f for f in val_list if any(case in f for case in to_do_cases)]

# ==================== Output all files found ====================
print("\n" + "="*50)
print(f"Found {len(test_list)} test files to process:")
for i, file_path in enumerate(test_list):
    print(f"  Test [{i+1}/{len(test_list)}]: {file_path}")

print("\n" + "="*50)
print(f"Found {len(val_list)} validation files to process:")
for i, file_path in enumerate(val_list):
    print(f"  Val [{i+1}/{len(val_list)}]: {file_path}")
print("="*50 + "\n")

# ==================== Process first validation file ====================
if len(val_list) > 0:
    print("\nProcessing first validation file...")
    file_path = "../"+val_list[0]
    print(f"Processing validation file: {file_path}")
    
    # Get corresponding output array path
    output_array_path = os.path.join(
        test_dict["save_folder"],
        test_dict["eval_save_folder"],
        os.path.basename(file_path).replace(".nii.gz", "_array_no_pad.npy")
    )
    
    # Check if the output array exists
    if os.path.exists(output_array_path):
        print(f"Found saved predictions at: {output_array_path}")
        
        # Load the predictions
        output_array = np.load(output_array_path)
        print(f"Loaded array shape: {output_array.shape}")
        
        # Denormalize the data from [-1, 1] to [-1024, 2976]
        denormalized_array = denormalize(output_array, min_val, max_val, norm_range)
        print(f"Denormalized array range: [{np.min(denormalized_array)}, {np.max(denormalized_array)}]")
        
        # Load the original nifti file to get header and affine
        original_nifti = nib.load(file_path)
        header = original_nifti.header
        affine = original_nifti.affine
        
        # Create output prefix
        output_prefix = f"val_{os.path.basename(file_path).replace('.nii.gz', '')}"
        
        # Process each metric one by one
        print("\nProcessing uncertainty metrics one by one...")
        
        # Variance
        process_and_save_metric(
            variance, "variance", denormalized_array, 
            affine, header, uncertainty_save_path, output_prefix
        )
        
        # Standard Deviation
        process_and_save_metric(
            standard_deviation, "std", denormalized_array, 
            affine, header, uncertainty_save_path, output_prefix
        )
        
        # Coefficient of Variation
        process_and_save_metric(
            coefficient_of_variation, "cv", denormalized_array, 
            affine, header, uncertainty_save_path, output_prefix
        )
        
        # Entropy
        process_and_save_metric(
            entropy, "entropy", denormalized_array, 
            affine, header, uncertainty_save_path, output_prefix
        )
        
        # Interquartile Range
        process_and_save_metric(
            interquartile_range, "iqr", denormalized_array, 
            affine, header, uncertainty_save_path, output_prefix
        )
        
        # Range Width
        process_and_save_metric(
            range_width, "range", denormalized_array, 
            affine, header, uncertainty_save_path, output_prefix
        )
        
        # Confidence Interval Width
        process_and_save_metric(
            confidence_interval_width, "ci_width", denormalized_array, 
            affine, header, uncertainty_save_path, output_prefix
        )
        
        # Predictive Variance
        process_and_save_metric(
            predictive_variance, "pred_var", denormalized_array, 
            affine, header, uncertainty_save_path, output_prefix
        )
        
        # Mutual Information
        process_and_save_metric(
            mutual_information, "mutual_info", denormalized_array, 
            affine, header, uncertainty_save_path, output_prefix
        )
        
        print("\nCompleted processing all metrics for validation file")
    else:
        print(f"Warning: No predictions found for {file_path}")

# ==================== Process first test file ====================
if len(test_list) > 0:
    print("\nProcessing first test file...")
    file_path = "../"+test_list[0]
    print(f"Processing test file: {file_path}")
    
    # Get corresponding output array path
    output_array_path = os.path.join(
        test_dict["save_folder"],
        test_dict["eval_save_folder"],
        os.path.basename(file_path).replace(".nii.gz", "_array_no_pad.npy")
    )
    
    # Check if the output array exists
    if os.path.exists(output_array_path):
        print(f"Found saved predictions at: {output_array_path}")
        
        # Load the predictions
        output_array = np.load(output_array_path)
        print(f"Loaded array shape: {output_array.shape}")
        
        # Denormalize the data from [-1, 1] to [-1024, 2976]
        denormalized_array = denormalize(output_array, min_val, max_val, norm_range)
        print(f"Denormalized array range: [{np.min(denormalized_array)}, {np.max(denormalized_array)}]")
        
        # Load the original nifti file to get header and affine
        original_nifti = nib.load(file_path)
        header = original_nifti.header
        affine = original_nifti.affine
        
        # Create output prefix
        output_prefix = f"test_{os.path.basename(file_path).replace('.nii.gz', '')}"
        
        # Process each metric one by one
        print("\nProcessing uncertainty metrics one by one...")
        
        # Variance
        process_and_save_metric(
            variance, "variance", denormalized_array, 
            affine, header, uncertainty_save_path, output_prefix
        )
        
        # Standard Deviation
        process_and_save_metric(
            standard_deviation, "std", denormalized_array, 
            affine, header, uncertainty_save_path, output_prefix
        )
        
        # Coefficient of Variation
        process_and_save_metric(
            coefficient_of_variation, "cv", denormalized_array, 
            affine, header, uncertainty_save_path, output_prefix
        )
        
        # Entropy
        process_and_save_metric(
            entropy, "entropy", denormalized_array, 
            affine, header, uncertainty_save_path, output_prefix
        )
        
        # Interquartile Range
        process_and_save_metric(
            interquartile_range, "iqr", denormalized_array, 
            affine, header, uncertainty_save_path, output_prefix
        )
        
        # Range Width
        process_and_save_metric(
            range_width, "range", denormalized_array, 
            affine, header, uncertainty_save_path, output_prefix
        )
        
        # Confidence Interval Width
        process_and_save_metric(
            confidence_interval_width, "ci_width", denormalized_array, 
            affine, header, uncertainty_save_path, output_prefix
        )
        
        # Predictive Variance
        process_and_save_metric(
            predictive_variance, "pred_var", denormalized_array, 
            affine, header, uncertainty_save_path, output_prefix
        )
        
        # Mutual Information
        process_and_save_metric(
            mutual_information, "mutual_info", denormalized_array, 
            affine, header, uncertainty_save_path, output_prefix
        )
        
        print("\nCompleted processing all metrics for test file")
    else:
        print(f"Warning: No predictions found for {file_path}")
