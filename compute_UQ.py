import os
import sys
import glob
import time
import numpy as np
import nibabel as nib
import torch

# Import the uncertainty quantification functions
from uncertain_quantification import (
    variance, standard_deviation, coefficient_of_variation, 
    entropy, interquartile_range, range_width, 
    confidence_interval_width, predictive_variance, 
    mutual_information, calculate_all_metrics
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
def denormalize(data, min_val=-1024, max_val=2976, norm_range=[-1, 1]):
    """Denormalize data from norm_range to [min_val, max_val]"""
    norm_min, norm_max = norm_range
    norm_range_width = norm_max - norm_min
    actual_range_width = max_val - min_val
    
    return ((data - norm_min) / norm_range_width) * actual_range_width + min_val

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

# ==================== Output all files found ====================
print("\n" + "="*50)
print(f"Found {len(test_list)} test files:")
for i, file_path in enumerate(test_list):
    print(f"  Test [{i+1}/{len(test_list)}]: {file_path}")

print("\n" + "="*50)
print(f"Found {len(val_list)} validation files:")
for i, file_path in enumerate(val_list):
    print(f"  Val [{i+1}/{len(val_list)}]: {file_path}")
print("="*50 + "\n")

# ==================== Process first validation file ====================
if len(val_list) > 0:
    print("\nProcessing first validation file...")
    file_path = val_list[0]
    print(f"Processing validation file: {file_path}")
    
    # Get corresponding output array path
    output_array_path = os.path.join(
        test_dict["save_folder"],
        test_dict["eval_save_folder"],
        os.path.basename(file_path).replace(".nii.gz", "_array.npy")
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
        
        # Calculate all uncertainty metrics
        print("Calculating uncertainty metrics...")
        metrics = calculate_all_metrics(denormalized_array, voxel_wise=True)
        
        # Save each metric as a nifti file
        for metric_name, metric_data in metrics.items():
            # Create output filename
            output_filename = os.path.join(
                uncertainty_save_path,
                f"val_{os.path.basename(file_path).replace('.nii.gz', '')}_{metric_name}.nii.gz"
            )
            
            # Create and save nifti file
            metric_nifti = nib.Nifti1Image(metric_data, affine, header)
            nib.save(metric_nifti, output_filename)
            print(f"Saved {metric_name} to {output_filename}")
    else:
        print(f"Warning: No predictions found for {file_path}")

# ==================== Process first test file ====================
if len(test_list) > 0:
    print("\nProcessing first test file...")
    file_path = test_list[0]
    print(f"Processing test file: {file_path}")
    
    # Get corresponding output array path
    output_array_path = os.path.join(
        test_dict["save_folder"],
        test_dict["eval_save_folder"],
        os.path.basename(file_path).replace(".nii.gz", "_array.npy")
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
        
        # Calculate all uncertainty metrics
        print("Calculating uncertainty metrics...")
        metrics = calculate_all_metrics(denormalized_array, voxel_wise=True)
        
        # Save each metric as a nifti file
        for metric_name, metric_data in metrics.items():
            # Create output filename
            output_filename = os.path.join(
                uncertainty_save_path,
                f"test_{os.path.basename(file_path).replace('.nii.gz', '')}_{metric_name}.nii.gz"
            )
            
            # Create and save nifti file
            metric_nifti = nib.Nifti1Image(metric_data, affine, header)
            nib.save(metric_nifti, output_filename)
            print(f"Saved {metric_name} to {output_filename}")
    else:
        print(f"Warning: No predictions found for {file_path}")
