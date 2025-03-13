import os
import sys
import glob
import time
import numpy as np
import nibabel as nib
import torch

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

# Load train_dict to get necessary parameters
train_dict = np.load(test_dict["save_folder"]+"dict.npy", allow_pickle=True)[()]

# ==================== data division ====================

# Load data division
data_div = np.load(os.path.join(test_dict["save_folder"], "data_division.npy"), allow_pickle=True)[()]

# Get test data list
test_list = data_div['test_list_X']
# Remove "../" prefix from paths and ensure correct path format
test_list = [path.replace("../", "") for path in test_list]
test_list = ["./"+path if not path.startswith("./") else path for path in test_list]
test_list.sort()  # Sort the test list

# Get validation data list if available
val_list = []
if 'val_list_X' in data_div:
    val_list = [path.replace("../", "") for path in data_div['val_list_X']]
    val_list = ["./"+path if not path.startswith("./") else path for path in val_list]
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

# ==================== Process validation files ====================
print("\nProcessing validation files...")
for file_idx, file_path in enumerate(val_list):
    print(f"Processing validation file [{file_idx+1}/{len(val_list)}]: {file_path}")
    
    # Get corresponding output array path
    output_array_path = os.path.join(
        test_dict["save_folder"],
        test_dict["eval_save_folder"],
        os.path.basename(file_path).replace(".nii.gz", "_array.npy")
    )
    
    # Check if the output array exists
    if os.path.exists(output_array_path):
        print(f"Found saved predictions at: {output_array_path}")
        
        # Here we would load and process the predictions
        # output_array = np.load(output_array_path)
        
        # TODO: Perform uncertainty quantification on the loaded predictions
        pass
    else:
        print(f"Warning: No predictions found for {file_path}")
        # TODO: Handle missing predictions
        pass

# ==================== Process test files ====================
print("\nProcessing test files...")
for file_idx, file_path in enumerate(test_list):
    print(f"Processing test file [{file_idx+1}/{len(test_list)}]: {file_path}")
    
    # Get corresponding output array path
    output_array_path = os.path.join(
        test_dict["save_folder"],
        test_dict["eval_save_folder"],
        os.path.basename(file_path).replace(".nii.gz", "_array.npy")
    )
    
    # Check if the output array exists
    if os.path.exists(output_array_path):
        print(f"Found saved predictions at: {output_array_path}")
        
        # Here we would load and process the predictions
        # output_array = np.load(output_array_path)
        
        # TODO: Perform uncertainty quantification on the loaded predictions
        pass
    else:
        print(f"Warning: No predictions found for {file_path}")
        # TODO: Handle missing predictions
        pass
