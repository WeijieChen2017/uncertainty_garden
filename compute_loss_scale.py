import os
import numpy as np
import nibabel as nib
import csv

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
test_dict["loss_scale_save_folder"] = "loss_scale"

# Create loss scale save folder if it doesn't exist
loss_scale_save_path = os.path.join(test_dict["save_folder"], test_dict["loss_scale_save_folder"])
os.makedirs(loss_scale_save_path, exist_ok=True)

def compute_mae_loss(pred1, pred2):
    """Compute MAE loss between two predictions"""
    return np.mean(np.abs(pred1 - pred2))

def create_loss_table(predictions, ground_truth):
    """
    Create a loss table of size (n_pred+1) × (n_pred+1)
    Each element (i,j) is the MAE loss between prediction i and j
    Last row/column contains MAE between predictions and ground truth
    """
    n_pred = len(predictions)
    table_size = n_pred + 1
    loss_table = np.zeros((table_size, table_size))
    
    # Fill the table with MAE losses between predictions
    for i in range(n_pred):
        for j in range(n_pred):
            loss_table[i, j] = compute_mae_loss(predictions[i], predictions[j])
    
    # Fill the last row and column with ground truth losses
    for i in range(n_pred):
        loss_table[i, -1] = compute_mae_loss(predictions[i], ground_truth)
        loss_table[-1, i] = loss_table[i, -1]  # Symmetric
    
    # Set the last element (ground truth vs ground truth) to 0
    loss_table[-1, -1] = 0
    
    return loss_table

def save_loss_table_to_csv(loss_table, case_key, save_path):
    """Save the loss table to a CSV file"""
    n_pred = loss_table.shape[0] - 1  # Subtract 1 for ground truth
    
    # Create headers
    headers = [f"Pred_{i+1}" for i in range(n_pred)]
    headers.append("Ground_Truth")
    
    # Save to CSV
    csv_path = os.path.join(save_path, f"{case_key}_mae_loss_table.csv")
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([''] + headers)  # Write headers with empty first cell
        
        # Write each row
        for i in range(n_pred + 1):
            row = [headers[i]] + [f"{loss_table[i,j]:.6f}" for j in range(n_pred + 1)]
            writer.writerow(row)
    
    print(f"Saved MAE loss table to {csv_path}")

# ==================== data division ====================

# Load data division - keep this path independent
data_div = np.load(os.path.join(test_dict["save_folder"], "data_division.npy"), allow_pickle=True)[()]

# Get test data list - original paths from data_div
test_list = data_div['test_list_X']
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
    
    # Extract case key from file path
    case_key = os.path.basename(file_path).split('_')[0]
    
    # Get ground truth path
    ground_truth_path = os.path.join(
        test_dict["save_folder"],
        "ground_truth",
        f"{case_key}_yte.nii.gz"
    )
    
    # Get predictions path
    predictions_path = os.path.join(
        test_dict["save_folder"],
        test_dict["eval_save_folder"],
        f"{case_key}_xte_array.npy"
    )
    
    # Check if both files exist
    if os.path.exists(ground_truth_path) and os.path.exists(predictions_path):
        print(f"Loading ground truth from: {ground_truth_path}")
        print(f"Loading predictions from: {predictions_path}")
        
        # Load ground truth
        ground_truth_nifti = nib.load(ground_truth_path)
        ground_truth = ground_truth_nifti.get_fdata()
        print(f"Ground truth shape: {ground_truth.shape}")
        
        # Load predictions
        predictions = np.load(predictions_path)
        print(f"Predictions shape: {predictions.shape}")
        
        # Create loss table
        loss_table = create_loss_table(predictions, ground_truth)
        
        # Save to CSV
        save_loss_table_to_csv(loss_table, case_key, loss_scale_save_path)
    else:
        print(f"Warning: Missing files for case {case_key}")
        if not os.path.exists(ground_truth_path):
            print(f"Ground truth not found at: {ground_truth_path}")
        if not os.path.exists(predictions_path):
            print(f"Predictions not found at: {predictions_path}")

# ==================== Process first test file ====================
if len(test_list) > 0:
    print("\nProcessing first test file...")
    file_path = "../"+test_list[0]
    print(f"Processing test file: {file_path}")
    
    # Extract case key from file path
    case_key = os.path.basename(file_path).split('_')[0]
    
    # Get ground truth path
    ground_truth_path = os.path.join(
        test_dict["save_folder"],
        "ground_truth",
        f"{case_key}_yte.nii.gz"
    )
    
    # Get predictions path
    predictions_path = os.path.join(
        test_dict["save_folder"],
        test_dict["eval_save_folder"],
        f"{case_key}_xte_array.npy"
    )
    
    # Check if both files exist
    if os.path.exists(ground_truth_path) and os.path.exists(predictions_path):
        print(f"Loading ground truth from: {ground_truth_path}")
        print(f"Loading predictions from: {predictions_path}")
        
        # Load ground truth
        ground_truth_nifti = nib.load(ground_truth_path)
        ground_truth = ground_truth_nifti.get_fdata()
        print(f"Ground truth shape: {ground_truth.shape}")
        
        # Load predictions
        predictions = np.load(predictions_path)
        print(f"Predictions shape: {predictions.shape}")
        
        # Create loss table
        loss_table = create_loss_table(predictions, ground_truth)
        
        # Save to CSV
        save_loss_table_to_csv(loss_table, case_key, loss_scale_save_path)
    else:
        print(f"Warning: Missing files for case {case_key}")
        if not os.path.exists(ground_truth_path):
            print(f"Ground truth not found at: {ground_truth_path}")
        if not os.path.exists(predictions_path):
            print(f"Predictions not found at: {predictions_path}") 