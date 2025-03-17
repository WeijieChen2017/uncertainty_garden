import os
import numpy as np
import nibabel as nib
import pandas as pd
from scipy import stats

# Define the model name
model_name = "Theseus_v2_181_200_rdp1"
print(f"Using model: {model_name}")

# Define the list of cases to process
to_do_cases = ["00219"]  # Add more case numbers as needed
print(f"Will process only files containing these case numbers: {to_do_cases}")

# Setup dictionaries
test_dict = {}
test_dict["project_name"] = model_name
test_dict["save_folder"] = "project_dir/"+test_dict["project_name"]+"/"
test_dict["eval_save_folder"] = "array"

# Define sample sizes to analyze
sample_sizes = [2, 4, 8, 16, 32, 64, 128]

def compute_correlation_for_samples(predictions, ground_truth, n_samples):
    """
    Compute correlation between mean predictions and ground truth for a given number of samples.
    
    Args:
        predictions: Array of shape (128, x, y, z) containing all predictions
        ground_truth: Array of shape (x, y, z) containing ground truth
        n_samples: Number of samples to use for computing mean prediction
        
    Returns:
        tuple: (pearson_r, spearman_r, kendall_tau)
    """
    # Number of groups we can make
    n_groups = 128 // n_samples
    
    # Initialize arrays to store correlations
    pearson_correlations = []
    spearman_correlations = []
    kendall_correlations = []
    
    # For each group
    for i in range(n_groups):
        # Randomly select n_samples indices
        indices = np.random.choice(128, n_samples, replace=False)
        
        # Compute mean prediction for these samples
        mean_pred = np.mean(predictions[indices], axis=0)
        
        # Flatten arrays for correlation computation
        pred_flat = mean_pred.flatten()
        truth_flat = ground_truth.flatten()
        
        # Compute correlations
        pearson_r = stats.pearsonr(pred_flat, truth_flat)[0]
        spearman_r = stats.spearmanr(pred_flat, truth_flat)[0]
        kendall_tau = stats.kendalltau(pred_flat, truth_flat)[0]
        
        pearson_correlations.append(pearson_r)
        spearman_correlations.append(spearman_r)
        kendall_correlations.append(kendall_tau)
    
    # Return mean correlations
    return (
        np.mean(pearson_correlations),
        np.mean(spearman_correlations),
        np.mean(kendall_correlations)
    )

# Load data division
data_div = np.load(os.path.join(test_dict["save_folder"], "data_division.npy"), allow_pickle=True)[()]

# Get test and validation lists
test_list = data_div['test_list_X']
test_list.sort()

val_list = []
if 'val_list_X' in data_div:
    val_list = data_div['val_list_X']
    val_list.sort()

# Filter lists to only include files with case numbers in to_do_cases
test_list = [f for f in test_list if any(case in f for case in to_do_cases)]
val_list = [f for f in val_list if any(case in f for case in to_do_cases)]

# Initialize results dictionary
results = {
    'n_samples': [],
    'pearson_r': [],
    'spearman_r': [],
    'kendall_tau': []
}

# Process validation files
if len(val_list) > 0:
    for file_path in val_list:
        print(f"\nProcessing validation file: {file_path}")
        
        # Get corresponding output array path
        output_array_path = os.path.join(
            test_dict["save_folder"],
            test_dict["eval_save_folder"],
            os.path.basename(file_path).replace(".nii.gz", "_array.npy")
        )
        
        if os.path.exists(output_array_path):
            print(f"Found saved predictions at: {output_array_path}")
            
            # Load predictions and ground truth
            predictions = np.load(output_array_path)
            ground_truth = nib.load("../"+file_path.replace("x", "y")).get_fdata()
            
            print(f"Loaded predictions shape: {predictions.shape}")
            print(f"Ground truth shape: {ground_truth.shape}")
            
            # For each sample size
            for n_samples in sample_sizes:
                print(f"\nComputing correlations for {n_samples} samples...")
                
                # Compute correlations
                pearson_r, spearman_r, kendall_tau = compute_correlation_for_samples(
                    predictions, ground_truth, n_samples
                )
                
                # Store results
                results['n_samples'].append(n_samples)
                results['pearson_r'].append(pearson_r)
                results['spearman_r'].append(spearman_r)
                results['kendall_tau'].append(kendall_tau)
                
                print(f"Results for {n_samples} samples:")
                print(f"  Pearson r: {pearson_r:.4f}")
                print(f"  Spearman r: {spearman_r:.4f}")
                print(f"  Kendall tau: {kendall_tau:.4f}")

# Process test files
if len(test_list) > 0:
    for file_path in test_list:
        print(f"\nProcessing test file: {file_path}")
        
        # Get corresponding output array path
        output_array_path = os.path.join(
            test_dict["save_folder"],
            test_dict["eval_save_folder"],
            os.path.basename(file_path).replace(".nii.gz", "_array.npy")
        )
        
        if os.path.exists(output_array_path):
            print(f"Found saved predictions at: {output_array_path}")
            
            # Load predictions and ground truth
            predictions = np.load(output_array_path)
            ground_truth = nib.load("../"+file_path.replace("x", "y")).get_fdata()
            
            print(f"Loaded predictions shape: {predictions.shape}")
            print(f"Ground truth shape: {ground_truth.shape}")
            
            # For each sample size
            for n_samples in sample_sizes:
                print(f"\nComputing correlations for {n_samples} samples...")
                
                # Compute correlations
                pearson_r, spearman_r, kendall_tau = compute_correlation_for_samples(
                    predictions, ground_truth, n_samples
                )
                
                # Store results
                results['n_samples'].append(n_samples)
                results['pearson_r'].append(pearson_r)
                results['spearman_r'].append(spearman_r)
                results['kendall_tau'].append(kendall_tau)
                
                print(f"Results for {n_samples} samples:")
                print(f"  Pearson r: {pearson_r:.4f}")
                print(f"  Spearman r: {spearman_r:.4f}")
                print(f"  Kendall tau: {kendall_tau:.4f}")

# Create DataFrame and save results
df = pd.DataFrame(results)
output_file = os.path.join(test_dict["save_folder"], "correlation_step.csv")
df.to_csv(output_file, index=False)
print(f"\nSaved correlation results to: {output_file}") 