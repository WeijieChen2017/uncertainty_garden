import os
import numpy as np
import nibabel as nib
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
from scipy.ndimage import binary_fill_holes
import csv

# Paths - using absolute paths as specified in the user query
ground_truth_path = "/shares/mimrtl/Users/Winston/files_to_dgx/SUREMI/uncertainty_garden/project_dir/Theseus_v2_181_200_rdp1/ground_truth/00219_yte.nii.gz"
prediction_path = "/shares/mimrtl/Users/Winston/files_to_dgx/SUREMI/uncertainty_garden/project_dir/Theseus_v2_181_200_rdp1/array/00219_xte_array.npy"
uncertainty_path = "/shares/mimrtl/Users/Winston/files_to_dgx/SUREMI/uncertainty_garden/project_dir/Theseus_v2_181_200_rdp1/uncertainty"

# Ensure results directory exists
results_path = "/shares/mimrtl/Users/Winston/files_to_dgx/SUREMI/uncertainty_garden/project_dir/Theseus_v2_181_200_rdp1/correlation_results"
os.makedirs(results_path, exist_ok=True)

# Normalization parameters
min_val = -1024
max_val = 2976
norm_range = [-1, 1]

# Mask threshold
mask_threshold = -500  # Threshold for creating the mask

# Function to denormalize data from [-1, 1] to [min_val, max_val]
def denormalize(data, min_val=-1024, max_val=2976, norm_range=[-1, 1]):
    """Denormalize data from norm_range to [min_val, max_val]"""
    norm_min, norm_max = norm_range
    norm_range_width = norm_max - norm_min
    actual_range_width = max_val - min_val
    
    return ((data - norm_min) / norm_range_width) * actual_range_width + min_val

# Function to normalize a metric to [0,1] range for better comparison
def normalize_metric(data):
    """Normalize data to [0,1] range"""
    min_data = np.min(data)
    max_data = np.max(data)
    if max_data == min_data:
        return np.zeros_like(data)
    return (data - min_data) / (max_data - min_data)

print("Loading ground truth...")
try:
    # Load ground truth
    ground_truth_nifti = nib.load(ground_truth_path)
    ground_truth_raw = ground_truth_nifti.get_fdata()
    print(f"Ground truth loaded. Shape: {ground_truth_raw.shape}")
    
    # Check if ground truth needs to be denormalized
    # We'll assume ground truth is also normalized in the same range [-1, 1]
    ground_truth = denormalize(ground_truth_raw, min_val, max_val, norm_range)
    print(f"Denormalized ground truth range: [{np.min(ground_truth)}, {np.max(ground_truth)}]")
    
    # Get header and affine for saving files later
    affine = ground_truth_nifti.affine
    header = ground_truth_nifti.header
    
    # Save denormalized ground truth
    denorm_gt_nifti = nib.Nifti1Image(ground_truth, affine, header)
    denorm_gt_path = os.path.join(results_path, "00219_denormalized_ground_truth.nii.gz")
    nib.save(denorm_gt_nifti, denorm_gt_path)
    print(f"Saved denormalized ground truth to {denorm_gt_path}")
    
except FileNotFoundError:
    print(f"Error: Ground truth file not found at {ground_truth_path}")
    print("Please update the path to match your system configuration.")
    exit(1)

print("Loading predictions...")
try:
    # Load predictions
    predictions = np.load(prediction_path)
    print(f"Predictions loaded. Shape: {predictions.shape}")
    
    # Denormalize predictions
    denormalized_predictions = denormalize(predictions, min_val, max_val, norm_range)
    print(f"Denormalized predictions range: [{np.min(denormalized_predictions)}, {np.max(denormalized_predictions)}]")
    
    # Compute median prediction
    median_prediction = np.median(denormalized_predictions, axis=0)
    print(f"Median prediction shape: {median_prediction.shape}")
    
    # Save median prediction
    median_nifti = nib.Nifti1Image(median_prediction, affine, header)
    median_output_path = os.path.join(results_path, "00219_median_prediction.nii.gz")
    nib.save(median_nifti, median_output_path)
    print(f"Saved median prediction to {median_output_path}")
    
except FileNotFoundError:
    print(f"Error: Prediction file not found at {prediction_path}")
    print("Please update the path to match your system configuration.")
    exit(1)

# Create mask using threshold and binary fill holes
print(f"Creating mask using threshold {mask_threshold} and binary fill holes...")
mask = median_prediction > mask_threshold
mask = binary_fill_holes(mask).astype(np.float32)
print(f"Mask created. Number of voxels in mask: {np.sum(mask)}")
print(f"Mask percentage: {np.sum(mask) / mask.size * 100:.2f}%")

# Save mask
mask_nifti = nib.Nifti1Image(mask, affine, header)
mask_output_path = os.path.join(results_path, "00219_mask.nii.gz")
nib.save(mask_nifti, mask_output_path)
print(f"Saved mask to {mask_output_path}")

# Compute absolute error within mask
print("Computing absolute error...")
abs_error = np.abs(ground_truth - median_prediction)
abs_error_masked = abs_error * mask  # Apply mask to error

# Calculate stats for masked error
masked_indices = mask > 0
masked_error_values = abs_error[masked_indices]
print(f"Absolute error range within mask: [{np.min(masked_error_values)}, {np.max(masked_error_values)}]")
print(f"Mean absolute error within mask: {np.mean(masked_error_values)}")
print(f"Median absolute error within mask: {np.median(masked_error_values)}")

# Save absolute error (masked)
abs_error_nifti = nib.Nifti1Image(abs_error_masked, affine, header)
abs_error_output_path = os.path.join(results_path, "00219_absolute_error_masked.nii.gz")
nib.save(abs_error_nifti, abs_error_output_path)
print(f"Saved masked absolute error to {abs_error_output_path}")

# List of uncertainty metrics to evaluate
uncertainty_metrics = [
    "variance", "std", "cv", "entropy", "iqr", 
    "range", "ci_width", "pred_var", "mutual_info"
]

# Dictionary to store correlation results
pearson_correlations = {}
spearman_correlations = {}

# Dictionary to store additional statistics for each metric
metric_statistics = {}

print("\nComputing correlations between absolute error and uncertainty metrics (within mask)...")
for metric in uncertainty_metrics:
    metric_file = f"test_00219_xte_{metric}.nii.gz"
    metric_path = os.path.join(uncertainty_path, metric_file)
    
    try:
        # Load uncertainty metric
        metric_nifti = nib.load(metric_path)
        metric_data = metric_nifti.get_fdata()
        print(f"\nLoaded {metric}. Shape: {metric_data.shape}")
        
        # Apply mask to metric data
        metric_data_masked = metric_data * mask
        
        # Store statistics for the metric
        metric_stats = {
            'min': np.min(metric_data[masked_indices]),
            'max': np.max(metric_data[masked_indices]),
            'mean': np.mean(metric_data[masked_indices]),
            'median': np.median(metric_data[masked_indices]),
            'std': np.std(metric_data[masked_indices])
        }
        metric_statistics[metric] = metric_stats
        
        # Normalize metric for visualization
        normalized_metric = normalize_metric(metric_data)
        normalized_metric_masked = normalized_metric * mask
        
        # Save normalized metric for visualization (both masked and unmasked)
        norm_metric_nifti = nib.Nifti1Image(normalized_metric, affine, header)
        norm_metric_path = os.path.join(results_path, f"00219_normalized_{metric}.nii.gz")
        nib.save(norm_metric_nifti, norm_metric_path)
        
        norm_metric_masked_nifti = nib.Nifti1Image(normalized_metric_masked, affine, header)
        norm_metric_masked_path = os.path.join(results_path, f"00219_normalized_{metric}_masked.nii.gz")
        nib.save(norm_metric_masked_nifti, norm_metric_masked_path)
        
        # Flatten arrays for correlation (only within mask)
        flat_error = masked_error_values
        flat_metric = metric_data[masked_indices]
        
        # Calculate correlations
        pearson_corr, pearson_p = pearsonr(flat_error, flat_metric)
        spearman_corr, spearman_p = spearmanr(flat_error, flat_metric)
        
        pearson_correlations[metric] = (pearson_corr, pearson_p)
        spearman_correlations[metric] = (spearman_corr, spearman_p)
        
        print(f"{metric} - Pearson correlation: {pearson_corr:.4f} (p={pearson_p:.4e})")
        print(f"{metric} - Spearman correlation: {spearman_corr:.4f} (p={spearman_p:.4e})")
        
    except FileNotFoundError:
        print(f"Warning: {metric} file not found at {metric_path}")

# Print summary of correlations
print("\n" + "="*50)
print("CORRELATION SUMMARY (ordered by Spearman correlation)")
print("="*50)

# Sort metrics by Spearman correlation (absolute value, descending)
sorted_metrics = sorted(uncertainty_metrics, 
                       key=lambda x: abs(spearman_correlations.get(x, (0, 1))[0]) if x in spearman_correlations else 0,
                       reverse=True)

# Print sorted results
print(f"{'Metric':<15} {'Pearson':<12} {'p-value':<12} {'Spearman':<12} {'p-value':<12}")
print("-"*65)

for metric in sorted_metrics:
    if metric in pearson_correlations and metric in spearman_correlations:
        pearson_corr, pearson_p = pearson_correlations[metric]
        spearman_corr, spearman_p = spearman_correlations[metric]
        print(f"{metric:<15} {pearson_corr:>10.4f}   {pearson_p:>10.4e}   {spearman_corr:>10.4f}   {spearman_p:>10.4e}")

# Create bar plot of correlation values
plt.figure(figsize=(12, 8))
metrics = [m for m in sorted_metrics if m in spearman_correlations]
pearson_values = [pearson_correlations[m][0] for m in metrics]
spearman_values = [spearman_correlations[m][0] for m in metrics]

x = np.arange(len(metrics))
width = 0.35

plt.bar(x - width/2, pearson_values, width, label='Pearson')
plt.bar(x + width/2, spearman_values, width, label='Spearman')

plt.xlabel('Uncertainty Metric')
plt.ylabel('Correlation with Absolute Error')
plt.title('Correlation between Absolute Error and Uncertainty Metrics (within mask)')
plt.xticks(x, metrics, rotation=45)
plt.legend()
plt.tight_layout()
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(results_path, 'correlation_plot.png'))
print(f"Saved correlation plot to {os.path.join(results_path, 'correlation_plot.png')}")

# Save the correlation results to a text file
with open(os.path.join(results_path, 'correlation_results.txt'), 'w') as f:
    f.write("CORRELATION BETWEEN ABSOLUTE ERROR AND UNCERTAINTY METRICS (WITHIN MASK)\n")
    f.write("="*70 + "\n\n")
    
    f.write(f"Mask threshold: {mask_threshold}\n")
    f.write(f"Total voxels in mask: {np.sum(mask)} ({np.sum(mask) / mask.size * 100:.2f}% of volume)\n\n")
    
    f.write(f"{'Metric':<15} {'Pearson':<12} {'p-value':<12} {'Spearman':<12} {'p-value':<12}\n")
    f.write("-"*65 + "\n")
    
    for metric in sorted_metrics:
        if metric in pearson_correlations and metric in spearman_correlations:
            pearson_corr, pearson_p = pearson_correlations[metric]
            spearman_corr, spearman_p = spearman_correlations[metric]
            f.write(f"{metric:<15} {pearson_corr:>10.4f}   {pearson_p:>10.4e}   {spearman_corr:>10.4f}   {spearman_p:>10.4e}\n")
    
    f.write("\n" + "="*70 + "\n")
    f.write("Analysis completed on: " + np.datetime64('now').astype(str) + "\n")

# Save the correlation results to a CSV file
csv_file_path = os.path.join(results_path, 'correlation_results.csv')
with open(csv_file_path, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    
    # Write header
    csvwriter.writerow(['Metric', 'Pearson', 'Pearson_p_value', 'Spearman', 'Spearman_p_value', 
                       'Min', 'Max', 'Mean', 'Median', 'Std'])
    
    # Write data for each metric
    for metric in sorted_metrics:
        if metric in pearson_correlations and metric in spearman_correlations:
            pearson_corr, pearson_p = pearson_correlations[metric]
            spearman_corr, spearman_p = spearman_correlations[metric]
            
            # Add statistics if available
            stats = metric_statistics.get(metric, {})
            min_val = stats.get('min', float('nan'))
            max_val = stats.get('max', float('nan'))
            mean_val = stats.get('mean', float('nan'))
            median_val = stats.get('median', float('nan'))
            std_val = stats.get('std', float('nan'))
            
            csvwriter.writerow([
                metric, 
                pearson_corr, pearson_p, 
                spearman_corr, spearman_p,
                min_val, max_val, mean_val, median_val, std_val
            ])

print(f"Saved correlation results to CSV: {csv_file_path}")

# Create a second CSV with simplified format for easier analysis
simple_csv_path = os.path.join(results_path, 'correlation_summary.csv')
with open(simple_csv_path, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    
    # Write header with analysis details
    csvwriter.writerow(['Analysis Details'])
    csvwriter.writerow(['Case', '00219'])
    csvwriter.writerow(['Mask threshold', mask_threshold])
    csvwriter.writerow(['Mask voxel count', int(np.sum(mask))])
    csvwriter.writerow(['Mask percentage', f"{np.sum(mask) / mask.size * 100:.2f}%"])
    csvwriter.writerow(['Mean absolute error', f"{np.mean(masked_error_values):.4f}"])
    csvwriter.writerow([])  # Empty row as separator
    
    # Correlation results
    csvwriter.writerow(['Metric', 'Pearson', 'Spearman'])
    
    for metric in sorted_metrics:
        if metric in pearson_correlations and metric in spearman_correlations:
            pearson_corr = pearson_correlations[metric][0]
            spearman_corr = spearman_correlations[metric][0]
            csvwriter.writerow([metric, f"{pearson_corr:.4f}", f"{spearman_corr:.4f}"])

print(f"Saved simplified correlation summary to CSV: {simple_csv_path}")

print("\nAnalysis complete! Results saved to:", results_path) 