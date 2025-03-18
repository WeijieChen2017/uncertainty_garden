import os
import numpy as np
import nibabel as nib
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
from scipy.ndimage import binary_fill_holes, binary_dilation, binary_erosion
import csv

# Flag to control whether to overwrite existing files
OVERWRITE = True  # Set to True to force recomputation and overwrite existing files

case_key = "00219"

# Paths - using absolute paths as specified in the user query
ground_truth_path = f"/shares/mimrtl/Users/Winston/files_to_dgx/SUREMI/uncertainty_garden/project_dir/Theseus_v2_181_200_rdp1/ground_truth/{case_key}_yte.nii.gz"
prediction_path = f"/shares/mimrtl/Users/Winston/files_to_dgx/SUREMI/uncertainty_garden/project_dir/Theseus_v2_181_200_rdp1/array/{case_key}_xte_array.npy"
uncertainty_path = "/shares/mimrtl/Users/Winston/files_to_dgx/SUREMI/uncertainty_garden/project_dir/Theseus_v2_181_200_rdp1/uncertainty"

# Ensure results directory exists
results_path = "/shares/mimrtl/Users/Winston/files_to_dgx/SUREMI/uncertainty_garden/project_dir/Theseus_v2_181_200_rdp1/correlation_results"
os.makedirs(results_path, exist_ok=True)

# Create a directory for denormalized predictions
denorm_pred_path = os.path.join(results_path, "denormalized_predictions")
os.makedirs(denorm_pred_path, exist_ok=True)

# Normalization parameters
min_val = -1024
max_val = 2976
norm_range = [0, 1]

# Mask threshold and processing parameters
mask_threshold = -500  # Threshold for creating the mask
dilation_iterations = 3  # Number of iterations for mask dilation
erosion_iterations = 3   # Number of iterations for mask erosion

# Function to denormalize data from [0, 1] to [min_val, max_val]
def denormalize(data, min_val=-1024, max_val=2976, norm_range=[0, 1]):
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
# Define the denormalized ground truth path
denorm_gt_path = os.path.join(results_path, f"{case_key}_denormalized_ground_truth.nii.gz")

# Check if the denormalized ground truth file already exists
if os.path.exists(denorm_gt_path) and not OVERWRITE:
    print(f"Found existing denormalized ground truth at {denorm_gt_path}")
    ground_truth_nifti = nib.load(denorm_gt_path)
    ground_truth = ground_truth_nifti.get_fdata()
    affine = ground_truth_nifti.affine
    header = ground_truth_nifti.header
    print(f"Loaded denormalized ground truth. Shape: {ground_truth.shape}")
    print(f"Denormalized ground truth range: [{np.min(ground_truth)}, {np.max(ground_truth)}]")
else:
    try:
        # Load ground truth
        ground_truth_nifti = nib.load(ground_truth_path)
        ground_truth_raw = ground_truth_nifti.get_fdata()
        print(f"Ground truth loaded. Shape: {ground_truth_raw.shape}")
        
        # Check if ground truth needs to be denormalized
        # We'll assume ground truth is also normalized in the range [0, 1]
        ground_truth = denormalize(ground_truth_raw, min_val, max_val, norm_range)
        print(f"Denormalized ground truth range: [{np.min(ground_truth)}, {np.max(ground_truth)}]")
        
        # Get header and affine for saving files later
        affine = ground_truth_nifti.affine
        header = ground_truth_nifti.header
        
        # Save denormalized ground truth
        denorm_gt_nifti = nib.Nifti1Image(ground_truth, affine, header)
        nib.save(denorm_gt_nifti, denorm_gt_path)
        print(f"Saved denormalized ground truth to {denorm_gt_path}")
        
    except FileNotFoundError:
        print(f"Error: Ground truth file not found at {ground_truth_path}")
        print("Please update the path to match your system configuration.")
        exit(1)

print("Loading predictions...")
# Define paths for denormalized predictions array and median prediction
denorm_pred_array_path = os.path.join(results_path, f"{case_key}_denormalized_predictions_array.npy")
median_output_path = os.path.join(results_path, f"{case_key}_median_prediction.nii.gz")
min_path = os.path.join(results_path, f"{case_key}_min_prediction.nii.gz")
max_path = os.path.join(results_path, f"{case_key}_max_prediction.nii.gz")
mean_path = os.path.join(results_path, f"{case_key}_mean_prediction.nii.gz")

# Check if denormalized predictions already exist
if os.path.exists(denorm_pred_array_path) and os.path.exists(median_output_path) and \
   os.path.exists(min_path) and os.path.exists(max_path) and os.path.exists(mean_path) and not OVERWRITE:
    print(f"Found existing denormalized predictions at {denorm_pred_array_path}")
    denormalized_predictions = np.load(denorm_pred_array_path)
    print(f"Loaded denormalized predictions. Shape: {denormalized_predictions.shape}")
    
    # Load statistics
    median_prediction = nib.load(median_output_path).get_fdata()
    min_prediction = nib.load(min_path).get_fdata()
    max_prediction = nib.load(max_path).get_fdata()
    mean_prediction = nib.load(mean_path).get_fdata()
    
    print(f"Loaded median, min, max, and mean predictions")
    print(f"Median prediction shape: {median_prediction.shape}")
    print(f"Denormalized predictions range: [{np.min(denormalized_predictions)}, {np.max(denormalized_predictions)}]")
else:
    try:
        # Load predictions
        predictions = np.load(prediction_path)
        print(f"Predictions loaded. Shape: {predictions.shape}")
        
        # Denormalize predictions
        denormalized_predictions = denormalize(predictions, min_val, max_val, norm_range)
        print(f"Denormalized predictions range: [{np.min(denormalized_predictions)}, {np.max(denormalized_predictions)}]")
        
        # Save denormalized predictions array
        np.save(denorm_pred_array_path, denormalized_predictions)
        print(f"Saved denormalized predictions array to {denorm_pred_array_path}")
        
        # Save individual denormalized prediction samples as nifti files (first 5 samples)
        num_samples_to_save = min(5, denormalized_predictions.shape[0])
        for i in range(num_samples_to_save):
            sample_pred = denormalized_predictions[i]
            sample_nifti = nib.Nifti1Image(sample_pred, affine, header)
            sample_path = os.path.join(denorm_pred_path, f"{case_key}_denorm_pred_sample_{i+1}.nii.gz")
            nib.save(sample_nifti, sample_path)
            print(f"Saved denormalized prediction sample {i+1} to {sample_path}")
        
        # Compute median prediction
        median_prediction = np.median(denormalized_predictions, axis=0)
        print(f"Median prediction shape: {median_prediction.shape}")
        
        # Save median prediction
        median_nifti = nib.Nifti1Image(median_prediction, affine, header)
        nib.save(median_nifti, median_output_path)
        print(f"Saved median prediction to {median_output_path}")
        
        # Save min, max, mean of the predictions
        min_prediction = np.min(denormalized_predictions, axis=0)
        max_prediction = np.max(denormalized_predictions, axis=0)
        mean_prediction = np.mean(denormalized_predictions, axis=0)
        
        min_nifti = nib.Nifti1Image(min_prediction, affine, header)
        max_nifti = nib.Nifti1Image(max_prediction, affine, header)
        mean_nifti = nib.Nifti1Image(mean_prediction, affine, header)
        
        nib.save(min_nifti, min_path)
        nib.save(max_nifti, max_path)
        nib.save(mean_nifti, mean_path)
        
        print(f"Saved min/max/mean predictions to {results_path}")
        
    except FileNotFoundError:
        print(f"Error: Prediction file not found at {prediction_path}")
        print("Please update the path to match your system configuration.")
        exit(1)

# Check if mask already exists
mask_output_path = os.path.join(results_path, f"{case_key}_mask.nii.gz")
if os.path.exists(mask_output_path) and not OVERWRITE:
    print(f"\nFound existing mask at {mask_output_path}")
    mask_nifti = nib.load(mask_output_path)
    mask = mask_nifti.get_fdata().astype(np.float32)
    print(f"Loaded mask. Number of voxels in mask: {np.sum(mask)}")
    print(f"Mask percentage: {np.sum(mask) / mask.size * 100:.2f}%")
else:
    # Create mask using threshold, dilation, erosion, and z-slice binary fill holes
    print(f"Creating mask using threshold {mask_threshold} with dilation and erosion...")
    
    # Initial threshold using ground truth instead of median prediction
    initial_mask = ground_truth > mask_threshold
    print(f"Initial thresholding on ground truth - Number of voxels: {np.sum(initial_mask)}")
    
    # Dilate the mask
    dilated_mask = binary_dilation(initial_mask, iterations=dilation_iterations)
    print(f"After dilation ({dilation_iterations} iterations) - Number of voxels: {np.sum(dilated_mask)}")
    
    # Erode the mask
    eroded_mask = binary_erosion(dilated_mask, iterations=erosion_iterations)
    print(f"After erosion ({erosion_iterations} iterations) - Number of voxels: {np.sum(eroded_mask)}")
    
    # Apply binary fill holes slice by slice along z-axis
    mask = np.zeros_like(eroded_mask, dtype=np.float32)
    for z in range(eroded_mask.shape[2]):
        mask[:, :, z] = binary_fill_holes(eroded_mask[:, :, z])
    
    print(f"After z-slice binary fill holes - Number of voxels: {np.sum(mask)}")
    print(f"Mask created. Percentage: {np.sum(mask) / mask.size * 100:.2f}%")

    # Save mask
    mask_nifti = nib.Nifti1Image(mask.astype(np.float32), affine, header)
    nib.save(mask_nifti, mask_output_path)
    print(f"Saved mask to {mask_output_path}")

# Check if absolute error already exists
abs_error_output_path = os.path.join(results_path, f"{case_key}_absolute_error_masked.nii.gz")
abs_error_unmasked_path = os.path.join(results_path, f"{case_key}_absolute_error_unmasked.nii.gz")

if os.path.exists(abs_error_output_path) and os.path.exists(abs_error_unmasked_path) and not OVERWRITE:
    print(f"\nFound existing absolute error at {abs_error_output_path}")
    abs_error_masked_nifti = nib.load(abs_error_output_path)
    abs_error_masked = abs_error_masked_nifti.get_fdata()
    
    abs_error_unmasked_nifti = nib.load(abs_error_unmasked_path)
    abs_error = abs_error_unmasked_nifti.get_fdata()
    
    # Calculate stats for masked error
    masked_indices = mask > 0
    masked_error_values = abs_error[masked_indices]
    print(f"Loaded absolute error. Range within mask: [{np.min(masked_error_values)}, {np.max(masked_error_values)}]")
    print(f"Mean absolute error within mask: {np.mean(masked_error_values)}")
    print(f"Median absolute error within mask: {np.median(masked_error_values)}")
else:
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
    nib.save(abs_error_nifti, abs_error_output_path)
    print(f"Saved masked absolute error to {abs_error_output_path}")

    # Also save the unmasked absolute error for reference
    abs_error_unmasked_nifti = nib.Nifti1Image(abs_error, affine, header)
    nib.save(abs_error_unmasked_nifti, abs_error_unmasked_path)
    print(f"Saved unmasked absolute error to {abs_error_unmasked_path}")

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
    metric_file = f"test_{case_key}_xte_{metric}.nii.gz"
    metric_path = os.path.join(uncertainty_path, metric_file)
    
    # Define paths for normalized metrics
    norm_metric_path = os.path.join(results_path, f"{case_key}_normalized_{metric}.nii.gz")
    norm_metric_masked_path = os.path.join(results_path, f"{case_key}_normalized_{metric}_masked.nii.gz")
    
    # Check if normalized metrics already exist
    if os.path.exists(norm_metric_path) and os.path.exists(norm_metric_masked_path) and not OVERWRITE:
        print(f"\nFound existing normalized {metric} at {norm_metric_path}")
        metric_data = nib.load(metric_path).get_fdata()
        normalized_metric = nib.load(norm_metric_path).get_fdata()
        normalized_metric_masked = nib.load(norm_metric_masked_path).get_fdata()
        
        # Extract metrics stats
        masked_indices = mask > 0
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
    
    else:
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
            nib.save(norm_metric_nifti, norm_metric_path)
            
            norm_metric_masked_nifti = nib.Nifti1Image(normalized_metric_masked, affine, header)
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
plt.savefig(os.path.join(results_path, f'{case_key}_correlation_plot.png'))
print(f"Saved correlation plot to {os.path.join(results_path, f'{case_key}_correlation_plot.png')}")

# Generate a second plot with absolute correlation values
plt.figure(figsize=(12, 8))
abs_pearson_values = [abs(v) for v in pearson_values]
abs_spearman_values = [abs(v) for v in spearman_values]

# Sort metrics by absolute Spearman correlation
sorted_indices = np.argsort(abs_spearman_values)[::-1]
sorted_metrics_abs = [metrics[i] for i in sorted_indices]
sorted_abs_pearson = [abs_pearson_values[i] for i in sorted_indices]
sorted_abs_spearman = [abs_spearman_values[i] for i in sorted_indices]

x_abs = np.arange(len(sorted_metrics_abs))
plt.bar(x_abs - width/2, sorted_abs_pearson, width, label='|Pearson|')
plt.bar(x_abs + width/2, sorted_abs_spearman, width, label='|Spearman|')

plt.xlabel('Uncertainty Metric')
plt.ylabel('Absolute Correlation with Error')
plt.title('Absolute Correlation between Error and Uncertainty Metrics (within mask)')
plt.xticks(x_abs, sorted_metrics_abs, rotation=45)
plt.legend()
plt.tight_layout()
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(results_path, f'{case_key}_abs_correlation_plot.png'))
print(f"Saved absolute correlation plot to {os.path.join(results_path, f'{case_key}_abs_correlation_plot.png')}")

# Save the correlation results to a text file
with open(os.path.join(results_path, f'{case_key}_correlation_results.txt'), 'w') as f:
    f.write("CORRELATION BETWEEN ABSOLUTE ERROR AND UNCERTAINTY METRICS (WITHIN MASK)\n")
    f.write("="*70 + "\n\n")
    
    f.write(f"Normalization Range: [{norm_range[0]}, {norm_range[1]}]\n")
    f.write(f"HU Range: [{min_val}, {max_val}]\n")
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
csv_file_path = os.path.join(results_path, f'{case_key}_correlation_results.csv')
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
            min_val_metric = stats.get('min', float('nan'))
            max_val_metric = stats.get('max', float('nan'))
            mean_val = stats.get('mean', float('nan'))
            median_val = stats.get('median', float('nan'))
            std_val = stats.get('std', float('nan'))
            
            csvwriter.writerow([
                metric, 
                pearson_corr, pearson_p, 
                spearman_corr, spearman_p,
                min_val_metric, max_val_metric, mean_val, median_val, std_val
            ])

print(f"Saved correlation results to CSV: {csv_file_path}")

# Create a second CSV with simplified format for easier analysis
simple_csv_path = os.path.join(results_path, f'{case_key}_correlation_summary.csv')
with open(simple_csv_path, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    
    # Write header with analysis details
    csvwriter.writerow(['Analysis Details'])
    csvwriter.writerow(['Case', f'{case_key}'])
    csvwriter.writerow(['Normalization Range', f"[{norm_range[0]}, {norm_range[1]}]"])
    csvwriter.writerow(['HU Range', f"[{min_val}, {max_val}]"])
    csvwriter.writerow(['Mask threshold', mask_threshold])
    csvwriter.writerow(['Mask voxel count', int(np.sum(mask))])
    csvwriter.writerow(['Mask percentage', f"{np.sum(mask) / mask.size * 100:.2f}%"])
    csvwriter.writerow(['Mean absolute error', f"{np.mean(masked_error_values):.4f}"])
    csvwriter.writerow([])  # Empty row as separator
    
    # Correlation results
    csvwriter.writerow(['Metric', 'Pearson', 'Spearman', 'Abs_Pearson', 'Abs_Spearman'])
    
    for metric in sorted_metrics:
        if metric in pearson_correlations and metric in spearman_correlations:
            pearson_corr = pearson_correlations[metric][0]
            spearman_corr = spearman_correlations[metric][0]
            csvwriter.writerow([
                metric, 
                f"{pearson_corr:.4f}", 
                f"{spearman_corr:.4f}",
                f"{abs(pearson_corr):.4f}",
                f"{abs(spearman_corr):.4f}"
            ])

print(f"Saved simplified correlation summary to CSV: {simple_csv_path}")

print("\nAnalysis complete! Results saved to:", results_path) 