import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Define the model name
model_name = "Theseus_v2_181_200_rdp1"
print(f"Using model: {model_name}")

# Setup paths
test_dict = {}
test_dict["project_name"] = model_name
test_dict["save_folder"] = "project_dir/"+test_dict["project_name"]+"/"
test_dict["loss_scale_save_folder"] = "loss_scale"

# Get the loss scale save path
loss_scale_save_path = os.path.join(test_dict["save_folder"], test_dict["loss_scale_save_folder"])

def plot_bar(csv_path, save_path):
    """Plot bar chart from CSV file and save it"""
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Get the case key from the filename
    case_key = os.path.basename(csv_path).split('_')[0]
    
    # Calculate mean MAE for each prediction and ground truth
    data = df.iloc[:, 1:].values  # Skip the first column (row labels)
    n_pred = data.shape[0] - 1  # Number of predictions (excluding ground truth)
    
    # Calculate mean MAE for each prediction (excluding self)
    pred_means = []
    for i in range(n_pred):
        # Get all values except diagonal (self)
        values = np.concatenate([data[i, :i], data[i, i+1:]])
        pred_means.append(np.mean(values))
    
    # Calculate mean MAE for ground truth
    gt_means = np.mean(data[-1, :-1])  # Last row, excluding last element (self)
    
    # Combine all means
    all_means = pred_means + [gt_means]
    
    # Create figure
    plt.figure(figsize=(12, 6))
    
    # Create bar plot with custom colors and edges
    bars = plt.bar(range(len(all_means)), all_means, edgecolor='black', linewidth=1)
    
    # Set colors: yellow for predictions, specified RGB for ground truth
    for i, bar in enumerate(bars):
        if i < n_pred:
            bar.set_color('#FFD700')  # Decent yellow for predictions
        else:
            bar.set_color('#3B4887')  # RGB(59, 72, 135) for ground truth
    
    # Set y-axis label and ticks
    plt.ylabel('Mean MAE')
    plt.yticks(np.arange(0, 0.11, 0.02))  # Ticks from 0 to 0.1 with step 0.02
    
    # Remove x-axis labels and ticks
    plt.xticks([])
    plt.xlabel('')
    
    # Set the range
    plt.ylim(0, 0.1)
    
    # Remove gaps at the edges
    plt.xlim(-0.5, len(all_means)-0.5)
    
    # Make the plot tight
    plt.tight_layout()
    
    # Save the plot with no extra padding
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0)
    print(f"Saved bar plot to {save_path}")
    plt.close()

def main():
    # Create a directory for bar plots if it doesn't exist
    bar_save_path = os.path.join(loss_scale_save_path, "bar_plots")
    os.makedirs(bar_save_path, exist_ok=True)
    
    # Find all CSV files in the loss scale directory
    csv_files = [f for f in os.listdir(loss_scale_save_path) if f.endswith('_mae_loss_table.csv')]
    
    if not csv_files:
        print("No MAE loss table CSV files found!")
        return
    
    print(f"Found {len(csv_files)} CSV files to process")
    
    # Process each CSV file
    for csv_file in csv_files:
        csv_path = os.path.join(loss_scale_save_path, csv_file)
        case_key = csv_file.split('_')[0]
        
        # Create save path for the bar plot
        bar_file = f"{case_key}_mae_loss_bar.png"
        bar_path = os.path.join(bar_save_path, bar_file)
        
        print(f"\nProcessing case {case_key}...")
        plot_bar(csv_path, bar_path)

if __name__ == "__main__":
    main() 