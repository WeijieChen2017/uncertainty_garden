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

def plot_heatmap(csv_path, save_path):
    """Plot heatmap from CSV file and save it"""
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Get the case key from the filename
    case_key = os.path.basename(csv_path).split('_')[0]
    
    # Create figure with larger size
    plt.figure(figsize=(12, 10))
    
    # Create heatmap without annotations
    sns.heatmap(df.iloc[:, 1:],  # Skip the first column (row labels)
                annot=False,  # Don't show values in cells
                cmap='YlOrRd',  # Use yellow-orange-red colormap
                cbar_kws={'label': 'MAE Loss'},  # Add label to colorbar
                square=True)  # Make cells square
    
    # Customize the plot
    plt.title(f'MAE Loss Heatmap - Case {case_key}', pad=20)
    plt.xlabel('Predictions and Ground Truth')
    plt.ylabel('Predictions and Ground Truth')
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved heatmap to {save_path}")
    plt.close()

def main():
    # Create a directory for heatmap plots if it doesn't exist
    heatmap_save_path = os.path.join(loss_scale_save_path, "heatmaps")
    os.makedirs(heatmap_save_path, exist_ok=True)
    
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
        
        # Create save path for the heatmap
        heatmap_file = f"{case_key}_mae_loss_heatmap.png"
        heatmap_path = os.path.join(heatmap_save_path, heatmap_file)
        
        print(f"\nProcessing case {case_key}...")
        plot_heatmap(csv_path, heatmap_path)

if __name__ == "__main__":
    main() 