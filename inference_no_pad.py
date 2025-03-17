# here we inference all the models in the model pools and save the results

import os
import copy
import glob
import time
import sys

import numpy as np
import nibabel as nib
import torch.nn as nn
import torch.nn.functional as F

import torch

# Add parent directory to path so we can import modules from there
sys.path.append("../")

# from monai.networks.nets.unet import UNet
from monai.networks.layers.factories import Act, Norm
from monai.inferers import sliding_window_inference
from scipy.stats import zscore
# import bnn

# Import the model class from the model package
from model import UNet_Theseus
from utils import iter_all_order


model_list = [
    "Theseus_v2_181_200_rdp1",
]




print("Model index: ", end="")
current_model_idx = int(input()) - 1
print(model_list[current_model_idx])
time.sleep(1)

name = model_list[current_model_idx]


# for name in model_list:
test_dict = {}
test_dict = {}
test_dict["time_stamp"] = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
test_dict["project_name"] = name # "Bayesian_MTGD_v2_unet_do10_MTGD15"
test_dict["save_folder"] = "../project_dir/"+test_dict["project_name"]+"/"

# Ask user for which part of the dataset to process (0-4 for 5 parts)
print("Dataset part to process (0-4): ", end="")
dataset_part = int(input())
if dataset_part < 0 or dataset_part > 4:
    print("Invalid dataset part. Using part 0.")
    dataset_part = 0

# Map dataset part to GPU ID
# th1 and th4 completed
# we do it in 0, 5, 6, while the data is 1, 2, 4
# gpu_mapping = {0: 0, 1: 1, 2: 2, 3: 6, 4: 7}
gpu_mapping = {0: 0, 1: 0, 2: 5, 3: 6, 4: 6}
gpu_id = gpu_mapping[dataset_part]
test_dict["gpu_ids"] = [gpu_id]
print(f"Using GPU ID: {gpu_id}")

test_dict["eval_file_cnt"] = 0
test_dict["eval_save_folder"] = "array"
test_dict["special_cases"] = [
    # "03773",
    # "05628",
]
test_dict["save_tag"] = ""
test_dict["stride_division"] = 8

train_dict = np.load(test_dict["save_folder"]+"dict.npy", allow_pickle=True)[()]

test_dict["seed"] = train_dict["seed"]
test_dict["input_size"] = train_dict["input_size"]
test_dict["alt_blk_depth"] = [2,2,2,2,2,2,2]

print("input size:", test_dict["input_size"])
print("alt_blk_depth", test_dict["alt_blk_depth"])


# Create all necessary directories
for path in [test_dict["save_folder"], test_dict["save_folder"]+test_dict["eval_save_folder"]]:
    os.makedirs(path, exist_ok=True)

np.save(test_dict["save_folder"]+"test_dict.npy", test_dict)


# ==================== basic settings ====================

np.random.seed(test_dict["seed"])
gpu_list = ','.join(str(x) for x in test_dict["gpu_ids"])
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
print('export CUDA_VISIBLE_DEVICES=' + gpu_list)
device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')

model_list = sorted(glob.glob(os.path.join(test_dict["save_folder"], "model_best_*.pth")))
if "curr" in model_list[-1]:
    print("Remove model_best_curr")
    model_list.pop()
target_model = model_list[-1]
# target_model = test_dict["save_folder"]+test_dict["best_model_name"]
model = torch.load(target_model, map_location=torch.device('cpu'))
print("--->", target_model, " is loaded.")
# print(model)
# exit()

# ==================== data division ====================

data_div = np.load(os.path.join(test_dict["save_folder"], "data_division.npy"), allow_pickle=True)[()]
X_list = data_div['test_list_X']

# Modify input paths to include "../" prefix
X_list = ["../"+path if not path.startswith("../") else path for path in X_list]

# Append validation list to X_list
if 'val_list_X' in data_div:
    val_list = ["../"+path if not path.startswith("../") else path for path in data_div['val_list_X']]
    X_list.extend(val_list)
    print(f"Added {len(data_div['val_list_X'])} validation files to the list.")

# Divide the dataset into 5 parts
total_files = len(X_list)
X_list.sort()  # Sort the list first

# Process files with index % 5 == dataset_part
file_list = []
for i, file_path in enumerate(X_list):
    if i % 5 == dataset_part:
        file_list.append(file_path)

print(f"Processing part {dataset_part} of the dataset: {len(file_list)} files with indices mod 5 = {dataset_part}")

# Define the list of cases to process
to_do_cases = ["00008"]  # Add more case numbers as needed
print(f"Will process only files containing these case numbers: {to_do_cases}")

# Filter file_list to only include files containing the specified case numbers
filtered_file_list = []
for file_path in file_list:
    file_name = os.path.basename(file_path)
    if any(case in file_name for case in to_do_cases):
        filtered_file_list.append(file_path)

file_list = filtered_file_list
print(f"Found {len(file_list)} files matching the specified cases")

if test_dict["eval_file_cnt"] > 0:
    file_list = file_list[:test_dict["eval_file_cnt"]]

# ==================== training ====================
iter_tag = "test"
cnt_total_file = len(file_list)
cnt_each_cube = 1
model.eval()
model = model.to(device)

for cnt_file, file_path in enumerate(file_list):
    
    x_path = file_path
    y_path = file_path.replace("x", "y")
    file_name = os.path.basename(file_path)
    
    # Check if output file already exists
    output_array_save_dir = train_dict["save_folder"]+test_dict["eval_save_folder"]
    output_array_save_name = output_array_save_dir+"/"+file_name.replace(".nii.gz", test_dict["save_tag"]+"_array_no_pad.npy")
    
    if os.path.exists(output_array_save_name):
        print(iter_tag + " ===> Case[{:03d}/{:03d}]: ".format(cnt_file+1, cnt_total_file), x_path, "<--- SKIPPED (file exists)")
        continue
    
    print(iter_tag + " ===> Case[{:03d}/{:03d}]: ".format(cnt_file+1, cnt_total_file), x_path, "<---", end="") # 
    x_file = nib.load(x_path)
    y_file = nib.load(y_path)
    x_data = x_file.get_fdata()
    y_data = y_file.get_fdata()

    ax, ay, az = x_data.shape
    case_loss = 0

    input_data = x_data
    input_data = np.expand_dims(input_data, (0,1))
    input_data = torch.from_numpy(input_data).float().to(device)

    order_list, _ = iter_all_order(test_dict["alt_blk_depth"])
    # order_list = iter_all_order([2,2,2,2,2,2,2,2,2])
    order_list_cnt = len(order_list)
    output_array = np.zeros((order_list_cnt, ax, ay, az))

    for idx_es in range(order_list_cnt):
        with torch.no_grad():
            # print(order_list[idx_es])
            y_hat = sliding_window_inference(
                    inputs = input_data, 
                    roi_size = test_dict["input_size"], 
                    sw_batch_size = 64, 
                    predictor = model,
                    overlap=1/test_dict["stride_division"], 
                    mode="gaussian", 
                    sigma_scale=0.125, 
                    padding_mode="constant", 
                    cval=0.0, 
                    sw_device=device, 
                    device=device,
                    order=order_list[idx_es],
                    )
            output_array[idx_es, :, :, :] = y_hat.cpu().detach().numpy()

    # Save the original output_array as a .npy file
    os.makedirs(output_array_save_dir, exist_ok=True)  # Ensure directory exists
    np.save(output_array_save_name, output_array)
    print(f"Saved output array to: {output_array_save_name}")
