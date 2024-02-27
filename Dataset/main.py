import os
import shutil
from random import shuffle

# Define paths
dataset_dir = "FullDataset"
train_dir = os.path.join(dataset_dir, "train")
val_dir = os.path.join(dataset_dir, "validation")
test_dir = os.path.join(dataset_dir, "test")

# Create output directories if they don't exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Define proportions
train_proportion = 0.8
val_proportion = 0.1
test_proportion = 0.1

# List all files by folder
files_by_folder = {
    "0": os.listdir(os.path.join(dataset_dir, "0")),
    "1": os.listdir(os.path.join(dataset_dir, "1")),
    "2": os.listdir(os.path.join(dataset_dir, "2")),
}

# Split each folder's files based on proportions
for folder_name, files in files_by_folder.items():
    shuffle(files)

    num_train_files = int(train_proportion * len(files))
    num_val_files = int(val_proportion * len(files))
    num_test_files = int(test_proportion * len(files))

    train_files = files[:num_train_files]
    val_files = files[num_train_files:num_train_files + num_val_files]
    test_files = files[num_train_files + num_val_files:]

    # Create target sub-folders if needed
    os.makedirs(os.path.join(train_dir, folder_name), exist_ok=True)
    os.makedirs(os.path.join(val_dir, folder_name), exist_ok=True)
    os.makedirs(os.path.join(test_dir, folder_name), exist_ok=True)

    # Copy files to respective sub-folders
    for file in train_files:
        shutil.copy(os.path.join(dataset_dir, folder_name, file), os.path.join(train_dir, folder_name, file))

    for file in val_files:
        shutil.copy(os.path.join(dataset_dir, folder_name, file), os.path.join(val_dir, folder_name, file))

    for file in test_files:
        shutil.copy(os.path.join(dataset_dir, folder_name, file), os.path.join(test_dir, folder_name, file))

print(f"Dataset split into train ({len(train_files)}), validation ({len(val_files)}), and test ({len(test_files)}) sets, maintaining folder structure.")

