import os
import shutil

# Paths
base_path = "archive"  # Replace with the path to your 'original' folder
output_path = "dataset"  # Replace with the path to your 'new' folder

# Ensure the dataset folder exists
os.makedirs(output_path, exist_ok=True)

# Subfolders in archive
subfolders = ["test", "train", "val"]

for subfolder in subfolders:
    subfolder_path = os.path.join(base_path, subfolder)
    
    # Iterate over each class folder in subfolder
    for class_folder in os.listdir(subfolder_path):
        class_folder_path = os.path.join(subfolder_path, class_folder)
        
        if os.path.isdir(class_folder_path):  # Check if it's a directory
            # Destination path for the class folder
            destination_path = os.path.join(output_path, class_folder)
            
            if not os.path.exists(destination_path):
                # Move or merge the folder
                shutil.move(class_folder_path, destination_path)
            else:
                # If the folder already exists, move contents
                for item in os.listdir(class_folder_path):
                    item_path = os.path.join(class_folder_path, item)
                    shutil.move(item_path, destination_path)
                    
                # Remove the now empty class folder
                os.rmdir(class_folder_path)

print("All data has been moved to the 'dataset' folder.")
