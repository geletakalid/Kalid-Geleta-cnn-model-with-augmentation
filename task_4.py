import os
import shutil
import random

# Define paths
base_dir = r"C:\Users\kalid\Desktop\pythonProject34\PetImages"  # Original dataset folder
output_dir = r"C:\Users\kalid\Desktop\pythonProject34\PetImages_Split"  # New folder for split dataset

train_dir = os.path.join(output_dir, "train")
validation_dir = os.path.join(output_dir, "validation")

# Define train-validation split ratio (e.g., 80% train, 20% validation)
split_ratio = 0.8

# Get all class subdirectories (i.e., 'Cat' and 'Dog')
class_names = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

# Process each class separately
for class_name in class_names:
    class_folder = os.path.join(base_dir, class_name)
    images = [img for img in os.listdir(class_folder) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]

    random.shuffle(images)  # Shuffle images for randomness
    split_index = int(len(images) * split_ratio)

    train_images = images[:split_index]
    validation_images = images[split_index:]

    # Create class subfolders inside train and validation directories
    os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
    os.makedirs(os.path.join(validation_dir, class_name), exist_ok=True)

    # Move images to respective directories
    for img in train_images:
        shutil.move(os.path.join(class_folder, img), os.path.join(train_dir, class_name, img))

    for img in validation_images:
        shutil.move(os.path.join(class_folder, img), os.path.join(validation_dir, class_name, img))

    print(f"Class '{class_name}': {len(train_images)} train images, {len(validation_images)} validation images.")

print("Dataset split complete!")