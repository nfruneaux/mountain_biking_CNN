# %% 20240408 Mountain Biking Image Segmentation CNN
# %% IMPORTS, INPUTS, SETUP OF CUSTOM DATASET CLASS
import time
import os
import xml.etree.ElementTree as ET
import math

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import resize

from matplotlib.colors import ListedColormap

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %% DIRECTORIES SUMMARY

#Training Images and XML Labels
image_dir = '/Users/nicholasfruneaux/Projects/Python Deep Learn/20230920 MTB NN-clean/Training Images' 
annotations_dir = '/Users/nicholasfruneaux/Projects/Python Deep Learn/20230920 MTB NN-clean/Training Images XML'

#Load Validation images
val_image_dir = "/Users/nicholasfruneaux/Projects/Python Deep Learn/20230920 MTB NN-clean/Validation Images"  #directory of validation images
model_name = "fcn_multi_label.pth"
save_dir = "/Users/nicholasfruneaux/Projects/Python Deep Learn/20230920 MTB NN-clean/Test_Model_Saves" #directory to save the NN model

#Load model for Visualization
visual_load_dir = save_dir
model_path = os.path.join(visual_load_dir, model_name)
visual_save_dir = save_dir #change the image save directory as needed

#Save model Parameters (.txt file)
model_parameters_save_dir = save_dir

os.makedirs(save_dir, exist_ok=True)

# %% DATA PREPARATION AND SETUP (extract labels, create masks, etc.)
to_tensor = transforms.ToTensor()

labels = ["Trees", "Trail", "Sky"] #["Trees", "Sky", "Trail"]  #"Bike", "Biker"] #, "Sky", "Trail"
labels_count = len(labels) 

class CustomDataset(Dataset):
    def __init__(self, image_dir, annotations_dir, labels, mode="single"):
        self.image_dir = image_dir
        self.annotations_dir = annotations_dir
        self.mode = mode
        self.labels = labels  # Can be a single label or multiple labels

        all_image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]
        self.image_files = [f for f in all_image_files if os.path.exists(os.path.join(annotations_dir, f.replace('.png', '.xml')))]
        self.max_width, self.max_height = self.get_max_dimensions()
        
        if mode not in ["single", "multi"]:
            raise ValueError("Mode should be either 'single' or 'multi'.")

        print("Initialized dataset with labels:", self.labels) #debugging
        
    def parse_annotation(self, annotation_path):
        tree = ET.parse(annotation_path)
        root = tree.getroot()
        annotations = []
        for object_elem in root.findall('object'):
            label = object_elem.find('name').text
            if label not in self.labels:  # Check against the labels list
                continue
            bndbox = object_elem.find('bndbox')
            xmin = int(round(float(bndbox.find('xmin').text)))
            ymin = int(round(float(bndbox.find('ymin').text)))
            xmax = int(round(float(bndbox.find('xmax').text)))
            ymax = int(round(float(bndbox.find('ymax').text)))
            annotations.append((label, (xmin, ymin, xmax, ymax)))
        return annotations
        
    def __len__(self):
        return len(self.image_files)
    
    def generate_single_label_mask(self, annotations, target_label, img):
        mask = torch.zeros(img.size[1], img.size[0])
        for box in annotations:
            if box[0] == target_label:
                _, (xmin, ymin, xmax, ymax) = box
                mask[ymin:ymax, xmin:xmax] = 1
        return mask
    
    def generate_multi_label_mask(self, annotations, labels, img):
        mask = torch.zeros(img.size[1], img.size[0])  # Initial mask
        label_to_index = {label: idx for idx, label in enumerate(labels)}  # Dynamically created based on labels
        
        for box in annotations:
            label, (xmin, ymin, xmax, ymax) = box
            # Ensure bounding box values are within image dimensions
            xmin, ymin = max(0, xmin), max(0, ymin)
            xmax, ymax = min(img.size[0], xmax), min(img.size[1], ymax)

            if label in label_to_index:
                mask[ymin:ymax, xmin:xmax] = label_to_index[label]
        return mask

    def get_max_dimensions(self):
        max_width, max_height = 0, 0
        for image_file in self.image_files:
            img_path = os.path.join(self.image_dir, image_file)
            img = Image.open(img_path).convert("RGB")
            width, height = img.size
            max_width = max(max_width, width)
            max_height = max(max_height, height)
        return max_width, max_height

    
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        img = Image.open(img_path).convert("RGB")
        annotation_path = os.path.join(self.annotations_dir, self.image_files[idx].replace('.png', '.xml'))
        annotations = self.parse_annotation(annotation_path)
        
        # Resize the image to max dimensions
        mask = torch.zeros(self.max_width, self.max_height)
        
        if self.mode == "single":
            mask = self.generate_single_label_mask(annotations, self.labels[0], img)
        elif self.mode == "multi":
            mask = self.generate_multi_label_mask(annotations, self.labels, img)
        
        # Check the dimensions of the mask and resize accordingly
        if mask.dim() == 1:  
            raise ValueError("Unexpected 1D mask. Ensure that the mask creation process is correct.")
        elif mask.dim() == 2:  # If the mask is a 2D tensor
            mask = mask.unsqueeze(0)  # Add dummy channel dimension [1, H, W]
            mask = resize(mask, (self.max_height, self.max_width))  # Resize based on H and W only
            mask = mask.squeeze(0)  # Remove dummy channel dimension [H, W]
        elif mask.dim() == 3:  
            # Handle this case as required.
            pass
        else:
            raise ValueError("Unexpected mask dimension. Expected 1D, 2D, or 3D tensor.")

        return to_tensor(img), mask.unsqueeze(0)

def custom_collate_fn(batch):
    # Extract images and masks from the batch
    images, masks = zip(*batch)

    # Find the max height and max width among all images in the batch
    max_height = max([img.size(1) for img in images])
    max_width = max([img.size(2) for img in images])
    
    # Resize each image and mask to the size of the largest image in the batch, preserving their aspect ratios
    resized_images = []
    resized_masks = []

    for img, mask in zip(images, masks):
        aspect_ratio = img.size(2) / img.size(1)  # width / height
        
        if aspect_ratio > 1:
            new_width = max_width
            new_height = int(max_width / aspect_ratio)
        else:
            new_width = int(max_height * aspect_ratio)
            new_height = max_height

        resized_img = resize(img, (new_height, new_width))
        resized_mask = resize(mask, (new_height, new_width))

        # Padding (if needed) to match the largest dimensions
        padding_left = (max_width - new_width) // 2
        padding_right = max_width - new_width - padding_left
        padding_top = (max_height - new_height) // 2
        padding_bottom = max_height - new_height - padding_top

        resized_img = torch.nn.functional.pad(resized_img, (padding_left, padding_right, padding_top, padding_bottom), mode='constant', value=0)
        resized_mask = torch.nn.functional.pad(resized_mask, (padding_left, padding_right, padding_top, padding_bottom), mode='constant', value=0)

        resized_images.append(resized_img)
        resized_masks.append(resized_mask)

    # Convert the list of resized images and masks to tensors
    images_tensor = torch.stack(resized_images, dim=0)  # Shape: [batch_size, C, H, W]
    
    # Stack along the 0th dimension (i.e., batch dimension)
    masks_tensor = torch.stack(resized_masks, dim=0)  # Shape: [batch_size, num_classes, H, W]
    
    return images_tensor, masks_tensor

# %% CNN MODEL SETUP
class FCN(nn.Module):
    def __init__(self, num_classes):
        super(FCN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=num_classes, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x):
        orig_size = x.size()[2:]  # Store the original height and width
        
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x))) 

        # Upsampling to the input size
        x = F.interpolate(x, size=orig_size, mode='bilinear', align_corners=False)
        
        x = self.conv4(x)
        
        return x

# Create the FCN instance
fcn = FCN(num_classes=labels_count).to(device)

# %% TRAINING PRE-CALCULATIONS

# Loss Function: Distribution Loss
def compute_actual_distribution(ground_truth, num_classes):
    # Consider only pixels with classes that model predicts
    valid_pixels_mask = (ground_truth >= 0) & (ground_truth < num_classes)
    filtered_ground_truth = ground_truth[valid_pixels_mask]
    
    # Calculate the distribution of valid pixels
    class_counts = filtered_ground_truth.bincount(minlength=num_classes)
    total_pixels = filtered_ground_truth.numel()
    return class_counts.float() / total_pixels

def distribution_loss(predicted, target):
    # Predicted is of shape [batch_size, num_classes, H, W]
    # Target is of shape [batch_size, H, W]

    # Ground Truth Distribution
    num_classes = predicted.shape[1]  # assuming predicted is [batch_size, num_classes, H, W]
    gt_distribution = compute_actual_distribution(target, num_classes)

    # Predicted Distribution
    pred_sum = predicted.sum(dim=[2,3])
    pred_distribution = pred_sum / pred_sum.sum(dim=1, keepdim=True)
    
    # KL-Divergence Loss
    loss = (F.kl_div(F.log_softmax(pred_distribution, dim=1), gt_distribution.unsqueeze(0) , reduction='batchmean'))

    return loss

# Loss Function: Entropy Loss
def compute_class_weights(dataset, num_classes):
    all_masks = torch.cat([masks for _, masks in dataset], dim=0)
    valid_pixels_mask = (all_masks >= 0) & (all_masks < num_classes)
    filtered_ground_truth = all_masks[valid_pixels_mask].long()
    class_counts = filtered_ground_truth.bincount(minlength=num_classes)
    class_counts = class_counts.clamp(min=1)
    weights = 1.0 / class_counts
    weights = weights / weights.sum()
    return weights

# %% MODEL TRAINING  

training_start_time = time.time()

num_epochs = 20
my_batch_size = 3
learning_rate = 0.0015
alpha_regul = 0.5
beta_dist = 0.5

dataset = CustomDataset(image_dir, annotations_dir, labels=labels, mode="multi")
dataloader = DataLoader(dataset, batch_size=my_batch_size, shuffle=True, collate_fn=custom_collate_fn)

weights = compute_class_weights(dataset, labels_count)

criterion = nn.CrossEntropyLoss(weight=weights).to(device)
optimizer = torch.optim.Adam(fcn.parameters(), lr=learning_rate)

print("TRAINING ONGOING")

for epoch in range(num_epochs):
    for i, (regions, masks) in enumerate(dataloader):
        
        regions = regions.to(device)
        
        # Forward pass
        outputs = fcn(regions)
            
        masks = masks.squeeze(dim=1).long().to(device)
        
        # Modify masks to set non-recognized labels to -1
        masks = torch.where((masks >= 0) & (masks < labels_count), masks, -1)
        
        ## Regularization Term    
        probs = F.softmax(outputs,dim=1)
        regularization_term = torch.mean(torch.std(probs, dim=[1,2,3]))
        print("all-in regul loss: ", alpha_regul * regularization_term)
        
        ## Distribution Term
        dist_loss = distribution_loss(probs, masks)
        print("all-in dist loss: ", beta_dist * dist_loss)
           
        # Calculate loss
        segmentation_loss = criterion(outputs, masks)
        print("segmentation loss: ", segmentation_loss)
        
        # Combine the losses
        total_loss = segmentation_loss + alpha_regul * regularization_term + beta_dist * dist_loss
            
        # Backward pass and optimization
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Print loss every [1] batches
        if (i+1) % 1 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}/{len(dataloader)}], Loss: {total_loss.item():.4f}")

torch.save(fcn.state_dict(), os.path.join(save_dir, model_name))

training_end_time = time.time()
total_training_time_sec = (training_end_time - training_start_time)
total_training_time_min = (total_training_time_sec / 60)

print("Finished Training | ",total_training_time_sec, " seconds | ",total_training_time_min," |")

# %% LOADING VALIDATION IMAGES

class ValidationDataset(Dataset):
    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        img = Image.open(img_path).convert("RGB")
        return to_tensor(img)

# DataLoader for validation data
val_dataset = ValidationDataset(val_image_dir)
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

# %% VISUALIZATION OF OUTPUTS 

number_of_validation_images = 9
flag_save_visualisation = True  # or False; user decision

thresholds = {
    'Trees': 0.0,
    'Sky': 0.0,
    'Trail': 0.0,
    # ... and so on for any other labels
}

# Define a colormap - adjust as needed
color_map = {
    #   R  G  B   
    0: [0, 1, 0],  
    1: [0, 0, 1], 
    2: [1, 0, 0],  
    #... add more colors if you have more labels
}

colors = [color_map[i] for i in range(len(color_map))]
custom_cmap = ListedColormap(colors)

fcn = FCN(num_classes=labels_count).to(device)
fcn.load_state_dict(torch.load(model_path))
fcn.eval()

# Inference for the first [x] validation images
counter = 0
for val_image in val_dataloader:
    if counter >= number_of_validation_images:  # Only process the specified number of validation images
        break

    val_image = val_image.to(device)
    
    #Run validation image through the loaded NN model in evaluation mode (no grad):
    with torch.no_grad():
        outputs = fcn(val_image)

    # Compute the label map
    label_map = torch.argmax(outputs, dim=1).squeeze().cpu().numpy()

    # Convert outputs to probabilities
    outputs_prob = torch.sigmoid(outputs).squeeze().cpu().numpy()

    # Filters out probabilities below specified thresholds (if any)
    for idx, label in enumerate(labels):
        if label in thresholds:
            outputs_prob[idx][outputs_prob[idx] < thresholds[label]] = 0

    # Visualize the result
    num_labels = len(labels) + 3  # +1 for the original image, +1 for combined visualization, +1 for label map
    fig, axes = plt.subplots(1, num_labels, figsize=(25, 5))  # Adjust figsize accordingly

    # Original image
    axes[0].imshow(val_image.squeeze().permute(1, 2, 0).cpu().numpy())
    axes[0].set_title("Original Image")
    axes[0].axis('off')  # Hide axes for better visualization

    # Visualizing each label output
    for idx, label in enumerate(labels):
        ax = axes[idx + 1]
        ax.imshow(outputs_prob[idx], cmap='jet')
        ax.set_title(f"{label} Label Heatmap")
        ax.axis('off')

    # Initialize combined_image before using it
    combined_image = np.zeros(val_image.squeeze().permute(1, 2, 0).cpu().numpy().shape[:2] + (3,))
    for idx, label in enumerate(labels):
        combined_image += np.expand_dims(outputs_prob[idx], axis=-1) * color_map[idx]

    # Display the combined image
    axes[-2].imshow(np.clip(combined_image, 0, 1))
    axes[-2].set_title("Combined Heatmap")
    axes[-2].axis('off')

    # Compute the Label Argmax Map and display
    with torch.no_grad():
        label_map = torch.argmax(torch.tensor(outputs_prob), dim=0)  # Get the label index for each pixel
    axes[-1].imshow(label_map, cmap=custom_cmap)  # or any other colormap you prefer
    axes[-1].set_title("Label Argmax Map")
    axes[-1].axis('off')

    # Save the visualized image to the output directory
    if flag_save_visualisation:
        plt.savefig(os.path.join(visual_save_dir, f"visualized_image_{counter}.png"))

    plt.tight_layout()
    plt.show()
    
    counter += 1

# %% SAVE THE MODEL PARAMETERS IN A TEXT FILE 

def save_model_parameters(fcn, save_path):
    global learning_rate, num_epochs, batch_size  # Access global variables
    
    # Check if directory exists, and create if not
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    file_path = os.path.join(save_path, 'model_parameters.txt')
    with open(file_path, 'w') as f:
        # Write the model parameters
        for name, module in fcn.named_modules():
            if isinstance(module, (nn.Conv2d, nn.MaxPool2d)):
                f.write(f"Layer: {name}\n")
                if hasattr(module, 'in_channels'):
                    f.write(f"In channels: {module.in_channels}\n")
                if hasattr(module, 'out_channels'):
                    f.write(f"Out channels: {module.out_channels}\n")
                if hasattr(module, 'kernel_size'):
                    f.write(f"Kernel size: {module.kernel_size}\n")
                if hasattr(module, 'stride'):
                    f.write(f"Stride: {module.stride}\n")
                if hasattr(module, 'padding'):
                    f.write(f"Padding: {module.padding}\n")
                f.write("\n")

        # Write the training parameters
        f.write(f"Learning rate: {learning_rate}\n")
        f.write(f"Number of epochs: {num_epochs}\n")
        f.write(f"Batch size: {my_batch_size}\n")
        f.write(f"alpha_regul: {alpha_regul}\n")
        f.write(f"beta_dist: {beta_dist}\n")
        f.write(f"Criterion: {criterion}\n")
        f.write(f" Labels Trained: {labels}\n")
        f.write(f"Training Time: {total_training_time_sec} seconds | {total_training_time_min}")

# Usage:
save_model_parameters(fcn, save_path = model_parameters_save_dir)
