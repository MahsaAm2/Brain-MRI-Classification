"""
python submission.py --data-dir /path/to/data-dir --predictions-file-path /path/to/submission.csv
"""

from typing import Optional

import click
from pathlib import Path
import pydicom
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from skimage import morphology
from PIL import Image
from PIL import ImageOps
import torch
import os
import cv2
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models, transforms  # For pre-trained models like VGG16 and data transforms
from sklearn.svm import SVC
import joblib
from torchvision.models import ResNet50_Weights
import tensorflow as tf
from tensorflow.keras.models import load_model

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import regularizers, Input
from tensorflow.keras.layers import Flatten, BatchNormalization, Dense, Activation, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import ResNet50

HERE = Path(__file__).absolute().resolve().parent

device = "cpu"#torch.device("cuda" if torch.cuda.is_available() else "cpu")


def process_and_save_dicom(src_folder, dst_folder):
    for root, _, files in os.walk(src_folder):
        for file in files:
            if file.endswith(".dcm"):
                dicom_path = os.path.join(root, file)
                dicom_data = pydicom.dcmread(dicom_path)
                
                if 'T2W_TSE'  in dicom_data.ProtocolName:
                    p_name = 'T2W_'
                    # continue
                elif 'T2W_FLAIR'  in dicom_data.ProtocolName:
                    p_name = 'FLAIR_'
                elif 'T1W'  in dicom_data.ProtocolName:
                    p_name = 'T1W_'
                    # continue

                # Extract metadata
                slice_orientation = dicom_data[(0x2001, 0x100b)].value
                
                pixel_array = dicom_data.pixel_array
                protocol_name = dicom_data.ProtocolName.replace(" ", "_")  # Remove spaces in description
                instance_number = dicom_data.InstanceNumber

                # Extract Image Orientation (Patient) tag to determine left-right orientation
                image_orientation = dicom_data[0x0020, 0x0037].value

                # Check if the first element of image_orientation is negative
                # This would indicate a left-to-right orientation, requiring a flip
                if image_orientation[0] < 0:
                    # Flip the image horizontally to make it right-to-left
                    pixel_array = np.fliplr(pixel_array)

                # Get the original folder name (parent of current root)
                original_folder_name = os.path.basename(root)

                # Create a new folder for the subject using original folder name + PatientID
                subject_folder_name = f"{original_folder_name}"
                subject_folder = os.path.join(dst_folder, p_name+subject_folder_name ,)
                os.makedirs(subject_folder, exist_ok=True)

                # Construct the save path with instance number in the filename
                save_path = os.path.join(subject_folder, f"{os.path.splitext(file)[0]}_{protocol_name}_Instance{instance_number}.npy")

                # Save the numpy array
                np.save(save_path, pixel_array)
                
def crop_imgs_and_preprocess_imgs(set_name, add_pixels_value=10, img_size=(224, 224), min_contour_area=500):
    """
    Finds the extreme points on the image and crops the rectangular region out of them
    for each slice in the 3D .npy data.
    """
    set_new = []
    
    # Iterate over the set_name which contains 3D volumes (multiple slices)
    for volume in set_name:
        cropped_slices = []
        selected_slices = volume[4:8]
        # Iterate over each 2D slice in the 3D volume
        for img in selected_slices:
            # Ensure the image is in grayscale format
            if len(img.shape) == 3:  # (height, width, channels)
                image_2d = img[:, :, 0]  # Use the first channel for cropping
            else:
                image_2d = img
                
            binary_image = image_2d > np.percentile(image_2d, 60)

            # Remove small connected components (noise) - optional
            binary_image = morphology.remove_small_objects(binary_image, min_size=100)

            coords = np.column_stack(np.where(binary_image))
            if coords.size == 0:
                # If no valid cropping, resize the original image
                cropped_image = cv2.resize(
                    img,  # Fix here: use img instead of undefined cropped_image
                    dsize=img_size,
                    interpolation=cv2.INTER_CUBIC
                )
            else:
                # Crop the image using the bounding box of non-zero pixels
                min_row, min_col = coords.min(axis=0)
                max_row, max_col = coords.max(axis=0)
                cropped_image = img[min_row:max_row+1, min_col:max_col+1]

                # Resize cropped image as you originally had it
                cropped_image = cv2.resize(
                    cropped_image,  # Keep this as cropped_image as it was
                    dsize=img_size,
                    interpolation=cv2.INTER_CUBIC
                )

            # Ensure the resized image has a channel dimension
            if len(cropped_image.shape) == 2:
                cropped_image = np.expand_dims(cropped_image, axis=-1)
            
            cropped_slices.append(cropped_image)

        # Append the cropped volume (list of 2D slices)
        set_new.append(np.array(cropped_slices))

    return np.array(set_new)                
                


def extract_instance_number(filename):
    base_name = os.path.splitext(os.path.basename(filename))[0]
    instance_str = base_name.split('_')[-1]  # Get the last part after splitting by '_'
    instance_number = ''.join(filter(str.isdigit, instance_str))    
    return int(instance_number)


# def preprocess_imgs(set_name, img_size):
#     set_new = []
#     for volume in set_name:
#         # Discard all slices except slices 5, 6, 7, and 8
#         selected_slices = volume[4:8]  # Indices 4 to 7 correspond to slices 5 to 8

#         # Append the processed volume to the new set
#         set_new.append(np.array(selected_slices))
    
#     return np.array(set_new)



import numpy as np

def average_slices_4d_array(slice_array):
    """
    Averages the four slices of each sample in a 4D array and returns a 3D array.

    Parameters:
    - slice_array: 4D numpy array (n_samples, 4, img_height, img_width)
                   where each sample contains 4 slices (grayscale images).

    Returns:
    - averaged_array: 3D numpy array (n_samples, img_height, img_width)
                      where each sample is the averaged result of its 4 slices.
    """
    # Ensure the input is a 4D array with 4 slices per sample
    assert slice_array.shape[1] == 4, "Each sample must have exactly 4 slices."

    # Average the slices along the second axis (the slice dimension)
    averaged_array = np.mean(slice_array, axis=1)
    
    return averaged_array


def crop_background(image, target_size=(256, 256)):
        """
        Crops the black background from the image using a tighter thresholding and bounding box method.
        Assumes that the image has dimensions (height, width, channels).
        """
        # If image has multiple channels, use only the first channel for thresholding
        if len(image.shape) == 3:  # (height, width, channels)
            image_2d = image[:, :, 0]  # Use the first channel for cropping
        else:
            image_2d = image

        # Adjusted thresholding: using a higher percentile for a tighter crop
        binary_image = image_2d > np.percentile(image_2d, 60)  # Increase percentile to 15 or higher if needed

        # Remove small connected components (noise) - optional
        binary_image = morphology.remove_small_objects(binary_image, min_size=100)

        # Find the bounding box
        coords = np.column_stack(np.where(binary_image))
        if coords.size == 0:
            # Handle the case where the image is completely black
            return image  # Return the original image if there's no valid crop

        min_row, min_col = coords.min(axis=0)
        max_row, max_col = coords.max(axis=0)

        # Apply a tighter crop using the bounding box
        cropped_image = image[min_row:max_row+1, min_col:max_col+1]

        # Convert cropped image to PIL Image and ensure it is in 'L' mode (grayscale)
        cropped_image = Image.fromarray(cropped_image.squeeze())
        cropped_image = ImageOps.grayscale(cropped_image)
        
        cropped_image = cropped_image.resize(target_size, Image.LANCZOS)
        cropped_image = np.array(cropped_image)

        # Ensure the resized image has the correct shape (target_size, target_size, channels)
        if len(cropped_image.shape) == 2:  # Ensure it's still 2D
            cropped_image = np.expand_dims(cropped_image, axis=-1)

        return cropped_image
    
    
def compute_histograms_for_subject_h(subject_folder, num_bins=255):
    # Get all .npy files in the subject's folder
    npy_files = [os.path.join(subject_folder, f) for f in os.listdir(subject_folder) if f.endswith(".npy")]
    
    if not npy_files:
        print(f"No .npy files found for the subject folder: {subject_folder}")
        return None
    
    # Sort files to maintain order (assuming filenames include instance numbers)
    npy_files.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_')[-1].replace("Instance", "")))

    all_histograms = []

    for file_path in npy_files:
        # Load the image data from the .npy file
        image_data = np.load(file_path)
        
        # Resize the image to 256x256
        resized_image = cv2.resize(image_data, (256, 256), interpolation=cv2.INTER_LINEAR)
        
        # Split the image into left and right hemispheres
        left_hemisphere = resized_image[:, :128]
        right_hemisphere = resized_image[:, 128:]
        
        # Exclude zeros and compute histograms
        left_hist, _ = np.histogram(left_hemisphere[left_hemisphere > 0], bins=num_bins, range=(0, 255))
        right_hist, _ = np.histogram(right_hemisphere[right_hemisphere > 0], bins=num_bins, range=(0, 255))
        
        # If no non-zero pixels are found, skip this slice
        if left_hist.sum() == 0 and right_hist.sum() == 0:
            print(f"Skipping slice with no non-zero pixels in file: {file_path}")
            continue
        
        # Combine histograms for left and right hemispheres
        combined_hist = np.concatenate([left_hist, right_hist])
        
        # Normalize histogram to ensure equal contribution of each slice
        if combined_hist.sum() > 0:
            combined_hist = combined_hist / np.sum(combined_hist)
        
        all_histograms.append(combined_hist)
    
    # If no valid histograms were found, return None
    if not all_histograms:
        print(f"No valid histograms found for the subject folder: {subject_folder}")
        return None
    
    # If there are more than 16 slices, take the first 16
    # If there are fewer than 16 slices, pad with zeros
    num_slices = len(all_histograms)
    if num_slices < 16:
        all_histograms.extend([np.zeros(2 * num_bins)] * (16 - num_slices))
    elif num_slices > 16:
        all_histograms = all_histograms[:16]
    
    # Flatten the list of histograms into a single feature vector
    subject_histogram_vector = np.concatenate(all_histograms)
    
    return subject_histogram_vector    


def compute_histograms_for_subject(subject_folder, num_bins=255):
    # Get all .npy files in the subject's folder
    npy_files = [os.path.join(subject_folder, f) for f in os.listdir(subject_folder) if f.endswith(".npy")]
    
    if not npy_files:
        print(f"No .npy files found for the subject folder: {subject_folder}")
        return None
    
    # Sort files to maintain order (assuming filenames include instance numbers)
    npy_files.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_')[-1].replace("Instance", "")))

    all_histograms = []

    # We only want slices 5, 6, 7, and 8 (index 4 to 7)
    for i, file_path in enumerate(npy_files[4:8]):
        # Load the image data from the .npy file
        image_data = np.load(file_path)
        
        # Resize the image to 256x256
        resized_image = cv2.resize(image_data, (256, 256), interpolation=cv2.INTER_LINEAR)

        # Exclude zeros and compute histograms
        resized_image_nonzero = resized_image[resized_image > 0]

        resized_image_hist, _ = np.histogram(resized_image[resized_image > 0], bins=num_bins, range=(0, 255))
        
        # If no non-zero pixels are found, skip this slice
        if resized_image_hist.sum() == 0:
            print(f"Skipping slice with no non-zero pixels in file: {file_path}")
            continue
        
        # Normalize histogram to ensure equal contribution of each slice
        if resized_image_hist.sum() > 0:
            resized_image_hist = resized_image_hist / np.sum(resized_image_hist)
        
        all_histograms.append(resized_image_hist)
    
    # If no valid histograms were found, return None
    if not all_histograms:
        print(f"No valid histograms found for the subject folder: {subject_folder}")
        return None
    
    # Flatten the list of histograms into a single feature vector
    subject_histogram_vector = np.concatenate(all_histograms)
    
    return subject_histogram_vector


model_pre = models.resnet50()

# Load the saved weights
model_pre.load_state_dict(torch.load('resnet50_weights.pth', map_location=device ,weights_only=False))

# Freeze layers if needed
for param in model_pre.parameters():
    param.requires_grad = False

class CustomResNetF(nn.Module):
    def __init__(self, base_model):
        super(CustomResNetF, self).__init__()
        self.base_model = base_model
        self.base_model.fc = nn.Flatten()  # Flatten the output from ResNet50
        self.fc1 = nn.Linear(2048, 32)  
        self.batch_norm1 = nn.BatchNorm1d(32)
        self.fc2 = nn.Linear(32, 16)
        self.batch_norm2 = nn.BatchNorm1d(16)
        self.fc3 = nn.Linear(16, 1)
        self.dropout = nn.Dropout(0.4)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.base_model(x)
        x = self.fc1(x)
        x = self.batch_norm1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.batch_norm2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.activation(x)
        return x
class CustomResNetT1(nn.Module):
    def __init__(self, base_model):
        super(CustomResNetT1, self).__init__()
        self.base_model = base_model
        self.base_model.fc = nn.Flatten()  # Flatten the output from ResNet50
        self.fc1 = nn.Linear(2048, 16)  
        self.batch_norm1 = nn.BatchNorm1d(16)
        self.fc2 = nn.Linear(16, 1)
        self.dropout = nn.Dropout(0.4)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.base_model(x)
        x = self.fc1(x)
        x = self.batch_norm1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.activation(x)
        return x

class CustomResNetT2(nn.Module):
    
    def __init__(self, base_model):
        super(CustomResNetT2, self).__init__()
        self.base_model = base_model
        self.base_model.fc = nn.Flatten()  # Flatten the output from ResNet50
        self.fc1 = nn.Linear(2048, 32)  
        self.batch_norm1 = nn.BatchNorm1d(32)
        self.fc2 = nn.Linear(32, 16)
        self.batch_norm2 = nn.BatchNorm1d(16)
        self.fc3 = nn.Linear(16, 1)
        self.dropout = nn.Dropout(0.5)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.base_model(x)
        x = self.fc1(x)
        x = self.batch_norm1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.batch_norm2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.activation(x)
        return x
    
class MLPClassifier(nn.Module):
    def __init__(self, input_size):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.sigmoid(self.fc3(x))
        return x
    
    
    
################################################################    
################################################################
# CNN model
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        
        # First convolutional block
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Second convolutional block
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Third convolutional block
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Flatten and Dense layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=128*16*16, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=64)
        self.fc3 = nn.Linear(in_features=64, out_features=1)

    def forward(self, x):
        # First convolutional block
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        
        # Second convolutional block
        x = F.relu(self.conv3(x))
        x = self.pool2(x)
        
        # Third convolutional block
        x = F.relu(self.conv4(x))
        x = self.pool3(x)
        
        # Flatten and Dense layers
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        
        return x





def load_data(dir_path, img_size=(100,100)):
    """
    Load resized images as np.arrays to workspace
    """
    X = []

    # for path in tqdm(sorted(os.listdir(dir_path))):
    x_temp = []
    # print("dir_path: ", dir_path)
    if not dir_path.startswith('.'):
        #print("dir_path + path: ", dir_path + path)
        npy_files = [os.path.join(dir_path , f) for f in os.listdir(dir_path) if f.endswith(".npy")]
        if not npy_files:
            print("No .npy files found for the selected subject.")
            return
        # Sort files based on extracted instance numbers
        npy_files = sorted(npy_files, key=extract_instance_number)
        #print("npy_files: ", npy_files)
        for i, file_path in enumerate(npy_files):
            # Load the image data from the .npy file
            image_data = np.load(file_path)
            # Resize the image to a consistent shape (img_size)
            image_data = cv2.resize(image_data, img_size)
            x_temp.append(image_data)

        x_temp = np.array(x_temp)
        if len(x_temp) < 16:
            padding = np.zeros((16 - len(x_temp),) + x_temp.shape[1:])
            x_temp = np.concatenate((x_temp, padding), axis=0)
        x_temp = x_temp[:16]
        X.append(x_temp)
    X = np.array(X)
    X = X.astype(np.float32) 
    # print("X.shape",X.shape)
    # print("len(X[0])",len(X[0]))
#     y = np.array(y)
    return X

@click.command()
@click.option(
    "--data-dir",
    type=Path,
    help="path to data directory, which consists of folders of Dicom files, each one corresponding to a Dicom series.",
)
@click.option("--predictions-file-path", type=Path)
def main(data_dir: Path, predictions_file_path: Path):
   

# Check if a GPU is available and use it if possible
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    IMG_SIZE = (224,224)
   
    test_folder = data_dir
    test_folder_proc = "test_folder_proc"
    pred_tresh = 0.5
    thresh = 1
    all_preds = []

    print(tf.__version__)
    ###################
    process_and_save_dicom(test_folder, test_folder_proc)
    
    
    ##############################################################
    ##############################################################
    ## cnn model
    model_cnn = CNNModel()
    # model_cnn.load_state_dict(torch.load('model_cnn_128.pth'))
    model_cnn.load_state_dict(torch.load('model_cnn_128.pth', map_location=device))

    ###############################################################
    ###############################################################
    
    #mlp_b
    input_size = 16 * 2 * 255
    model_test_h = MLPClassifier(input_size=input_size)
    model_test_h.load_state_dict(torch.load('model_300.pth', map_location=device ,weights_only=False))
     #mlp_e
     
    #svm_b 
    model_test_T2 = SVC(kernel='rbf', C=30.0, probability=True) 
    model_test_T2 = joblib.load('svm_model_t2.pth')
    pca_t2 = joblib.load('pca_model_T2.pkl')
    model_test_T1 = SVC(kernel='poly', C=50.0, probability=True) 
    model_test_T1 = joblib.load('svm_model_t1.pth')
    pca_t1 = joblib.load('pca_model_T1.pkl')
    model_test_F = SVC(kernel='rbf', C=20.0, probability=True) 
    model_test_F = joblib.load('svm_model_flair.pth')
    pca_f = joblib.load('pca_model_flair.pkl')
    #svm_e
    
    #resnet_b 
    model_test_f = CustomResNetF(model_pre)
    model_test_t1 = CustomResNetT1(model_pre)
    model_test_t2 = CustomResNetF(model_pre)
    
    
    model_test_f.load_state_dict(torch.load('best_model_f.pt', map_location=device ,weights_only=False))
    model_test_t1.load_state_dict(torch.load('best_model_t1.pt', map_location=device ,weights_only=False))
    # model_test_t1.load_state_dict(torch.load('best_model_v2_0.4.pt', map_location=device ,weights_only=False))
    
    #k_b
    # model_test_t1_k = load_model('best_model_AB_vs.keras')
    model_k_T1 = tf.keras.models.load_model('best_model_AB_vs_O.keras')
    model_k_T2 = tf.keras.models.load_model('best_model_n.keras')
    model_k_F = tf.keras.models.load_model('best_model_AB.keras')

    
    # model_k_2 = tf.keras.models.load_model('best_model_AB_vs_n.keras')

    ###e

    # model_test_t2.load_state_dict(torch.load('best_model_f.pt', map_location=device ,weights_only=False))


    model_test_f.eval()
    model_test_t2.eval()
    model_test_t1.eval()

    #resnet_e
    ####################################################
    ####################################################
    
    
    ####################################################
    IMG_HEIGHT = 128
    IMG_WIDTH = 128
    
    
    subject_folders = [os.path.join(test_folder_proc, f) for f in os.listdir(test_folder_proc) if os.path.isdir(os.path.join(test_folder_proc, f))]

    with torch.no_grad():

        for inputs in subject_folders:
            
            ##################################################################
            ### cnn
            
            abnormal_count = 0
            npy_slices = [file for file in os.listdir(inputs) if file.endswith('.npy')]
            for slice_data_path in npy_slices:
                
                
                slice_data = np.load(os.path.join(inputs, slice_data_path))
                
                slice_data = slice_data.astype(np.float32)
                
                
                if len(slice_data.shape) == 2:
                    slice_data = np.expand_dims(slice_data, axis=-1)
                        # Ensure slice_data is of shape (IMG_HEIGHT, IMG_WIDTH, channels)
                if slice_data.shape[0] != IMG_HEIGHT or slice_data.shape[1] != IMG_WIDTH:
                        #print(type(slice_data))
                    slice_data = cv2.resize(slice_data, (IMG_HEIGHT, IMG_WIDTH))
                slice_data = torch.Tensor(slice_data)
                slice_data = torch.squeeze(slice_data, axis=-1)
                slice_data = torch.unsqueeze(slice_data, axis=0)
                slice_data = torch.unsqueeze(slice_data, axis=0)
                #print(slice.shape,device)
                slice_data.to(device)
                slice_predictions = model_cnn(slice_data)
                if slice_predictions[0] < 0.5:
                    abnormal_count = abnormal_count + 1
        # Count the number of abnormal predictions
        
        # If at least `abnormal_threshold` slices are predicted as abnormal (0), label the subject as abnormal (0)
            if abnormal_count >= 1:
                cnn_pred = 1  # Abnormal
            
            else:
                cnn_pred = 0  # Normal
            
            
            
            
            ####################################################################
            
            
            
            # print("inputs: ", inputs)
            val_preds = []
            histogram_vector = compute_histograms_for_subject(inputs, 255)
            #****
            histogram_vector_h = compute_histograms_for_subject_h(inputs, 255)
            if histogram_vector_h is not None:
                histogram_vector_h = torch.tensor(histogram_vector_h, dtype=torch.float32)
                outputs = model_test_h(histogram_vector_h.unsqueeze(0)).squeeze(1)  # Add batch dimension
                        
            preds = (outputs > pred_tresh).cpu().numpy()  # Binarize predictions
            #***
            X_train = load_data(inputs, IMG_SIZE)
            
            
            if 'T2W'  in inputs:
       
                
                X_train_crop = crop_imgs_and_preprocess_imgs(set_name=X_train)
                #X_train_prep = preprocess_imgs(set_name=X_train_crop, img_size=IMG_SIZE)
                averaged_X = average_slices_4d_array(X_train_crop)
                # print("averaged_X.shape: ", averaged_X.shape)
            #K_b
                averaged_X_rgb = np.repeat(averaged_X, 3, axis=-1)  # Convert (224, 224, 1) to (224, 224, 3)
                # print("averaged_X_rgb.shape: ", averaged_X_rgb.shape)

                # print("averaged_X.shape: ", averaged_X.shape)
                predictions_prob = model_k_T2.predict(averaged_X_rgb,verbose=0)
                #print(predictions_prob[:10])
                predictions = [1 if x > pred_tresh else 0 for x in predictions_prob]

            #K_e
                averaged_tensor = torch.from_numpy(averaged_X.astype(np.float32))  # Convert to float32 tensor
                # print("averaged_tensor.shape: ", averaged_tensor.shape)
                
                # X_train_tensor = torch.from_numpy(X_train) 
                averaged_tensor = averaged_tensor.permute(0, 3, 1, 2)

                averaged_tensor = averaged_tensor.repeat(1, 3, 1, 1)
                
                val_preds_batch = model_test_t2(averaged_tensor).squeeze()
                if isinstance(val_preds_batch, np.ndarray):
                   val_preds_batch = torch.tensor(val_preds_batch)
                val_preds.append(predictions)  # Store as tensors
                # print("val_preds: ", val_preds)
                
                # val_preds = [torch.tensor(pred) if isinstance(pred, np.ndarray) else pred for pred in val_preds]

                val_preds = [torch.tensor(pred) if isinstance(pred, (list, np.ndarray)) else pred for pred in val_preds]

                # Concatenate the list of tensors
                val_preds = torch.cat([pred.unsqueeze(0) if pred.dim() == 0 else pred for pred in val_preds]).cpu().numpy()


                # val_preds = [pred.unsqueeze(0) for pred in val_preds]
            #     val_preds = [pred.unsqueeze(0) if pred.dim() == 0 else pred for pred in val_preds]

            # # Convert predictions and labels to numpy arrays after the loop
            #     val_preds = torch.cat(val_preds).cpu().numpy()  # Concatenate and move to CPU
                val_preds_class = (val_preds > pred_tresh).astype(int)
                
            #SVM_b    
                if histogram_vector is not None:
                    histogram_vector= np.array(histogram_vector)
                    reduced_histograms_test = pca_t2.transform(histogram_vector.reshape(1, -1))

                outputs = model_test_T2.predict(reduced_histograms_test)
            #SVM_e    
              
                #all_preds.append(max(val_preds_class,outputs,preds))
                all_preds.append(1 if (predictions + preds + outputs + val_preds_class + cnn_pred) >= thresh else 0)
                #all_preds.append(predictions)


            elif 'FLAIR'  in inputs:
                # X_train_tensor = torch.from_numpy(X_train) 
                
                X_train_crop = crop_imgs_and_preprocess_imgs(set_name=X_train)
                #X_train_prep = preprocess_imgs(set_name=X_train_crop, img_size=IMG_SIZE)
                averaged_X = average_slices_4d_array(X_train_crop)
                
            #K_b
                averaged_X_rgb = np.repeat(averaged_X, 3, axis=-1)  # Convert (224, 224, 1) to (224, 224, 3)
                # print("averaged_X_rgb.shape: ", averaged_X_rgb.shape)

                # print("averaged_X.shape: ", averaged_X.shape)
                predictions_prob = model_k_F.predict(averaged_X_rgb,verbose=0)
                #print(predictions_prob[:10])
                predictions = [1 if x > pred_tresh else 0 for x in predictions_prob]

            #K_e
                
                #print("averaged_X.shape: ", averaged_X.shape)
                averaged_tensor = torch.from_numpy(averaged_X.astype(np.float32))  # Convert to float32 tensor
                # print("averaged_tensor.shape: ", averaged_tensor.shape)
                averaged_tensor = averaged_tensor.permute(0, 3, 1, 2)


                averaged_tensor = averaged_tensor.repeat(1, 3, 1, 1)
                
                val_preds_batch = model_test_f(averaged_tensor).squeeze()
                if isinstance(val_preds_batch, np.ndarray):
                    val_preds_batch = torch.tensor(val_preds_batch)
                val_preds.append(val_preds_batch)  # Store as tensors
                # print("val_preds: ", val_preds)
                
                # val_preds = [torch.tensor(pred) if isinstance(pred, np.ndarray) else pred for pred in val_preds]
                val_preds = [torch.tensor(pred) if isinstance(pred, (list, np.ndarray)) else pred for pred in val_preds]

                # Concatenate the list of tensors
                val_preds = torch.cat([pred.unsqueeze(0) if pred.dim() == 0 else pred for pred in val_preds]).cpu().numpy()
                # val_preds = [pred.unsqueeze(0) for pred in val_preds]
            #     val_preds = [pred.unsqueeze(0) if pred.dim() == 0 else pred for pred in val_preds]

            # # Convert predictions and labels to numpy arrays after the loop
            #     val_preds = torch.cat(val_preds).cpu().numpy()  # Concatenate and move to CPU
                val_preds_class = (val_preds > pred_tresh).astype(int)
           
            #SVM_b                   
                if histogram_vector is not None:
                    histogram_vector= np.array(histogram_vector)
                    reduced_histograms_test = pca_f.transform(histogram_vector.reshape(1, -1))

                outputs = model_test_F.predict(reduced_histograms_test)
            #SVM_e    
                
                #all_preds.append(max(val_preds_class,outputs,preds))
                # all_preds.append(predictions)
                all_preds.append(1 if (predictions + preds + outputs + val_preds_class + cnn_pred) >= thresh else 0)
                
            elif 'T1W'  in inputs:

                
                X_train_crop = crop_imgs_and_preprocess_imgs(set_name=X_train)
                #X_train_prep = preprocess_imgs(set_name=X_train_crop, img_size=IMG_SIZE)
                averaged_X = average_slices_4d_array(X_train_crop)
                # Assuming `averaged_X` is your input

            #K_b
                averaged_X_rgb = np.repeat(averaged_X, 3, axis=-1)  # Convert (224, 224, 1) to (224, 224, 3)
                # print("averaged_X_rgb.shape: ", averaged_X_rgb.shape)

                # print("averaged_X.shape: ", averaged_X.shape)
                predictions_prob = model_k_T1.predict(averaged_X_rgb,verbose=0)
                #print(predictions_prob[:10])
                predictions = [1 if x > pred_tresh else 0 for x in predictions_prob]

            #K_e
                
                
                
                averaged_tensor = torch.from_numpy(averaged_X.astype(np.float32))  # Convert to float32 tensor
                # print("averaged_tensor.shape: ", averaged_tensor.shape)
                averaged_tensor = averaged_tensor.permute(0, 3, 1, 2)

                averaged_tensor = averaged_tensor.repeat(1, 3, 1, 1)
                
                val_preds_batch = model_test_t1(averaged_tensor).squeeze()
                if isinstance(val_preds_batch, np.ndarray):
                    val_preds_batch = torch.tensor(val_preds_batch)
                val_preds.append(val_preds_batch)  # Store as tensors
                # print("val_preds: ", val_preds)

                # val_preds = [torch.tensor(pred) if isinstance(pred, np.ndarray) else pred for pred in val_preds]
                val_preds = [torch.tensor(pred) if isinstance(pred, (list, np.ndarray)) else pred for pred in val_preds]

                # Concatenate the list of tensors
                val_preds = torch.cat([pred.unsqueeze(0) if pred.dim() == 0 else pred for pred in val_preds]).cpu().numpy()
                # val_preds = [pred.unsqueeze(0) for pred in val_preds]
            #     val_preds = [pred.unsqueeze(0) if pred.dim() == 0 else pred for pred in val_preds]

            # # Convert predictions and labels to numpy arrays after the loop
            #     val_preds = torch.cat(val_preds).cpu().numpy()  # Concatenate and move to CPU
                val_preds_class = (val_preds > pred_tresh).astype(int)
            
            #SVM_b                   
                if histogram_vector is not None:
                    histogram_vector= np.array(histogram_vector)
                    reduced_histograms_test = pca_t1.transform(histogram_vector.reshape(1, -1))

                outputs = model_test_T1.predict(reduced_histograms_test)
            #SVM_e                   
                
                # all_preds.append(max(val_preds_class,outputs,preds, predictions))
                # all_preds.append(predictions)
                all_preds.append(1 if (predictions + preds + outputs + val_preds_class + cnn_pred) >= thresh else 0)
 



    # Convert lists to numpy arrays for metric calculations
    all_preds = np.array(all_preds)
    ###################

    
    
    series_instance_uid_list = os.listdir(test_folder_proc)
    # print("len(series_instance_uid_list) = ", len(series_instance_uid_list))
    # print("series_instance_uid_list = ", series_instance_uid_list)
    series_instance_uid_list = [name.split('_', 1)[1] for name in series_instance_uid_list]

    # print(series_instance_uid_list)

    # # series_instance_uid_list = os.listdir(test_folder_proc)
    # print("series_instance_uid_list = ", len(series_instance_uid_list))
    # print("all_preds = ", len(all_preds))
    all_preds = np.array(all_preds).flatten()

    predictions_df = pd.DataFrame(
                {
                    "SeriesInstanceUID": series_instance_uid_list,
                    "prediction": all_preds,
                }
            )
    predictions_df.to_csv(predictions_file_path, index=False)
    
if __name__ == "__main__":
    main()


