# Brain MRI Classification: Normal vs. Abnormal

This project classifies brain MRI images into two categories: normal and abnormal. The dataset, sourced from the [iAAA MRI Challenge](https://github.com/iAAA-event/iAAA-MRI-Challenge), consists of 3,132 MRI scans from 1,044 patients, including T1-weighted spin-echo (T1W_SE), T2-weighted turbo spin-echo (T2W_TSE), and T2-weighted FLAIR (T2W_FLAIR) images. Each patient has between 16 to 20 MRI slices, with conditions such as tumors, Alzheimer's, and atrophy represented in the dataset.

## Sample MRI Scans

### Normal MRI Examples
| ![n1](https://github.com/user-attachments/assets/7eadc09d-891f-4a70-90d0-3a7c2a40b7a7) | ![n2](https://github.com/user-attachments/assets/60cd1cc9-fe92-4f4a-ac24-93fe5fc2f31c) | ![n3](https://github.com/user-attachments/assets/1dc284c1-841b-4a62-8d5f-63e700389fa7) |
| --- | --- | --- |

### Abnormal MRI Examples
| ![ab](https://github.com/user-attachments/assets/518dbee5-0704-4dd4-b485-3483cb820640) | ![ab2](https://github.com/user-attachments/assets/c3955755-a340-4b3e-8b51-3954112c593a) | ![abb](https://github.com/user-attachments/assets/c7513e13-9798-4e62-8e43-9a03e7167412) |
| --- | --- | --- |


## Preprocessing Steps

The following preprocessing steps were applied to the MRI data to prepare it for classification:

1. **Organize Data by Label**: Split the dataset into `normal` and `abnormal` folders based on patient IDs in DICOM files, using labels from `train.csv`.
2. **Separate Series**: Divide images into T1W_SE, T2W_TSE, and T2W_FLAIR folders based on the `ProtocolName` field in the DICOM metadata.
3. **Exclude Sagittal Images**: Delete data where `slice_orientation` is sagittal to maintain consistent orientation across images.
4. **Sort Slices by Instance Number**: Order slices for each scan based on the `InstanceNumber` in the DICOM file to ensure sequential stacking.
5. **Check Image Orientation**: Identify left-to-right orientation by examining the first element in `ImageOrientation`. If negative, apply a flip to standardize orientation.
6. **Crop Background**: Remove unnecessary background pixels to focus on the brain region.
7. **Resize Images**: Resize images to a consistent resolution for model input.
8. **Save as NumPy Arrays**: Convert and save processed images as NumPy arrays for efficient loading during training.

## Handling Imbalanced Dataset

The dataset exhibited a significant class imbalance, with approximately 87% of the samples classified as normal. To address this issue, we implemented the following strategies:

1. **Weighted Sampling**: We used a sampler that assigns weights to each class during training, allowing the model to place greater emphasis on the minority class (abnormal samples).

2. **Loss Function Adjustment**: The weights from the sampler were incorporated into the loss function to ensure that misclassifications of the abnormal class carried more penalty, further enhancing the model's focus on learning from the minority class.

3. **Data Augmentation**: We applied various data augmentation techniques to artificially expand the dataset, including transformations such as rotation, flipping, and scaling for abnormal samples. This helped improve model robustness.

These methods were instrumental in enhancing the model's ability to generalize and effectively classify both normal and abnormal cases.

## Approaches

### First Approach: CNN Model

In the first approach, a Convolutional Neural Network (CNN) was used for classification. After training the CNN, thresholds were established to determine abnormality based on the number of slices. If the number of slices for a patient met the threshold, the patient was labeled as abnormal.

### Second Approach: Histogram and MLP Classification

In a second approach, 16 slices were selected from the beginning of the available slices for each patient. These slices were then divided into left and right hemispheres, allowing for the computation of two histograms for each slice—one for the right and one for the left. This resulted in a feature set with a size of 
2 x 16 x 255, where:
- **2** represents the left and right hemispheres,
- **16** is the number of selected slices,
- **255** corresponds to the number of intensity levels for the histogram.

A multi-layer perceptron (MLP) was used to classify these features based on the idea that abnormalities can affect the symmetry of brain structures, which is indicative of various diseases.


### Third Approach: SVM with Dimensionality Reduction

In the third approach, 16 slices were selected from the beginning of the available slices for each patient, yielding a feature set of size 16 x 255. Principal Component Analysis (PCA) was employed to reduce dimensionality before passing the features to a Support Vector Machine (SVM) model, where optimal kernel and regularization parameters were determined for effective classification.

### Fourth Approach: ResNet with Transfer Learning

For the fourth approach, a pre-trained ResNet50 model was utilized with transfer learning. The following steps were taken:

- **Loading Pre-trained Weights**: The model weights were loaded from a saved state.
- **Freezing Layers**: Initially, all layers were frozen to retain the learned features from the pre-trained model.
- **Custom ResNet Architecture**: A custom architecture was implemented, modifying the final fully connected layers to adapt to the binary classification task. The output from the ResNet base model was flattened and passed through several fully connected layers, with batch normalization and dropout layers included to prevent overfitting.

#### ResNet Custom Classes

- **CustomResNetF**: This class is tailored for FLAIR images with a modified architecture and an output of 1 neuron for binary classification.
- **CustomResNetT1**: Tailored for T1-weighted images with a different configuration.
- **CustomResNetT2**: Similar to CustomResNetF, but optimized for T2-weighted images.

The final activation function used in each model is a sigmoid function, suitable for binary classification tasks.


## Evaluation Metrics

For this challenge, the model's performance was assessed using three key evaluation metrics:

- **Precision**: Measures the accuracy of the positive predictions.
- **Recall**: Evaluates the model's ability to correctly identify positive cases.
- **AUC (Area Under the Curve)**: Represents the model's ability to distinguish between classes.

## Running the Models

To generate predictions using the saved models, place `submission.py` in the same directory as the saved model files. Then, run the following command:

```bash
python submission.py --data-dir /path/to/data-dir --predictions-file-path /path/to/submission.csv

