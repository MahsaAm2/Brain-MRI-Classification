# Brain MRI Classification: Normal vs. Abnormal

This project classifies brain MRI images into two categories: normal and abnormal. The dataset, sourced from the [iAAA MRI Challenge](https://github.com/iAAA-event/iAAA-MRI-Challenge), consists of 3,132 MRI scans from 1,044 patients, including T1-weighted spin-echo (T1W_SE), T2-weighted turbo spin-echo (T2W_TSE), and T2-weighted FLAIR (T2W_FLAIR) images. Each patient has between 16 to 20 MRI slices, with various brain conditions represented, including tumors, Alzheimer's, and atrophy.

## Preprocessing Steps

The following preprocessing steps were applied to the MRI data to prepare it for classification:

1. **Organize Data by Label**: Split the dataset into `normal` and `abnormal` folders based on patient IDs in the DICOM files, using labels from the `train.csv` file.
2. **Separate Series**: Divide images into `T1W_SE`, `T2W_TSE`, and `T2W_FLAIR` folders based on the `ProtocolName` field in the DICOM metadata.
3. **Exclude Sagittal Images**: Delete data where the `SliceOrientation` DICOM tag indicates sagittal orientation to maintain consistent orientation across images.
4. **Sort Slices by Instance Number**: Order slices for each scan based on the `InstanceNumber` in the DICOM file to ensure sequential stacking.
5. **Check Image Orientation**: Identify left-to-right orientation by examining the first element in `ImageOrientation`. If negative, apply a flip to standardize orientation.
6. **Crop Background**: Remove unnecessary background pixels to focus on the brain region.
7. **Resize Images**: Resize images to a consistent resolution for model input.
8. **Save as NumPy Arrays**: Convert and save processed images as NumPy arrays for efficient loading during training.

## CNN Model and Classification Approach

The CNN model for this project includes three convolutional blocks, each with convolutional and max-pooling layers, followed by fully connected layers. This structure enables the model to learn spatial features from MRI slices and classify each as normal or abnormal. The final layer outputs a probability score to support binary classification.

In the initial approach, the CNN model was trained on individual MRI slices to classify them as either normal or abnormal. After training, two thresholds were determined:

1. **CNN Prediction Threshold**: A threshold was set for the CNN model’s prediction confidence to label a slice as abnormal.
2. **Slice Count Threshold**: A threshold was established for the minimum number of abnormal slices within each patient’s scan required to classify the entire patient as abnormal.

Thus, while the model was trained on individual slices, a patient was labeled as abnormal if the number of slices labeled abnormal met or exceeded the slice count threshold.

## Second Approach: Histogram and MLP Classification

In a second approach, 16 slices were selected from the beginning of the available slices for each patient. These slices were then divided into left and right hemispheres, allowing for the computation of two histograms for each slice—one for the right and one for the left. This resulted in a feature set of size \(2 \times 16 \times 255\). A multi-layer perceptron (MLP) was used to classify these features based on the idea that abnormalities can affect the symmetry of brain structures, which is indicative of various diseases.

## Third Approach: SVM with PCA

In the third approach, 16 slices were selected from the beginning of each patient's available slices, resulting in a feature set of size \(16 \times 255\). Principal Component Analysis (PCA) was then applied to reduce the dimensionality of the feature set. The reduced features were fed into a Support Vector Machine (SVM) classifier, where various kernels and regularization parameters (C) were evaluated to find the optimal model for classifying the MRI slices as normal or abnormal.
