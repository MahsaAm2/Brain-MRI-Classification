# Brain MRI Classification: Normal vs. Abnormal

This project classifies brain MRI images into two categories: normal and abnormal. The dataset, sourced from the [iAAA MRI Challenge](https://github.com/iAAA-event/iAAA-MRI-Challenge), consists of 3,132 MRI scans from 1,044 patients, including T1-weighted spin-echo (T1W_SE), T2-weighted turbo spin-echo (T2W_TSE), and T2-weighted FLAIR (T2W_FLAIR) images. Each patient has between 16 to 20 MRI slices, with conditions such as tumors, Alzheimer's, and atrophy represented in the dataset.

## Preprocessing Steps

The following preprocessing steps were applied to the MRI data to prepare it for classification:

1. **Organize Data by Label**: Split the dataset into `normal` and `abnormal` folders based on patient IDs in the DICOM files, using labels from the `train.csv` file.
2. **Separate Series**: Divide images into `T1W_SE`, `T2W_TSE`, and `T2W_FLAIR` folders based on the `ProtocolName` field in the DICOM metadata.
3. **Exclude Sagittal Images**: Delete data where the `slice_orientation` is sagittal to maintain consistent orientation across images.
4. **Sort Slices by Instance Number**: Order slices for each scan based on the `InstanceNumber` in the DICOM file to ensure sequential stacking.
5. **Check Image Orientation**: Identify left-to-right orientation by examining the first element in `ImageOrientation`. If negative, apply a flip to standardize orientation.
6. **Crop Background**: Remove unnecessary background pixels to focus on the brain region.
7. **Resize Images**: Resize images to a consistent resolution for model input.
8. **Save as NumPy Arrays**: Convert and save processed images as NumPy arrays for efficient loading during training.

## Approach

After preprocessing, various deep learning and machine learning models were trained to classify the MRI images, with a focus on achieving optimal accuracy in distinguishing between normal and abnormal cases.
