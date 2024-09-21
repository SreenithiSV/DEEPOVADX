# DEEPOVADX

The provided code is a comprehensive deep learning project named DEEPOVADX, primarily focused on classifying ovarian cancer subtypes from histopathological images. It involves image processing, CNN model training, and evaluation. The code includes image preprocessing, model creation (both custom CNN and transfer learning with MobileNet), and performance analysis using metrics like accuracy and confusion matrices. It also demonstrates data augmentation and provides a function for making predictions on new images. The project aims to assist medical professionals in cancer diagnosis through automated screening.

## Overview

- DEEPOVADX is a deep learning project focused on classifying ovarian cancer subtypes based on histopathological images. The goal is to assist medical professionals in accurate and efficient cancer diagnosis.

## Project Structure

The project is organized into the following key components:
- `Image_Processing`: Contains scripts for image preprocessing and resizing based on specific keywords.
- `Data_Processing`: Encompasses data processing steps, including filtering and directory creation.
- `Model_Training`: Involves the training and evaluation of convolutional neural networks for subtype classification.
- `Results`: Stores visualizations, classification reports, and other outcome-related files.

## Getting Started

The dependencies required to use the notebook are listed as follows:

- TensorFlow
- Keras
- NumPy
- Pandas
- OpenCV
- Matplotlib
- Seaborn
- Scikit-learn

Clone the repository and navigate to it by running the following command in your termninal:

```bash
git clone https://github.com/SreenithiSV/DEEPOVADX.git
cd DEEPOVADX
```

Install the required dependenices using the following command:


```bash
pip install -r requirements.txt
```

## Usage

- Ensure the required datasets are available and follow the provided Jupyter notebooks for data processing, model training, and evaluation.

## Image Processing and Deep Learning Setup

- The project leverages TensorFlow and Keras for deep learning tasks. Image preprocessing involves resizing images based on specific keywords using OpenCV.

## Data Processing

- CSV data is processed, filtered based on keywords, and images are resized for further analysis. Directory structures are created to organize the processed data.

## Model Architecture

- The convolutional neural network (CNN) architecture comprises multiple layers of convolution, pooling, and dense layers. The model is designed for accurate ovarian cancer subtype classification.

## Training

- The model is trained using the Adam optimizer and categorical crossentropy loss function. Training details, including epochs and batch sizes, are specified in the training script.

## Evaluation

- Model performance is evaluated using classification reports, confusion matrices, and other relevant metrics. Visualization tools such as seaborn are employed for result analysis.

## Results

- The project's results include accuracy scores, classification reports, and visualizations. These outcomes offer insights into the model's effectiveness in cancer subtype classification.

## Future Enhancements

- Future enhancements may include incorporating more advanced architectures, exploring transfer learning, and collaborating with medical professionals for additional dataset annotations.
