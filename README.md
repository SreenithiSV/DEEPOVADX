# DEEPOVADX
## TEAM NAME: LeafLex
## TEAM MEMBERS: Shriyadithya Nair, Sree Nithi S V
## PROJECT TITLE: Revolutionizing Cancer Diagnosis: A Deep Learning Approach for Accurate Histopathological Image Classification.
## The project involves using deep learning, specifically Convolutional Neural Networks (CNNs), for the classification of cancer types based on histopathological images. The project includes image processing steps, resizing images, and implementing a CNN model. It explores different architectures, such as a custom CNN and a transfer learning approach using MobileNet. The models are trained, evaluated, and tested on a dataset, and their performance is assessed using metrics like accuracy and confusion matrices. Additionally, data augmentation techniques are applied to enhance model generalization. The project concludes with making predictions on new images and visualizing the results.
# USABILITY
- Medical Diagnosis - Cancer Classification
- Healthcare Automation - Automated Screening
- Research Support - Data Analysis
- Educational Tool - Training and Education
- Clinical Decision Support - Assisting Pathologists
- Future Enhancements - Integration with Healthcare Systems
# DEEPOVADX
The provided code is a comprehensive deep learning project named DEEPOVADX, primarily focused on classifying ovarian cancer subtypes from histopathological images. It involves image processing, CNN model training, and evaluation. The code includes image preprocessing, model creation (both custom CNN and transfer learning with MobileNet), and performance analysis using metrics like accuracy and confusion matrices. It also demonstrates data augmentation and provides a function for making predictions on new images. The project aims to assist medical professionals in cancer diagnosis through automated screening.

#STEPS
Certainly, let's condense the steps:

### 1. Setup:
   - Clone the GitHub repository.
   - Install Python and dependencies from `requirements.txt`.

### 2. Dataset:
   - Download Kaggle UBC dataset.
   - Organize images by cancer types in the `Cancer_Data` folder.

### 3. Model:
   - Use Jupyter notebooks in `Model_Training`.
   - Train the deep learning model.

### 4. Predictions:
   - Use the `pred_and_plot` function to make predictions.
   - Adjust file paths in the notebook if needed.

These steps should provide a more concise overview of the process.

## Overview

- DEEPOVADX is a deep learning project focused on classifying ovarian cancer subtypes based on histopathological images. The goal is to assist medical professionals in accurate and efficient cancer diagnosis.

## Project Structure

The project is organized into the following key components:
- `Image_Processing`: Contains scripts for image preprocessing and resizing based on specific keywords.
- `Data_Processing`: Encompasses data processing steps, including filtering and directory creation.
- `Model_Training`: Involves the training and evaluation of convolutional neural networks for subtype classification.
- `Results`: Stores visualizations, classification reports, and other outcome-related files.

## Dependencies

- TensorFlow
- Keras
- NumPy
- Pandas
- OpenCV
- Matplotlib
- Seaborn
- Scikit-learn

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

## Acknowledgments

- We acknowledge the UBC-OCEAN dataset and express gratitude to the TensorFlow and Keras communities for their powerful deep learning tools.

## Author

- Shriyadithya Nair
- Sree Nithi S V

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.



## Image Processing and Deep Learning Setup


````python
import os
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from skimage import io
from skimage.transform import rescale, resize

import matplotlib.pyplot as plt
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Conv2D, Activation, Concatenate, MaxPooling2D, GlobalAveragePooling2D, Dense
from tensorflow.keras.preprocessing import image
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
layers =  tf.keras.layers
models = tf.keras.models
````
Library Imports:
- import os: Operating system-related functions.  
- import numpy as np: NumPy for numerical operations.  
- import pandas as pd: Pandas for data manipulation.  
- from tqdm.auto import tqdm: TQDM for creating progress bars.  
- from skimage import io, from skimage.transform import rescale, resize: Image-related functions from scikit-image.  
- import matplotlib.pyplot as plt: Matplotlib for plotting.  
- from future import absolute_import, division, print_function: Enabling compatibility with future versions of Python.  

Deep Learning Libraries

## Image Processing: Resizing Images Based on Keyword

### Explanation with Print Statements

```python
# Library Imports:
import cv2  # OpenCV for image processing.
import os  # Operating system-related functions.
import pandas as pd  # Pandas for data manipulation.

# Function Definition:
def process_images_with_keyword(keyword):
    # Reading CSV Data:
    df = pd.read_csv('/kaggle/input/UBC-OCEAN/train.csv')  # Reads the CSV file into a Pandas DataFrame.

    # Filtering Data Based on Keyword:
    comparison_result = df['label'].str.contains(keyword, case=False)  # Compares the text in the 'label' column with the provided keyword.
    filtered_data = df[comparison_result]  # Uses the comparison result to filter the DataFrame.

    # Image Processing and Resizing:
    # Iterates through the filtered image data, loads each image, resizes it to (224, 224), and saves the resized image in a new directory.
    os.makedirs(destination_directory, exist_ok=True)  # Creates the destination directory if it doesn't exist.

    # Print Statements (Optional):
    # Uncomment the print statements if you want to display the paths for reference.
    for image_id in image_ids:
        source_path = os.path.join(source_directory, f"{image_id}_thumbnail.png")
        destination_path = os.path.join(destination_directory, f"{image_id}.png")

        # Load the image
        image = cv2.imread(source_path)

        if image is not None:
            # Resize the image
            resized_image = cv2.resize(image, (224, 224))

            # Save the resized image
            cv2.imwrite(destination_path, resized_image)

            # Print the paths for reference
            print("Source Path:", source_path)
            print("Destination Path:", destination_path)
```


## Image Processing: Resizing Images Based on Keyword


````python
import cv2
import os
import pandas as pd

def process_images_with_keyword(keyword):
    # Read the CSV file into the DataFrame 'df'
    df = pd.read_csv('/kaggle/input/UBC-OCEAN/train.csv')

    # Compare the text in the column with the provided keyword
    comparison_result = df['label'].str.contains(keyword, case=False)

    # Use the comparison result
    filtered_data = df[comparison_result]

    # Get the "image_id" column and convert it to strings
    image_ids = filtered_data['image_id'].astype(str)

    # Define the source and destination directories
    source_directory = '/kaggle/input/UBC-OCEAN/train_thumbnails/'
    destination_directory = f'/kaggle/working/Cancer_Data/{keyword}/'

    # Create the destination directory if it doesn't exist
    os.makedirs(destination_directory, exist_ok=True)

    # Iterate through image_ids, read images, and save resized images
    for image_id in image_ids:
        source_path = os.path.join(source_directory, f"{image_id}_thumbnail.png")
        destination_path = os.path.join(destination_directory, f"{image_id}.png")

        # Load the image
        image = cv2.imread(source_path)

        if image is not None:
            # Resize the image
            resized_image = cv2.resize(image, (224, 224))

            # Save the resized image
            cv2.imwrite(destination_path, resized_image)

        # Print the paths for reference
        # print("Source Path:", source_path)
        # print("Destination Path:", destination_path)
````
```python
from pathlib import Path

Path('/kaggle/working/Cancer_Data').mkdir(parents=True, exist_ok=True)
Path('/kaggle/working/Cancer_Data/HGSC').mkdir(parents=True, exist_ok=True)
Path('/kaggle/working/Cancer_Data/EC').mkdir(parents=True, exist_ok=True)
Path('/kaggle/working/Cancer_Data/CC').mkdir(parents=True, exist_ok=True)
Path('/kaggle/working/Cancer_Data/MC').mkdir(parents=True, exist_ok=True)
Path('/kaggle/working/Cancer_Data/LGSC').mkdir(parents=True, exist_ok=True)
```
````python
import cv2
import os

def process_images_with_keyword(keyword):
    # Read the CSV file into the DataFrame 'df'
    df = pd.read_csv('/kaggle/input/UBC-OCEAN/train.csv')

    # Compare the text in the 'label' column with the provided keyword
    comparison_result = df['label'].str.contains(keyword, case=False)

    # Use the comparison result to filter the DataFrame
    filtered_data = df[comparison_result]

    # Get the "image_id" column and convert it to strings
    image_ids = filtered_data['image_id'].astype(str)

    # Define the source and destination directories
    source_directory = '/kaggle/input/UBC-OCEAN/train_thumbnails/'
    destination_directory = f'/kaggle/working/Cancer_Data/{keyword}/'

    # Create the destination directory if it doesn't exist
    os.makedirs(destination_directory, exist_ok=True)

    # Iterate through image_ids, read images, and save resized images
    for image_id in image_ids:
        source_path = os.path.join(source_directory, f"{image_id}_thumbnail.png")
        destination_path = os.path.join(destination_directory, f"{image_id}.png")
        
        # Load the image using OpenCV
        image = cv2.imread(source_path)
        
        # Check if the image is not None
        if image is not None:
            # Resize the image to (224, 224)
            resized_image = cv2.resize(image, (224, 224))
            
            # Save the resized image
            cv2.imwrite(destination_path, resized_image)
        
        # Uncomment the following lines to print the paths for reference
        # print("Source Path:", source_path)
        # print("Destination Path:", destination_path)
````
````python
process_images_with_keyword("CC")
process_images_with_keyword("EC")
process_images_with_keyword("HGSC")
process_images_with_keyword("LGSC")
process_images_with_keyword("MC")
````
```python
image_set = '/kaggle/working/Cancer_Data/'

# Initialize empty lists to store filepaths and labels
filepaths = []
labels = []

# List all subdirectories in the given image_set directory
classlist = os.listdir(image_set)

# Iterate through each subdirectory (class)
for klass in classlist:
    classpath = os.path.join(image_set, klass)
    
    # Check if it's a directory
    if os.path.isdir(classpath):
        # List all files in the subdirectory
        flist = os.listdir(classpath)
        
        # Iterate through each file
        for f in flist:
            fpath = os.path.join(classpath, f)
            
            # Append file path and label to the lists
            filepaths.append(fpath)
            labels.append(klass)

# Create Pandas Series for filepaths and labels
Fseries = pd.Series(filepaths, name='filepaths')
Lseries = pd.Series(labels, name='labels')

# Concatenate the series to create a DataFrame 'lung_df'
lung_df = pd.concat([Fseries, Lseries], axis=1)

# Concatenate 'lung_df' into the final DataFrame 'df'
df = pd.concat([lung_df], axis=0).reset_index(drop=True)

# Print the counts of each label (class)
print(df['labels'].value_counts())
```
- The code iterates through each subdirectory (class) within the specified image_set directory.
- For each class, it lists all the files in the subdirectory and creates file paths by joining the class path with the file name.
- File paths and corresponding labels (class names) are appended to separate lists (filepaths and labels).
- Two Pandas Series are created from the lists, and then concatenated to form the DataFrame lung_df.
- The final DataFrame df is created by concatenating lung_df along the rows.
- The last line prints the counts of each label (class) in the DataFrame.

![FIRST]("C:\Users\SREENITHI\productathon\Screenshot 2024-02-04 024552.png")
```python
from sklearn.model_selection import train_test_split

# Define the proportions for training, testing, and validation sets
train_split = 0.5
test_split = 0.25
dummy_split = test_split / (1 - train_split)

# Split the data into training, testing, and validation sets
train_df, dummy_df = train_test_split(df, train_size=train_split, shuffle=True, random_state=123)
test_df, valid_df = train_test_split(dummy_df, train_size=dummy_split, shuffle=True, random_state=123)

# Print the lengths of the resulting dataframes
print('train_df length: ', len(train_df), ' _test_df length: ', len(test_df), '  valid_df length: ', len(valid_df))
```
## Importing Library:

- from sklearn.model_selection import train_test_split: This line imports the train_test_split function from the sklearn.model_selection module.
# Defining Proportions:

- train_split = 0.5: Sets the proportion of data to be used for training to 50%.
- test_split = 0.25: Sets the proportion of data to be used for testing to 25%.
- dummy_split = test_split / (1 - train_split): Calculates the proportion of data to be used for validation.
# Data Splitting:

- train_df, dummy_df = train_test_split(df, train_size=train_split, shuffle=True, random_state=123): Splits the original DataFrame (df) into training (train_df) and a dummy DataFrame (dummy_df).
- test_df, valid_df = train_test_split(dummy_df, train_size=dummy_split, shuffle=True, random_state=123): Further splits the dummy DataFrame into testing (test_df) and validation (valid_df).
# Print Lengths:

- print('train_df length: ', len(train_df), ' _test_df length: ', len(test_df), ' valid_df length: ', len(valid_df)): Prints the lengths of the resulting dataframes for reference.
![SECOND]("C:\Users\SREENITHI\productathon\2.png")

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define image parameters
height = 224
width = 224
channels = 3
batch_size = 8
img_shape = (height, width, channels)
img_size = (height, width)

# Calculate test batch size and steps
length = len(test_df)
test_batch_size = sorted([int(length / n) for n in range(1, length + 1) if length % n == 0 and length / n <= 80], reverse=True)[0]
test_steps = int(length / test_batch_size)

# Print test batch size and steps
print('test batch size: ', test_batch_size, '  test steps: ', test_steps)

# Define image preprocessing function
def scalar(img):
    return img / 127.5 - 1  # scale pixel between -1 and +1

# Create ImageDataGenerators for training, testing, and validation sets
gen = ImageDataGenerator(preprocessing_function=scalar)
train_set = gen.flow_from_dataframe(train_df, x_col='filepaths', y_col='labels', target_size=img_size,
                                    class_mode='categorical', color_mode='rgb', shuffle=True, batch_size=batch_size)
test_set = gen.flow_from_dataframe(test_df, x_col='filepaths', y_col='labels', target_size=img_size,
                                   class_mode='categorical', color_mode='rgb', shuffle=False, batch_size=test_batch_size)
validate_set = gen.flow_from_dataframe(valid_df, x_col='filepaths', y_col='labels', target_size=img_size,
                                       class_mode='categorical', color_mode='rgb', shuffle=True, batch_size=batch_size)
```
## Image Parameters:

- height, width, channels: Define the dimensions and channels of the input images.
- batch_size: Define the batch size for training.
## Test Batch Size and Steps:

- Calculate the appropriate test batch size and steps based on the length of the test dataset.
## Image Preprocessing Function:

- scalar(img): Define a function to scale pixel values between -1 and +1.
## ImageDataGenerators:

- Create ImageDataGenerators for training (train_set), testing (test_set), and validation (validate_set) sets.
- The generators use the defined preprocessing function and settings for batch size, image size, color mode, etc.V

![Third]("C:\Users\SREENITHI\productathon\3.png")

```python
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation

# Create a Sequential model
model = keras.models.Sequential()

# Add Convolutional layers with MaxPooling and Dropout
model.add(keras.layers.Conv2D(32, 3, activation='relu', input_shape=(224, 224, 3)))
model.add(keras.layers.Dropout(0.1))
model.add(keras.layers.MaxPooling2D())

model.add(keras.layers.Conv2D(64, 3, activation='relu'))
model.add(keras.layers.Dropout(0.15))
model.add(keras.layers.MaxPooling2D())

model.add(keras.layers.Conv2D(128, 3, activation='relu'))
model.add(keras.layers.Dropout(0.15))
model.add(keras.layers.MaxPooling2D())

# Flatten the output and add Dense layers
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(256, activation='relu'))
model.add(keras.layers.Dense(5, activation='softmax'))

# Compile the model with Adam optimizer and categorical crossentropy loss
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Execute the model on training data
history = model.fit(train_set, validation_data=validate_set, epochs=30, verbose=1)
```
![fourth]("C:\Users\SREENITHI\productathon\4.png")

```python
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
Y_pred = model.predict(test_set)
y_pred = np.argmax(Y_pred ,axis =1)

preds = model.predict(test_set,verbose=1)
predictions = preds.copy()
predictions[predictions <= 0.5] = 0
predictions[predictions > 0.5] = 1

print('Classification Report')
target_names = ['CC','EC','HGSC','LGSC','MC']
print(classification_report(test_set.classes, y_pred, target_names=target_names))

from sklearn.metrics import classification_report,confusion_matrix
cm = pd.DataFrame(data=confusion_matrix( y_true= test_set.classes, y_pred= y_pred, labels=[0, 1,2,3,4]), index=['Actual CC','Actual EC','Actual HGSC','Actual LGSC', 'Actual MC'],columns=['Predicted CC','Predicted EC','Predicted HGSC','Predicted LGSC', 'Predicted MC'])
import seaborn as sns
sns.heatmap(cm,annot=True,fmt="d",cmap="YlGn")
```
```python
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

# Predictions on the test set
Y_pred = model.predict(test_set)
y_pred = np.argmax(Y_pred, axis=1)

# Threshold predictions for binary classification
preds = model.predict(test_set, verbose=1)
predictions = preds.copy()
predictions[predictions <= 0.5] = 0
predictions[predictions > 0.5] = 1

# Print Classification Report
print('Classification Report')
target_names = ['CC', 'EC', 'HGSC', 'LGSC', 'MC']
print(classification_report(test_set.classes, y_pred, target_names=target_names))

# Create and Display Confusion Matrix
from sklearn.metrics import classification_report, confusion_matrix
cm = confusion_matrix(y_true=test_set.classes, y_pred=y_pred, labels=[0, 1, 2, 3, 4])
cm_df = pd.DataFrame(data=cm, index=['Actual CC', 'Actual EC', 'Actual HGSC', 'Actual LGSC', 'Actual MC'],
                      columns=['Predicted CC', 'Predicted EC', 'Predicted HGSC', 'Predicted LGSC', 'Predicted MC'])

# Display the Confusion Matrix using seaborn heatmap
import seaborn as sns
sns.heatmap(cm_df, annot=True, fmt="d", cmap="YlGn")
```

![six]("C:\Users\SREENITHI\productathon\6.png")

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define image dimensions and batch size
height = 224
width = 224
channels = 3
batch_size = 8
img_shape = (height, width, channels)
img_size = (height, width)
length = len(test_df)

# Determine optimal test batch size and steps
test_batch_size = sorted([int(length / n) for n in range(1, length + 1) if length % n == 0 and length / n <= 80], reverse=True)[0]
test_steps = int(length / test_batch_size)
print('test batch size:', test_batch_size, '  test steps:', test_steps)

# Define the augmentation function
def augment(image):
    # Apply augmentation techniques using ImageDataGenerator
    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        preprocessing_function=scalar
    )
    return datagen.random_transform(image)

# Define the scalar function
def scalar(img):
    return img / 127.5 - 1  # scale pixel values between -1 and +1

# Create ImageDataGenerators for training, testing, and validation sets
gen = ImageDataGenerator(preprocessing_function=augment)

train_set = gen.flow_from_dataframe(train_df, x_col='filepaths', y_col='labels', target_size=img_size,
                                    class_mode='categorical', color_mode='rgb', shuffle=True, batch_size=batch_size)

test_set = gen.flow_from_dataframe(test_df, x_col='filepaths', y_col='labels', target_size=img_size,
                                   class_mode='categorical', color_mode='rgb', shuffle=False, batch_size=test_batch_size)

validate_set = gen.flow_from_dataframe(valid_df, x_col='filepaths', y_col='labels', target_size=img_size,
                                       class_mode='categorical', color_mode='rgb', shuffle=True, batch_size=batch_size)
```
```python
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD

# Create a Sequential model
model = keras.models.Sequential()

# Add Convolutional layers with max-pooling and dropout
model.add(keras.layers.Conv2D(32, 3, activation='relu', input_shape=(224, 224, 3)))
model.add(keras.layers.Dropout(0.1))
model.add(keras.layers.MaxPooling2D())

model.add(keras.layers.Conv2D(64, 3, activation='relu'))
model.add(keras.layers.Dropout(0.15))
model.add(keras.layers.MaxPooling2D())

model.add(keras.layers.Conv2D(128, 3, activation='relu'))
model.add(keras.layers.Dropout(0.15))
model.add(keras.layers.MaxPooling2D())

# Flatten the output and add Dense layers
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(256, activation='relu'))
model.add(keras.layers.Dense(5, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Compile the model again (repeated line)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Execute the model training
history = model.fit(train_set, validation_data=validate_set, epochs=30, verbose=1)
```
```python
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

# Predictions using the model on the test set
Y_pred = model.predict(test_set)
y_pred = np.argmax(Y_pred, axis=1)

# Additional predictions for threshold-based classification
preds = model.predict(test_set, verbose=1)
predictions = preds.copy()
predictions[predictions <= 0.5] = 0
predictions[predictions > 0.5] = 1

# Display Classification Report
print('Classification Report')
target_names = ['CC', 'EC', 'HGSC', 'LGSC', 'MC']
print(classification_report(test_set.classes, y_pred, target_names=target_names))

# Display Confusion Matrix Heatmap
from sklearn.metrics import classification_report, confusion_matrix
cm = pd.DataFrame(data=confusion_matrix(y_true=test_set.classes, y_pred=y_pred, labels=[0, 1, 2, 3, 4]),
                  index=['Actual CC', 'Actual EC', 'Actual HGSC', 'Actual LGSC', 'Actual MC'],
                  columns=['Predicted CC', 'Predicted EC', 'Predicted HGSC', 'Predicted LGSC', 'Predicted MC'])

import seaborn as sns
sns.heatmap(cm, annot=True, fmt="d", cmap="YlGn")
```
```python
from keras.applications import MobileNet
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, Input

# Define the input size for MobileNet
img_size = [224, 224]

# Load the MobileNet model with pre-trained weights
MobileNet = MobileNet(input_shape=img_size + [3], weights='imagenet', include_top=False)

# Freeze existing weights to keep them unchanged during training
for layer in MobileNet.layers:
    layer.trainable = False

# Flatten the output of MobileNet
flatten = Flatten()(MobileNet.output)

# Add dense layers for classification
dense = Dense(256, activation='relu')(flatten)
dense = Dense(128, activation='relu')(dense)
prediction = Dense(5, activation='softmax')(dense)

# Create a new model using MobileNet as the base and the added dense layers for classification
model_2 = Model(inputs=MobileNet.input, outputs=prediction)
model_2.summary()

# Compile the model
model_2.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
history = model_2.fit(train_set, validation_data=(validate_set), epochs=50, verbose=1)

# Make predictions on the test set
Y_pred = model_2.predict(test_set)
y_pred = np.argmax(Y_pred, axis=1)

# Additional predictions for threshold-based classification
preds = model_2.predict(test_set, verbose=1)
predictions = preds.copy()
predictions[predictions <= 0.5] = 0
predictions[predictions > 0.5] = 1

# Display Classification Report
print('Classification Report')
target_names = ['CC', 'EC', 'HGSC', 'LGSC', 'MC']
print(classification_report(test_set.classes, y_pred, target_names=target_names))

# Display Confusion Matrix Heatmap
from sklearn.metrics import classification_report, confusion_matrix
cm = pd.DataFrame(data=confusion_matrix(y_true=test_set.classes, y_pred=y_pred, labels=[0, 1, 2, 3, 4]),
                  index=['Actual CC', 'Actual EC', 'Actual HGSC', 'Actual LGSC', 'Actual MC'],
                  columns=['Predicted CC', 'Predicted EC', 'Predicted HGSC', 'Predicted LGSC', 'Predicted MC'])

import seaborn as sns
sns.heatmap(cm, annot=True, fmt="d", cmap="YlGn")
```
```python
from keras.applications import MobileNet
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, Input

# Define the input size for MobileNet
img_size = [224, 224]

# Load the MobileNet model with pre-trained weights
MobileNet = MobileNet(input_shape=img_size + [3], weights='imagenet', include_top=False)

# Freeze existing weights to keep them unchanged during training
for layer in MobileNet.layers:
    layer.trainable = False

# Flatten the output of MobileNet
flatten = Flatten()(MobileNet.output)

# Add dense layers for classification
dense = Dense(256, activation='relu')(flatten)
dense = Dense(128, activation='relu')(dense)
prediction = Dense(5, activation='softmax')(dense)

# Create a new model using MobileNet as the base and the added dense layers for classification
model_2 = Model(inputs=MobileNet.input, outputs=prediction)
model_2.summary()

# Compile the model
model_2.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
history = model_2.fit(train_set, validation_data=(validate_set), epochs=50, verbose=1)

# Make predictions on the test set
Y_pred = model_2.predict(test_set)
y_pred = np.argmax(Y_pred, axis=1)

# Additional predictions for threshold-based classification
preds = model_2.predict(test_set, verbose=1)
predictions = preds.copy()
predictions[predictions <= 0.5] = 0
predictions[predictions > 0.5] = 1

# Display Classification Report
print('Classification Report')
target_names = ['CC', 'EC', 'HGSC', 'LGSC', 'MC']
print(classification_report(test_set.classes, y_pred, target_names=target_names))

# Display Confusion Matrix Heatmap
from sklearn.metrics import classification_report, confusion_matrix
cm = pd.DataFrame(data=confusion_matrix(y_true=test_set.classes, y_pred=y_pred, labels=[0, 1, 2, 3, 4]),
                  index=['Actual CC', 'Actual EC', 'Actual HGSC', 'Actual LGSC', 'Actual MC'],
                  columns=['Predicted CC', 'Predicted EC', 'Predicted HGSC', 'Predicted LGSC', 'Predicted MC'])

import seaborn as sns
sns.heatmap(cm, annot=True, fmt="d", cmap="YlGn")

# Create a function to import an image and resize it
def load_and_prep_image(filename, img_shape=224):
    # Read in the target file (an image)
    img = tf.io.read_file(filename)
    
    # Decode the read file into a tensor & ensure 3 colour channels
    img = tf.image.decode_image(img, channels=3)
    
    # Resize the image
    img = tf.image.resize(img, size=[img_shape, img_shape])
    
    # Rescale the image
    img = img / 255.
    return img

def pred_and_plot(model, filename, target_names):
    """
    Imports an image located at filename, makes a prediction on it with
    a trained model and plots the image with the predicted class as the title.
    """
    # Import the target image and preprocess it
    img = load_and_prep_image(filename)
    
    # Make a prediction
    pred = model.predict(tf.expand_dims(img, axis=0))
    
    # Get the predicted class
    pred_class = target_names[int(tf.round(pred)[0][0])]
    
    # Plot the image and predicted class
    plt.imshow(img)
    plt.title(f"Prediction: {pred_class}")
    plt.axis(False)
```
```python
# Test our model on a test image
pred_and_plot(model_2, "/kaggle/working/Cancer_Data/CC/45725.png", target_names)
```