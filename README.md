# 🤟🏻 Bisindo Sign Language using Deep Learning 🤟🏻
This project focuses on the classification of Indonesian Sign Language (BISINDO) alphabet images using Deep Learning techniques.

## Problem and Goal
The primary problem addressed by this project is the accurate classification of images representing the Indonesian Sign Language (BISINDO) alphabets. The main goal is to develop a robust Deep Learning model capable of identifying and categorizing these sign language gestures from image inputs.

## Key Insights
The provided Jupyter Notebook primarily focuses on the implementation of the deep learning pipeline without explicit exploratory data analysis sections. Therefore, the notebook does not contain detailed key insights derived from data exploration or visualizations of the dataset's characteristics (e.g., class distribution imbalances, image variations within classes).

## Analytical Workflow
The project's analytical workflow involves several key stages:
- Data Acquisition: The BISINDO alphabet dataset is downloaded from [Kaggle](https://www.kaggle.com/datasets/achmadnoer/alfabet-bisindo/data).
- Data Preparation:
  - Raw images are organized and combined into a unified directory structure.
  - The dataset is then split into training, validation, and testing sets using `train_test_split` with stratification to ensure a balanced distribution of classes across all sets.
- Data Augmentation: ImageDataGenerator is extensively used to augment the training data, introducing variations such as rotation, zoom, width/height shifts, shear, and horizontal flips. This helps in increasing the dataset's diversity and improving model generalization.
- Model Building: A Deep Learning model is constructed, leveraging transfer learning.
- Model Training: The model is trained using the prepared and augmented data with appropriate optimization strategies and callbacks.
- Model Evaluation: The trained model's performance is assessed on a dedicated test set.

## Feature Engineering
In this project, feature engineering is primarily conducted through Data Augmentation. The ImageDataGenerator from Keras is utilized to create diverse variations of the existing images, effectively expanding the training dataset and making the model more robust to different orientations and conditions of sign language gestures. The augmentation parameters include:
- `rotation_range=20`
- `zoom_range=0.2`
- `width_shift_range=0.2`
- `height_shift_range=0.2`
- `shear_range=0.2`
- `horizontal_flip=True`
- `fill_mode='nearest'`

The input images are resized to 200x200 pixels.

## Model Training and Performance
### Model Architecture
The model utilizes a transfer learning approach, building upon the pre-trained MobileNet architecture. The include_top parameter is set to False to remove the top classification layer, allowing for the addition of custom layers suitable for the BISINDO alphabet classification task. The weights are initialized from the 'imagenet' dataset.
Additional layers are added on top of the MobileNet base:
- A GlobalAveragePooling2D layer to reduce dimensionality.
- A Dense output layer with 26 units (corresponding to the 26 alphabet classes) and 'softmax' activation for multi-class classification.

### Training Configuration
- Optimizer: Adam optimizer with a learning_rate of 0.001.
- Loss Function: categorical_crossentropy, suitable for multi-class classification.
- Epochs: The model was trained for 100 epochs.
- Batch Size: 64 images per batch.
  - Callbacks: EarlyStopping: Monitors validation loss and stops training if no improvement is observed for 10 consecutive epochs (patience=10). It also restores_best_weights found during training.
  - ReduceLROnPlateau: Reduces the learning rate by a factor of 0.2 if the validation loss does not improve for 5 consecutive epochs (patience=5).

### Performance
The model achieved the following accuracy on the test set:
- Accuracy on test set: 85.45%

## Conclusion
This project successfully developed a deep learning model for classifying Indonesian Sign Language (BISINDO) alphabet images. Leveraging transfer learning with MobileNet and extensive data augmentation, the model achieved a commendable accuracy of 85.45% on the test set. This demonstrates the potential of deep learning in recognizing visual gestures for sign language, which can be a valuable step towards assistive technologies.

## Future Recommendations
To further enhance this project, the following recommendations can be considered:
- Data Expansion and Diversity: Gather a larger and more diverse dataset, including images with varying backgrounds, lighting conditions, skin tones, and hand orientations. This would improve the model's generalization capabilities.
- Explore Other Architectures: Experiment with other state-of-the-art pre-trained CNN architectures (e.g., EfficientNet, ResNet, VGG16) or custom CNN designs to potentially achieve higher accuracy and efficiency.
- Hyperparameter Tuning: Conduct a more exhaustive hyperparameter tuning process for the model, including different learning rates, batch sizes, optimizer variations, and regularization techniques.
- Real-time Inference: Develop a real-time application (e.g., using OpenCV) that can capture live video feed and classify BISINDO alphabets, making it more practical for interactive use.
- Multi-sign Recognition: Extend the project to recognize sequences of signs to form words or phrases, moving beyond single-letter classification.
- Edge Device Deployment: Optimize the model for deployment on edge devices (e.g., mobile phones, Raspberry Pi) for more accessible and portable sign language recognition.
