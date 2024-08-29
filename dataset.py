import json
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelBinarizer

# Define paths
json_file_path = 'C:/Users/dhars/PycharmProjects/NLP/_annotations.coco.json'  # Replace with the path to your JSON file
images_dir = 'C:/Users/dhars/PycharmProjects/NLP/test'  # Path to images

# Load and parse JSON file
with open(json_file_path, 'r') as file:
    data = json.load(file)

# COCO JSON structure parsing
image_info = {image['id']: image['file_name'] for image in data['images']}
annotations = data['annotations']
labels = [annotation['category_id'] for annotation in annotations]
image_ids = [annotation['image_id'] for annotation in annotations]

# Map image_ids to file paths
image_paths = [os.path.join(images_dir, image_info[image_id]) for image_id in image_ids]

# Create a mapping of category_ids to class labels (assuming categories are provided)
categories = {category['id']: category['name'] for category in data['categories']}
labels = [categories[label] for label in labels]

# Create a mapping of labels to integers
label_binarizer = LabelBinarizer()
labels_encoded = label_binarizer.fit_transform(labels)

# Parameters
target_size = (224, 224)
batch_size = 32
num_classes = len(label_binarizer.classes_)

# Prepare data for ImageDataGenerator
def data_generator(image_paths, labels_encoded, batch_size, target_size):
    while True:
        for start in range(0, len(image_paths), batch_size):
            end = min(start + batch_size, len(image_paths))
            batch_images = []
            batch_labels = []
            for i in range(start, end):
                img = load_img(image_paths[i], target_size=target_size)
                img_array = img_to_array(img) / 255.0  # Rescale image to [0, 1]
                batch_images.append(img_array)
                batch_labels.append(labels_encoded[i])
            yield np.array(batch_images), np.array(batch_labels)

# Define the VGG16 model
base_model = VGG16(weights=None, include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = Flatten()(x)
x = Dense(512, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Create ImageDataGenerator for augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Create train_generator using the custom data generator
train_generator = data_generator(image_paths, labels_encoded, batch_size, target_size)

# Train the model
model.fit(
    train_generator,
    epochs=50,
    steps_per_epoch=len(image_paths) // batch_size
)

# Save the model
model.save('my_custom_vgg16_model.h5')
