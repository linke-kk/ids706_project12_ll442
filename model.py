# customary imports:
import numpy as np
# import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from sklearn.model_selection import StratifiedKFold
import mlflow
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
import requests
import zipfile
# import os

# Function to download a file from a URL
def download_file(url, filename):
    response = requests.get(url)
    with open(filename, 'wb') as file:
        file.write(response.content)

# Function to unzip a file
def unzip_file(zip_filepath, extract_to):
    with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

# import dataset from the website
def import_data():
    # Download the file
    data_url = "https://data.mendeley.com/public-files/datasets/snkd93bnjr/files/2fc38728-2ae7-4a62-a857-032af82334c3/file_downloaded"
    download_file(data_url, "data.zip")

    # Unzip the downloaded file
    unzip_file("data.zip", "./data")

    # Unzip another file within the extracted contents
    # unzip_file("./data/PBC_dataset_normal_DIB.zip", "./data")
    print("Importing data...")

    sample_image = Image.open("./data/PBC_dataset_normal_DIB/basophil/BA_100102.jpg")

# preprocess the dataset for better training
def load_and_crop(image_path, crop_size, normalized=True):
    image = Image.open(image_path).resize([200,200])
    width, height = image.size   # Get dimensions
    left = (width - crop_size)/2
    top = (height - crop_size)/2
    right = (width + crop_size)/2
    bottom = (height + crop_size)/2
    # Crop the center of the image
    image = ImageOps.grayscale(image.crop((left, top, right, bottom)))
    if normalized:
        return np.array(image).astype(np.float32) / 255.0
    else:
        return np.array(image).astype(np.float32)
    
# create the dataset
def dataset_creation():
    # code to load all the data, assuming dataset is at PBC_dataset_normal_DIB relative path
    # cell_types = ['basophil', 'eosinophil', 'erythroblast', 'ig', 'lymphocyte', 'monocyte', 'neutrophil', 'platelet']
    cell_types = ['basophil', 'eosinophil']
    cell_inds = np.arange(0, len(cell_types))
    x_data = []
    y_data = []
    # print(cell_types)
    x_train, y_train, x_val, y_val = [], [], [], []

    for cell_ind in cell_inds:
        # directory = 'PBC_dataset_normal_DIB/' + cell_types[cell_ind]
        # pattern = directory + '/*.jpg'
        # all_images = glob.glob(pattern)
        directory = Path('./data/PBC_dataset_normal_DIB') / cell_types[cell_ind]
        all_images = [f for f in directory.iterdir() if f.is_file() and f.suffix == '.jpg']
        print(all_images)
        x_data += [load_and_crop(image_path, 128) for image_path in all_images]
        y_data += [cell_ind]*len(all_images)
        # print(x_data)

    # adding a fake color channel
    x_data = np.array(x_data).reshape(-1, 128, 128, 1)
    y_data = np.array(y_data)

    folder = StratifiedKFold(5, shuffle=True)
    x_indices = np.arange(0, len(x_data))
    train_indices, val_indices = folder.split(x_indices, y_data).__next__()
    # print(train_indices, val_indices)
    # # shuffling
    np.random.shuffle(train_indices)

    x_train = x_data[train_indices]
    y_train = np.eye(len(cell_types))[y_data[train_indices]]

    x_val = x_data[val_indices]
    y_val = np.eye(len(cell_types))[y_data[val_indices]]

    print("shape of x_train, y_train, x_val, y_val:")
    print(x_train.shape, y_train.shape, x_val.shape, y_val.shape)
    return x_train, y_train, x_val, y_val

# create the model
def create_model():
    image_size=(128,128,1)

    cnn_model = tf.keras.models.Sequential([
        # 1-Input layer
        tf.keras.layers.Input(image_size),

        # 1,2-convolutional layers:
        tf.keras.layers.Conv2D(filters=6, kernel_size=5),
        tf.keras.layers.Conv2D(filters=6, kernel_size=5),

        # 3-max pooling layer:
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        # 4,5-convolutional layers:
        tf.keras.layers.Conv2D(filters=6, kernel_size=5),
        tf.keras.layers.Conv2D(filters=6, kernel_size=5),

        # max pooling layer:
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        # dense layer:
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(8, activation='softmax')

    ])
    return cnn_model

# pipeline for model training
def model_training():
    mlflow.start_run()
    # 1. import dataset: run only for the first time to import the data set
    # import_data()

    # 2. preprocessing
    x_train, y_train, x_val, y_val = dataset_creation()

    # 3. create model
    model = create_model()
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),  # pick an optimizer
                     loss=tf.keras.losses.CategoricalCrossentropy(),  # pick a loss
                     metrics=['accuracy'])  # pick a metric to monitor
    
    # 4.log model
    mlflow.sklearn.log_model(model, "model")

    # 5. train model
    hist = model.fit(x_train, y_train,
              epochs=10,
              batch_size=32,
              validation_data=(x_val, y_val))
    
    # 6. log metrics
    mlflow.log_metric("loss", hist.history['loss'][-1])
    mlflow.log_metric("accuracy", hist.history['accuracy'][-1])
    
    mlflow.end_run()