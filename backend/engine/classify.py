import tensorflow as tf
import tensorflow_hub as hub
import os
from IPython.display import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from matplotlib.pyplot import imread
from sklearn.metrics import classification_report, accuracy_score
import warnings
# from datasets.dataloader import download_dataset
warnings.filterwarnings("ignore")

batch_size = 32
img_size = 224

class Classify:
    # Define the dataset and path
    dataset = 'alxmamaev/flowers-recognition'  # Replace with the actual dataset path
    data_dir = 'datasets/input'  # Local directory to check/download

    # Create a simple function to return a tuple (image, label)
    def get_image_label(image_path, label):
        image = self.process_image(image_path)
        return image, label

    # Create a function to turn data into batches
    def create_data_batches(X, y=None, batch_size=batch_size, test_data=False):
        if test_data:
            print("Creating test data batches...")
            data = tf.data.Dataset.from_tensor_slices((tf.constant(X))) # only filepaths (no labels)
            data_batch = data.map(self.process_image).batch(batch_size)
            return data_batch
        else:
            print("Creating data batches...")
            # Turn filepaths and labels into Tensors
            data = tf.data.Dataset.from_tensor_slices((tf.constant(X), tf.constant(y)))
            # Shuffling pathnames and labels before mapping image processor function is faster than shuffling images
            data = data.shuffle(buffer_size=len(X))

            # Create (image, label) tuples (this also turns the iamge path into a preprocessed image)
            data = data.map(self.get_image_label)

            # Turn the training data into batches
            data_batch = data.batch(batch_size)
            return data_batch

    # Create a function for preprocessing images
    def process_image(image_path, img_size=img_size):
        # Read in an image file
        image = tf.io.read_file(image_path)
        # Turn the jpeg image into numerical Tensor with 3 colour channels (Red, Green, Blue)
        image = tf.image.decode_jpeg(image, channels=3)
        # Convert the colour channel values from 0-255 to 0-1 values
        image = tf.image.convert_image_dtype(image, tf.float32)
        # Resize the image to our desired value (224, 224)
        image = tf.image.resize(image, size=[img_size, img_size])

        return image

    def classify_image(self):
        # Download Dataset
        # download_dataset(self.dataset, self.data_dir)
        
        # Kinds of flowers
        kinds = np.array(os.listdir(f'{self.data_dir}/flowers'))
        print(f"Flower kinds in this dataset: {kinds}")

        path = f'{self.data_dir}/flowers'
        kind_path = [path + "/" + flower for flower in kinds]
        print(kind_path)
        
        # Numbers of flowers for each kinds
        for i, kind in enumerate(kind_path):
            print(f"There are {len(os.listdir(kind))} flowers in {kinds[i]}")
            
        # All ids
        id_df = []
        for i in range(len(kinds)):
            id = [img.split(".")[0] for img in os.listdir(kind_path[i])]
            id_df = id_df + id
        print(len(id_df))

        # All kinds
        kind_df = []
        for i, kind in enumerate(kinds):
            for x in range(len(os.listdir(kind_path[i]))):
                kind_df.append(kind)
        print(len(kind_df))
        
        # Create a dataframe
        df = pd.DataFrame(columns=["id", "kind"])
        df["id"] = id_df
        df["kind"] = kind_df
        print(df.tail())
        
        # Check numbers
        print(df["kind"].value_counts())
        
        filenames = []
        for i in range(len(kinds)):
            file = [kind_path[i] + "/" + kind for kind in os.listdir(kind_path[i])]
            filenames = filenames + file
        print(filenames[:5])
        
        # Check a random flower
        print(filenames[2317])
        print(df.loc[2317])
        
        boolean_kinds = [kind == kinds for kind in kind_df]
        print(boolean_kinds[:5])
        
        X = filenames
        y = boolean_kinds
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=18)
        print(len(X_train), len(X_test), len(y_train), len(y_test))
        
        image = imread(filenames[15])
        print(image.shape, image.max(), image.min())
        
        # Create training and validation data batches
        train_data = self.create_data_batches(X_train, y_train)
        test_data = self.create_data_batches(X_test, test_data=True)

        train_images, train_labels = next(train_data.as_numpy_iterator())

        # # Visualizing data batches
        # plt.figure(figsize=(10, 10))
        # for i in range(25):
        #     ax = plt.subplot(5, 5, i+1)
        #     plt.imshow(train_images[i])
        #     plt.title(kinds[train_labels[i].argmax()])
        #     plt.axis("off")

        # # Setup input shape to the model
        # input_shape = [None, img_size, img_size, 3] # batch, height, width, colour channels

        # # Setup output shape of our model
        # output_shape = len(kinds)
        
        # # Load model
        # model = tf.keras.Sequential([
        # hub.KerasLayer("/kaggle/input/mobilenet-v2/tensorflow2/140-224-classification/2"), #input layer
        #     tf.keras.layers.Dense(units=output_shape,
        #     activation="softmax") # output layer
        # ])
        
        return 1