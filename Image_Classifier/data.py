import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
import glob
import os
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

class Data:
    def __init__(self,config):
        self.config = config
    def get_dataset(self):
        train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
                            self.config['img_path_dir'],
                            validation_split = self.config['val_split'],
                            subset = self.config['subset'],
                            seed = self.config['seed'],
                            image_size = (self.config['imag_height'], self.config['img_width']),
                            batch_size = self.config['batch_size'])
        classes = train_ds.class_names
        AUTOTUNE = tf.data.AUTOTUNE
        train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
        val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
        return train_ds, val_ds, classes
    def normalize_data(self,train_ds):
        normalization_layer = layers.Rescaling(1./255)
        normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
        image_batch, labels_batch = next(iter(normalized_ds))