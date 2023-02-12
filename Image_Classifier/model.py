import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
import glob
import os
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

class Model():
    def __init__(self,config):
        self.config = config
    def createCustomModel(self,num_classes):
        model = Sequential([
          layers.Rescaling(1./255, input_shape=(self.config['imag_height'], self.config['img_width'], 3)),
          layers.Conv2D(16, 3, padding='same', activation='relu'),
          layers.MaxPooling2D(),
          layers.Conv2D(32, 3, padding='same', activation='relu'),
          layers.MaxPooling2D(),
          layers.Conv2D(64, 3, padding='same', activation='relu'),
          layers.MaxPooling2D(),
          layers.Flatten(),
          layers.Dense(128, activation='relu'),
          layers.Dense(num_classes)
        ])
        return model
    def createTransferLearningModel(self,num_classes):
        model = ''
        return model
    def compileAndFitModel(self,model,train_ds,val_ds):
        model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
        with tf.device(self.config['device']):
            history = model.fit(
              train_ds,
              validation_data=val_ds,
              epochs=self.config['epochs']
            )
            model.save(self.config['model_save_path'])
        return history
    def visualizeTrainingModel(self,history):
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']

        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs_range = range(self.config['epochs'])

        plt.figure(figsize=(8, 8))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Training Accuracy')
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.show()
        plt.savefig(self.config['training_path'])
        return True