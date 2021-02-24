import tempfile
import os
import tensorflow as tf
import numpy as np
from tensorflow import keras
import config.settings as cnst
import pandas as pd
from keras import optimizers
from utils import utils

# Load MNIST dataset
mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
# Normalize the input image so that each pixel value is between 0 to 1.
train_images = train_images / 255.0
test_images = test_images / 255.0

tr_datadf = pd.read_csv(cnst.DATA_SOURCE_PATH + cnst.ESC + "p" + str(0) + ".csv", header=None)
t1_x_train, t1_y_train = tr_datadf.iloc[:, 0].values, tr_datadf.iloc[:, 1].values
t1_batch_size = 16

'''
# Define the model architecture.
model = keras.Sequential([
  keras.layers.InputLayer(input_shape=(28, 28)),
  keras.layers.Reshape(target_shape=(28, 28, 1)),
  keras.layers.Conv2D(filters=12, kernel_size=(3, 3), activation='relu'),
  keras.layers.MaxPooling2D(pool_size=(2, 2)),
  keras.layers.Flatten(),
  keras.layers.Dense(10)
])
# Train the digit classification model
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=4, validation_split=0.1,)
_, baseline_model_accuracy = model.evaluate(test_images, test_labels, verbose=0)
print('Baseline test accuracy:', baseline_model_accuracy)

_, keras_file = tempfile.mkstemp('.h5')
tf.keras.models.save_model(model, keras_file, include_optimizer=False)
print('Saved baseline model to:', keras_file)'''


# model = tf.keras.models.load_model('C:\\Users\\anand\\AppData\\Local\\Temp\\tmp16pepe3r.h5')
model = tf.keras.models.load_model('..\\..\\model\\echelon_byte_0.h5')
#model.summary()

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
tflite_quant_model = converter.convert()
print("converted")

exit()

import tensorflow_model_optimization as tfmot
prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
# Compute end step to finish pruning after 2 epochs.
batch_size = 16
epochs = 1
validation_split = 0.1 # 10% of training set will be used for validation set.

num_images = train_images.shape[0] * (1 - validation_split)
end_step = np.ceil(num_images / batch_size).astype(np.int32) * epochs

t1_train_steps = 1

# ######################################################################################## Pruning

# Define model for pruning.
pruning_params = {'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.80, final_sparsity=0.90, begin_step=0, end_step=1)}
model_for_pruning = prune_low_magnitude(model, **pruning_params)

# `prune_low_magnitude` requires a recompile.
model_for_pruning.compile(loss='binary_crossentropy',  metrics=['accuracy'])
model_for_pruning.summary()

logdir = tempfile.mkdtemp()
callbacks = [tfmot.sparsity.keras.UpdatePruningStep(), tfmot.sparsity.keras.PruningSummaries(log_dir=logdir)]

data_gen = utils.direct_data_generator(t1_x_train, t1_y_train)
model_for_pruning.fit(data_gen, steps_per_epoch=t1_train_steps, callbacks=callbacks)  # batch_size=batch_size, epochs=epochs, validation_split=validation_split)

# _, model_for_pruning_accuracy = model_for_pruning.evaluate_generator(data_gen, steps=1)
# print('Pruned test accuracy:', model_for_pruning_accuracy)

#  ###########################################################  create a compressible model for TensorFlow
print("creating a compressible model for TensorFlow")
model_for_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning)

_, pruned_keras_file = tempfile.mkstemp('.h5')
tf.keras.models.save_model(model_for_export, pruned_keras_file, include_optimizer=False)
print('Saved pruned Keras model to:', pruned_keras_file)
model_for_export.summary()
'''
#  ###########################################################  create a compressible model for TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model_for_export)
pruned_tflite_model = converter.convert()

_, pruned_tflite_file = tempfile.mkstemp('.tflite')

with open(pruned_tflite_file, 'wb') as f:
  f.write(pruned_tflite_model)

print('Saved pruned TFLite model to:', pruned_tflite_file)'''



