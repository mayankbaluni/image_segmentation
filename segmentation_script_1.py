'''
Image Segmentation using Mask and RGB
Task : Given an input RGB image, predict the mask of the concerned object.
Input: Given an RGB input (Height, Width, 3vals: R,G,B) - each pixel has 3 values corresponding to a single pixel
Output : Mask(R, G, 1)
'''

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

'''
Dataset : Image segementation masks 3 classses
Generic info: 37 diff catregories, ~200 samples from each class
'''

##tfds.list_builders()
## ds = tfds.load('oxford_iiit_pet', split = 'train', shuffle_files=True)
dataset = tfds.load('oxford_iiit_pet', shuffle_files=True)

print(dataset) ## printing the datastructure to see what kind of dataset is printed
print(dataset.keys()) ## printing what was the datastructure to be printed


##lets iterate over a single record from the dataset and understand the kind of parameter and points present in the datset
## so lets extract 1 single instance from the training dataset
ds = dataset['train'].take(1)
for example in ds:  # example is `{'image': tf.Tensor, 'label': tf.Tensor}`
  print(list(example.keys()))
  image = example["image"]
  label = example["label"]
  mask = example['segmentation_mask']
  print(label) ## tells us about the field of data present in the dataset record in the dataset
  print(image.shape) ## this tells us about the dimensionality of a single pricute
  print(mask.shape)
  print(example['species'])
  print('Image values:', tf.reduce_min(image), tf.reduce_max(image))
  flattened_tensor = tf.reshape(mask, [-1])
  print('Mask values:', tf.unique(flattened_tensor))  # this part gives the details on the mask part of the instance datastructure



## from the above output of the segmentation mask we see that tensorflow is using 3 labels in the images, to perform segmentation

# Displaying the dataset
for example in ds:
  image = example["image"]
  mask = example["segmentation_mask"]

  plt.figure(figsize=(15, 15))
  plt.subplot(1, 5, 1)
  plt.title('Image')
  plt.imshow(image)

  plt.subplot(1, 5, 2)
  plt.title('Mask')
  plt.imshow(mask) ## entire segmentation_mask

  # Plotting the individual masks from the labels (2,3,1) we see which label indicated what
  plt.subplot(1, 5, 3)
  plt.title('Mask=1')
  plt.imshow(mask==1) ## segmentation of the foreground

  plt.subplot(1, 5, 4)
  plt.title('Mask=2')
  plt.imshow(mask==2) ## segmentation of the background

  plt.subplot(1, 5, 5)
  plt.title('Mask=3')
  plt.imshow(mask==3) ## segmentation of the ambigious region (i.e the border beteween background and foreground)

dataset['train']

## Since the image and mask sizes are varying across the dataset, we have to standardize before pouyring it in the  model
# 1. Standardization of the dataset values present inside the data
# 2. Train + Test split the dataset for better


def normalize_img(ds):
  image = ds["image"]
  mask = ds["segmentation_mask"]
  ## Now we have to standardize the image size and also check for the interpolation method for image_segmentation and lebelling
  image = tf.image.resize(image, [128, 128])
  ## since in masking and image segmentation we have particular labels having particular meaning associated to it, we shouldnt loose it
  mask = tf.image.resize(mask, [128, 128], method = 'nearest') ## so we change the interpolation method

  """Normalizes images: `uint8` -> `float32`."""
  ## since the TFDS provide images of type tf.uint8, while the model expects tf.float32. Therefore, you need to normalize images.
  ## image normalization
  image = tf.cast(image, tf.float32)/255.0
  mask = tf.cast(mask-1, tf.float32)## since typically classification problems are 0 indexed we should make the mask values, as mask-1
   ## as the segmentation_mask values are 1,2,3
  return image, mask


ds_train = dataset['train'].map(
    normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_train = ds_train.cache() ## caching stores the data, images in a cache format before flusing out completely to save time during runtime

ds_train = ds_train.shuffle(1000)

ds_train = ds_train.batch(128) ## the data is fed in batches to the model

ds_train = ds_train.prefetch(tf.data.AUTOTUNE)## getting the next batch ready before the current batch is already passed

## Now we have to do all those things and procedures with the test dataset as well, whcih we did with training dataset

ds_test = dataset['test'].map(normalize_img, num_parallel_calls = tf.data.AUTOTUNE)

ds_test = ds_test.cache()

ds_test = ds_test.batch(128)

ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

print(ds_train)

'''
Post Processing, what we have is:-
1. All images in (128 X 128) size specifically
2. all the labelling and segmentation_masking is 0-indexed
3.
'''



# Sequential Layering based modelling
model = tf.keras.Sequential()

model.add(tf.keras.layers.Dense(8))
model.add(tf.keras.layers.Dense(1))

model.build((None, 16))

model.summary()
## Here each layer is a tranformation which converts the input to output

## Now  if we want to find out the length of the model weights
print(len(model.weights))

model1 = tf.keras.Sequential()
model1.add(tf.keras.Input(shape=(16,)))
model1.add(tf.keras.layers.Dense(4))
model1.add(tf.keras.layers.Dense(1))

model1.summary()

## Since input is already included in the layers we dont have to seperately build the model
print(len(model1.weights))

## Functional API based model building
inputs = tf.keras.Input(shape=(3,))
x = tf.keras.layers.Dense(4, activation=tf.nn.relu)(inputs) ## Here we are feeding the previous formed inputs as parameter
outputs = tf.keras.layers.Dense(5, activation=tf.nn.softmax)(x) ## here we also have the option of passing selected paramters as input
model = tf.keras.Model(inputs=inputs, outputs=outputs)

model.summary()

base_model = tf.keras.applications.mobilenet_v2.MobileNetV2(input_shape=[128, 128, 3], include_top=False)
base_model.summary()

# Use the activations of these layers
layer_names = [
    'block_1_expand_relu',   # 64x64
    'block_3_expand_relu',   # 32x32
    'block_6_expand_relu',   # 16x16
    'block_13_expand_relu',  # 8x8
    'block_16_project',      # 4x4
]
base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

# Create the feature extraction model
down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)

down_stack.trainable = False

!pip install git+https://github.com/tensorflow/examples.git
from tensorflow_examples.models.pix2pix import pix2pix
up_stack = [
    pix2pix.upsample(512, 3),  # 4x4 -> 8x8
    pix2pix.upsample(256, 3),  # 8x8 -> 16x16
    pix2pix.upsample(128, 3),  # 16x16 -> 32x32
    pix2pix.upsample(64, 3),   # 32x32 -> 64x64
]

def unet_model(output_channels):
  inputs = tf.keras.layers.Input(shape=[128, 128, 3])

  # Downsampling through the model
  skips = down_stack(inputs)
  x = skips[-1]
  skips = reversed(skips[:-1])

  # Upsampling and establishing the skip connections
  for up, skip in zip(up_stack, skips):
    x = up(x)
    concat = tf.keras.layers.Concatenate()
    x = concat([x, skip])

  # This is the last layer of the model
  last = tf.keras.layers.Conv2DTranspose(
      filters=output_channels, kernel_size=3, strides=2,
      padding='same')  #64x64 -> 128x128

  x = last(x)

  return tf.keras.Model(inputs=inputs, outputs=x)

## since in the segmentation_masking/ labelling we have 3 areas that we have to break the image down,
## we have 3 output parameters for the Unet model
model = unet_model(3)
model.summary()

# Image Segm: Per-pixel classification

model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)

tf.keras.utils.plot_model(model, show_shapes=True)

history = model.fit(
    ds_train,
    epochs=12,
    validation_data=ds_test,
)

## jUst to visualize the val_loss and how and if it decreased over a period of epochs/ iterations we can visualize it
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure()
plt.plot(history.epoch, loss, 'r', label = "Training Loss")
plt.plot(history.epoch, val_loss, 'g', label = "Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Value Loss")
plt.legend()
plt.show()

print(ds_train)

for (image, mask) in ds_test:
  pred_mask = model.predict(image)
  pred_mask = tf.math.argmax(pred_mask, axis=-1)
  pred_mask = pred_mask[..., tf.newaxis]
  print(image.shape, mask.shape, pred_mask.shape)

  plt.figure(figsize=(10, 10))
  plt.subplot(1, 3, 1)
  plt.title('Image')
  plt.imshow(image[0])

  plt.subplot(1, 3, 2)
  plt.title('GT Mask')
  plt.imshow(mask[0])

  plt.subplot(1, 3, 3)
  plt.title('Pred Mask')
  plt.imshow(pred_mask[0])

