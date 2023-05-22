import os
import time
import numpy as np

import tensorflow as tf
from tensorflow.keras import layers

from IPython import display

def load_image(image_path, channels=3):
  """Loads and preprocesses images.
  arguments:
    image_path: String. File path or url to read from. If set url, the url must start with "http" or "https.
    channels: Int. Number of channels of images.
  """
  if 'http' in image_path:
    # Get image file from url and cache locally.
    image_path = tf.keras.utils.get_file(os.path.basename(image_path)[-128:], image_path)

  # Load and convert to float32 numpy array, and normalize to range [0, 1].
  image = tf.io.decode_image(tf.io.read_file(image_path), channels=channels, dtype=tf.float32,)
  # Normalize to range [-1, 1]
  image = image*2.0-1.0
  return image

def augmentation_image(image, image_size=(64, 64), expand=1.1):
  expand_size = [int(x*expand) for x in image_size]
  image = tf.image.resize(image, expand_size)
  image_shape = image_size + (image.shape[-1],)
  image = tf.image.random_crop(value=image, size=image_shape)
  image = tf.image.random_flip_left_right(image)
  return image

def load_and_preprocessing_data(path_list, image_size=(64, 64), batch_size=64, channels=3, aug_num=5, expand=1.1):
  prep_images=[]
  print('Number of files to load:', len(path_list))
  for path in path_list:
    image = load_image(path, channels)
    for _ in range(aug_num):
      prep_image =augmentation_image(image, image_size, expand)
      prep_images.append(prep_image)

  print('Number of augmented images:',len(prep_images))
  buffer_size = len(prep_images)

  dataset = tf.data.Dataset.from_tensor_slices(prep_images)
  dataset = dataset.shuffle(buffer_size).batch(batch_size)

  print('Batch size:',batch_size)
  print('Num batchs', len(dataset))

  return dataset

def build_generator(
        latent_dim = 100, # Dimension of random noise (latent space vectors)
        image_size = (64, 64), # Image size
        channels = 3, # Number of channels of images
        num_filters = 64, # Number of filters for the last conv layer of the generator
        gen_kernel_size = (4, 4), # Kernel size for the generator
        activation='LeakyReLU', # Activation for the generator
        dense=True
        ):
  
  '''Generator Model for DCGAN
  The generator uses tf.keras.layers.Conv2DTranspose (upsampling) layers to produce an image from a seed (random noise),
  which is a vector with the dimention of latent_dim.

  Start with a Dense layer that takes this seed as input.
  Project the output of the dense layer to a project_shape that is defined by project_size and channels.
  ex: project_size=[4, 4] and channels=3 make project_shape=[4, 4, 3].
  Upsample 4 times through Conv2DTranspose layer. This process generates an image that is 2 to the power of 4 times larger than the input.
  As project_shape=[4, 4, 3] the image generated has the shpae of [64, 64, 3].
  Notice the tf.keras.layers.LeakyReLU or the tf.keras.layers.ReLU activation for each layer, except the output layer which uses tanh.
  The default is ReLU.

  arguments:
    latent_dim: Int. Dimension of random noise (latent space vectors).
    image_size: Tuple. Image size with the sphape of (height, width).
    channels: Int. Number of channels of images
    num_filters: Int. Number of filters for the 1st conv layer of the discriminator.
    gen_kernel_size: Tuple. Kernel size for the generator
    activation: str: Str, the activation functions for each layer except the output layer. If you specify "LeakyReLU", the activations are LeakyReLU, otherwise ReLU.
    dense: Boolean, whether the 1st layer is a Dense layer. If False, the 1st layer is a Conv2DTranspose layer.
  '''
  # Project size
  project_size = [x//16 for x in image_size]
  # Project shape
  project_shape = project_size + [num_filters*16]
  dense_units = project_size[0] * project_size[1] * num_filters*16

  model = tf.keras.Sequential()
  
  if dense:
    model.add(tf.keras.layers.Dense(dense_units, use_bias=False, input_dim=latent_dim))
  else:
    model.add(tf.keras.layers.Input(shape=(latent_dim)))
    model.add(tf.keras.layers.Reshape((1,1,latent_dim)))
    model.add(tf.keras.layers.Conv2DTranspose(num_filters*16, project_size, strides=(1, 1), padding='valid', use_bias=False))
  model.add(tf.keras.layers.BatchNormalization())
  if 'leakyrelu' == activation.lower():
    activation_layer = tf.keras.layers.LeakyReLU()
  else:
    activation_layer = tf.keras.layers.ReLU()
  model.add(activation_layer)
  model.add(tf.keras.layers.Reshape(project_shape))

  # conv1
  model.add(tf.keras.layers.Conv2DTranspose(num_filters*8, gen_kernel_size, strides=(2, 2), padding='same', use_bias=False))
  model.add(tf.keras.layers.BatchNormalization())
  if 'leakyrelu' == activation.lower():
    activation_layer = tf.keras.layers.LeakyReLU()
  else:
    activation_layer = tf.keras.layers.ReLU()
  model.add(activation_layer)

  # conv2
  model.add(tf.keras.layers.Conv2DTranspose(num_filters*4, gen_kernel_size, strides=(2, 2), padding='same', use_bias=False))
  model.add(tf.keras.layers.BatchNormalization())
  if 'leakyrelu' == activation.lower():
    activation_layer = tf.keras.layers.LeakyReLU()
  else:
    activation_layer = tf.keras.layers.ReLU()
  model.add(activation_layer)
      
  # conv4
  model.add(tf.keras.layers.Conv2DTranspose(num_filters*2, gen_kernel_size, strides=(2, 2), padding='same', use_bias=False))
  model.add(tf.keras.layers.BatchNormalization())
  if 'leakyrelu' == activation.lower():
    activation_layer = tf.keras.layers.LeakyReLU()
  else:
    activation_layer = tf.keras.layers.ReLU()
  model.add(activation_layer)
      
  # conv5
  model.add(tf.keras.layers.Conv2DTranspose(channels, gen_kernel_size, strides=(2, 2), padding='same', use_bias=False, activation='tanh'))

  return model


def build_discriminator(
        image_size = (64, 64), # Image size
        channels = 3, # Number of channels of images
        num_filters = 64, # Number of filters for the 1st conv layer of the discriminator
        disc_kernel_size = (4, 4), # Kernel size for the discriminator
        dropout_rate = 0.4, # Dropout rate for the discriminator
        batchnorm=False, 
        dropout=True, 
        dense=True
        ):
  '''The Discriminator
  The discriminator is a CNN-based image classifier.
  Use the (as yet untrained) discriminator to classify the generated images as real or fake.
  The model will be trained to output positive values for real images, and negative values for fake images.

  arguments:
    image_size: Tuple. Image size with the sphape of (height, width).
    channels: Int. Number of channels of images.
    num_filters: Int. Number of filters for the 1st conv layer of the discriminator.
    disc_kernel_size: Tuple. Kernel size for the discriminator.
    dropout_rate: Float. Dropout rate for the discriminator.
    batchnorm: Boolean, whether add Batchnormalization.
    dropout: Boolean, whether add Dropout.
    dense: Boolean, whether use a Dense layer for output. If False, the last output layer is a Conv2D layer. 
  '''
  # Input Shape
  input_shape = list(image_size) + [channels]

  model = tf.keras.Sequential()

  # conv1
  model.add(tf.keras.layers.Conv2D(num_filters, disc_kernel_size, strides=(2, 2), padding='same', input_shape=input_shape))
  model.add(tf.keras.layers.LeakyReLU())
  if dropout:
    model.add(tf.keras.layers.Dropout(dropout_rate))

  # conv2
  model.add(tf.keras.layers.Conv2D(num_filters*2, disc_kernel_size, strides=(2, 2), padding='same'))
  if batchnorm:
    model.add(tf.keras.layers.BatchNormalization())
  model.add(tf.keras.layers.LeakyReLU())
  if dropout:
    model.add(tf.keras.layers.Dropout(dropout_rate))

  # conv3
  model.add(tf.keras.layers.Conv2D(num_filters*4, disc_kernel_size, strides=(2, 2), padding='same'))
  if batchnorm:
    model.add(tf.keras.layers.BatchNormalization())
  model.add(tf.keras.layers.LeakyReLU())
  if dropout:
    model.add(tf.keras.layers.Dropout(dropout_rate))

  # conv4
  model.add(tf.keras.layers.Conv2D(num_filters*8, disc_kernel_size, strides=(2, 2), padding='same'))
  if batchnorm:
    model.add(tf.keras.layers.BatchNormalization())
  model.add(tf.keras.layers.LeakyReLU())
  if dropout:
    model.add(tf.keras.layers.Dropout(dropout_rate))

  # Output
  if dense:
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1))
  else:
    model.add(tf.keras.layers.Conv2D(1, disc_kernel_size, strides=(2, 2), padding='valid'))

  return model

class DCgan():
  '''DCGAN (Deep Convolutional Generative Adversarial Network).
  DCGAN is a type of generative model that uses deep convolutional neural networks
  for both the generator and discriminator components.
  It was introduced in the 2015 paper titled
  "Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks"
  by Radford et al.
  
  DCGAN has been widely used for generating realistic images across various domains,
  such as generating human faces, landscapes, and even artistic images. 

  In a DCGAN (Deep Convolutional Generative Adversarial Network),
  there are two main components: the generator and the discriminator.

    Generator: The generator takes random noise (latent space vectors) as input and generates synthetic/generated images.
    It typically consists of layers such as dense (fully connected) layers, transposed convolutional layers, batch normalization layers,
    and activation functions like ReLU or Tanh. The goal of the generator is to learn to generate realistic images that resemble the training data.

    Discriminator: The discriminator takes an image (real or generated) as input and determines whether it is real or fake.
    It consists of convolutional layers, batch normalization layers, and activation functions like LeakyReLU.
    The discriminator is trained to distinguish between real images from the training dataset and fake/generated images produced by the generator.

  The build_generator method constructs the generator model using transposed convolutional layers.
  The build_discriminator method constructs the discriminator model using convolutional layers.
  '''

  def __init__(self,
               latent_dim = 100, # Dimension of random noise (latent space vectors)
               image_size = (64, 64), # Image size
               channels = 3, # Number of channels of images
               num_filters = 64, # Number of filters for the 1st conv layers of the discriminator
               gen_kernel_size = (4, 4), # Kernel size for the generator
               disc_kernel_size = (4, 4), # Kernel size for the discriminator
               dropout_rate = 0.4, # Dropout rate for the discriminator
               activation='LeakyReLU', # Activation for the generator
               batchnorm=False, 
               dropout=True, 
               dense=True,
               learning_rate = 0.0002, # Learning rate for the discriminator and the generator optimizers
               beta1 = 0.5,
               checkpoint_prefix = None,
               ):

    self.latent_dim = latent_dim
    self.checkpoint_prefix = checkpoint_prefix
    
    self.generator = build_generator(latent_dim, image_size, channels, num_filters, gen_kernel_size, activation, dense)
    self.discriminator = build_discriminator(image_size, channels, num_filters, disc_kernel_size, dropout_rate, batchnorm, dropout, dense)

    self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    # Define the optimizers
    # The discriminator and the generator optimizers are different since you will train two networks separately.
    self.discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=beta1)
    self.generator_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=beta1)

    self.checkpoint = tf.train.Checkpoint(
        generator_optimizer=self.generator_optimizer,
        discriminator_optimizer=self.discriminator_optimizer,
        generator=self.generator,
        discriminator=self.discriminator)

  # Define the discriminator loss function
  def discriminator_loss(self, real_output, fake_output):
      real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
      fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
      total_loss = real_loss + fake_loss
      return total_loss

  # Define the generator loss function
  def generator_loss(self, fake_output):
      return self.cross_entropy(tf.ones_like(fake_output), fake_output)

  def save(self, checkpoint_prefix=None):
    if checkpoint_prefix:
      return self.checkpoint.save(file_prefix=checkpoint_prefix)
    elif self.checkpoint_prefix:
      return self.checkpoint.save(file_prefix=self.checkpoint_prefix)
    else:
      return None

  def restore(self, save_path):
    return self.checkpoint.restore(save_path)

  # Define the training step
  @tf.function
  def train_step(self, images):
      # Generate random noise
      batch_size = images.shape[0]
      noise = tf.random.normal([batch_size, self.latent_dim])

      with tf.GradientTape(persistent=True) as tape:
          # Generate fake images
          generated_images = self.generator(noise, training=True)

          # Discriminate image
          real_output = self.discriminator(images, training=True)
          fake_output = self.discriminator(generated_images, training=True)

          # Discriminator loss
          disc_loss = self.discriminator_loss(real_output, fake_output)
          # Generator loss
          gen_loss = self.generator_loss(fake_output)

      # Calculate gradients
      disc_grad = tape.gradient(disc_loss, self.discriminator.trainable_variables)
      gen_grad = tape.gradient(gen_loss, self.generator.trainable_variables)

      # Update discriminator weights
      self.discriminator_optimizer.apply_gradients(zip(disc_grad, self.discriminator.trainable_variables))

      # Update generator weights
      self.generator_optimizer.apply_gradients(zip(gen_grad, self.generator.trainable_variables))

      return disc_loss, gen_loss

  def train(self, dataset, epochs):
    gen_loss_list = []
    disc_loss_list = []
    epoch_list = []
    for epoch in range(epochs):
      start = time.time()

      for i, image_batch in enumerate(dataset):
        disc_loss, gen_loss = self.train_step(image_batch)
        print('{}th epoch {}th batch >>> Disc loss: {} , Gen loss: {}'.format(epoch+1, i, disc_loss, gen_loss))
        display.clear_output(wait=True)
      
      # Save the model every 10 epochs
      if (epoch + 1) % 10 == 0:
        print('Model saved:', self.save())
      
      print ('Time for epoch {} is {} sec'.format(epoch+1, time.time()-start))
      print('Disc loss: {} , Gen loss: {}'.format(disc_loss, gen_loss))

      epoch_list.append(epoch+1)
      gen_loss_list.append(gen_loss)
      disc_loss_list.append(disc_loss)

    return np.array([epoch_list, gen_loss_list, disc_loss_list])

