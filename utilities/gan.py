import time
import os

import numpy as np
from IPython import display

import tensorflow as tf
from tensorflow.keras import layers

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

class ganTraining():
  '''A Generative Adversarial Network (GAN) is a type of machine learning model used for generative tasks,
  such as generating new data samples that resemble a given training dataset. GANs were first introduced by
  Ian Goodfellow and his colleagues in 2014.
   A GAN consists of two neural networks: the generator and the discriminator. These two networks are trained
  together in a competitive process, where the generator tries to create realistic data, and the discriminator
  tries to distinguish between real data and fake data generated by the generator.
  '''
  
  def __init__(self,
               generator, # Tf.Model, generator model.
               discriminator, #Tf.Model, discriminator model.
               latent_dim = 100, # Dimension of random noise (latent space vectors)
               # learning_rate = 0.0002, # Learning rate for the discriminator and the generator optimizers
               # beta_1 = 0.5,
               # beta_2 = 0.999,
               learning_rate = 0.001,
               beta_1 = 0.9,
               beta_2 = 0.999,
               checkpoint_prefix = None,
               ):

    # learning_rate=0.001,
    # beta_1=0.9,
    # beta_2=0.999,
    # epsilon=1e-07,

    self.latent_dim = latent_dim
    self.checkpoint_prefix = checkpoint_prefix
    
    self.generator = generator
    self.discriminator = discriminator

    self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    # Define the optimizers
    # The discriminator and the generator optimizers are different since you will train two networks separately.
    self.discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2)
    self.generator_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2)

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
    checkpoint_prefix = checkpoint_prefix or self.checkpoint_prefix
    if checkpoint_prefix:
      return self.checkpoint.save(file_prefix=checkpoint_prefix)
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
          generated_images = self.generator(noise)

          # Discriminate image
          real_output = self.discriminator(images)
          fake_output = self.discriminator(generated_images)

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
  
