import time
import numpy as np

import tensorflow as tf

from IPython import display

class gantrain():
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
               generator, # Tf.Model, generator model.
               discriminator, #Tf.Model, discriminator model.
               latent_dim = 100, # Dimension of random noise (latent space vectors)
              #  image_size = (64, 64), # Image size
              #  channels = 3, # Number of channels of images
              #  num_filters = 64, # Number of filters for the 1st conv layers of the discriminator
              #  gen_kernel_size = (4, 4), # Kernel size for the generator
              #  disc_kernel_size = (4, 4), # Kernel size for the discriminator
              #  dropout_rate = 0.4, # Dropout rate for the discriminator
              #  activation='LeakyReLU', # Activation for the generator
              #  batchnorm=False, 
              #  dropout=True, 
              #  dense=True,
               learning_rate = 0.0002, # Learning rate for the discriminator and the generator optimizers
               beta1 = 0.5,
               checkpoint_prefix = None,
               ):

    self.latent_dim = latent_dim
    self.checkpoint_prefix = checkpoint_prefix
    
    self.generator = generator
    self.discriminator = discriminator

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
  
