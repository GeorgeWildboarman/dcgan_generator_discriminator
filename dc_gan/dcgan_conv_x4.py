import tensorflow as tf
from tensorflow.keras import layers

class DCGAN_4conv():
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
               image_size = [64, 64], # Image size
               channels = 3, # Number of channels of images
               num_filters = 64, # Number of filters for the 1st conv layers of the discriminator
               gen_kernel_size = (4, 4), # Kernel size for the generator
               disc_kernel_size = (4, 4), # Kernel size for the discriminator
               dropout_rate = 0.4, # Dropout rate for the discriminator
               ):
    super(DCGAN_4conv, self).__init__()

    self.latent_dim = latent_dim
    self.image_size = image_size
    self.channels = channels
    
    # Input Shape
    self.input_shape = image_size + [channels]
    
    self.num_filters = num_filters
    self.gen_kernel_size = gen_kernel_size
    self.disc_kernel_size = disc_kernel_size

    self.dropout_rate = dropout_rate


  def build_generator(self, activation='LeakyReLU', dense=True):
    '''Generator Model
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
      activation: Activation function to use for each layer except the output layer.
        If you specify LeakyReLU, the activation is LeakyReLU, otherwise the activation is ReLU.
      dense: Boolean, whether the 1st layer is a Dense layer. If False, the 1st layer is a Conv2DTranspose layer.
    '''
    # Project size
    project_size = [x//16 for x in self.image_size]
    # Project shape
    project_shape = project_size + [self.num_filters*16]
    dense_units = project_size[0] * project_size[1] * self.num_filters*16

    model = tf.keras.Sequential()
    
    if dense:
      model.add(tf.keras.layers.Dense(dense_units, use_bias=False, input_dim=self.latent_dim))
    else:
      model.add(tf.keras.layers.Input(shape=(1,1,self.latent_dim)))
      model.add(tf.keras.layers.Conv2DTranspose(self.num_filters*16, project_size, strides=(1, 1), padding='valid', use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    if 'leakyrelu' == activation.lower():
      activation_layer = tf.keras.layers.LeakyReLU()
    else:
      activation_layer = tf.keras.layers.ReLU()
    model.add(activation_layer)
    model.add(tf.keras.layers.Reshape(project_shape))

    # conv1
    model.add(tf.keras.layers.Conv2DTranspose(self.num_filters*8, self.gen_kernel_size, strides=(2, 2), padding='same', use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    if 'leakyrelu' == activation.lower():
      activation_layer = tf.keras.layers.LeakyReLU()
    else:
      activation_layer = tf.keras.layers.ReLU()
    model.add(activation_layer)

    # conv2
    model.add(tf.keras.layers.Conv2DTranspose(self.num_filters*4, self.gen_kernel_size, strides=(2, 2), padding='same', use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    if 'leakyrelu' == activation.lower():
      activation_layer = tf.keras.layers.LeakyReLU()
    else:
      activation_layer = tf.keras.layers.ReLU()
    model.add(activation_layer)
        
    # conv4
    model.add(tf.keras.layers.Conv2DTranspose(self.num_filters*2, self.gen_kernel_size, strides=(2, 2), padding='same', use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    if 'leakyrelu' == activation.lower():
      activation_layer = tf.keras.layers.LeakyReLU()
    else:
      activation_layer = tf.keras.layers.ReLU()
    model.add(activation_layer)
        
    # conv5
    model.add(tf.keras.layers.Conv2DTranspose(self.channels, self.gen_kernel_size, strides=(2, 2), padding='same', use_bias=False, activation='tanh'))

    return model

  def build_discriminator(self, batchnorm=False, dropout=True, dense=True):
    '''The Discriminator
    The discriminator is a CNN-based image classifier.
    Use the (as yet untrained) discriminator to classify the generated images as real or fake.
    The model will be trained to output positive values for real images, and negative values for fake images.

    arguments:
      batchnorm: Boolean, whether add Batchnormalization.
      dropout: Boolean, whether add Dropout.
      dense: Boolean, whether use a Dense layer for output. If False, the last output layer is a Conv2DTranspose layer. 
    '''
    
    model = tf.keras.Sequential()

    # conv1
    model.add(tf.keras.layers.Conv2D(self.num_filters, self.disc_kernel_size, strides=(2, 2), padding='same', input_shape=self.input_shape))
    model.add(tf.keras.layers.LeakyReLU())
    if dropout:
      model.add(tf.keras.layers.Dropout(self.dropout_rate))

    # conv2
    model.add(tf.keras.layers.Conv2D(self.num_filters*2, self.disc_kernel_size, strides=(2, 2), padding='same'))
    if batchnorm:
      model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    if dropout:
      model.add(tf.keras.layers.Dropout(self.dropout_rate))

    # conv3
    model.add(tf.keras.layers.Conv2D(self.num_filters*4, self.disc_kernel_size, strides=(2, 2), padding='same'))
    if batchnorm:
      model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    if dropout:
      model.add(tf.keras.layers.Dropout(self.dropout_rate))

    # conv4
    model.add(tf.keras.layers.Conv2D(self.num_filters*8, self.disc_kernel_size, strides=(2, 2), padding='same'))
    if batchnorm:
      model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    if dropout:
      model.add(tf.keras.layers.Dropout(self.dropout_rate))

    # Output
    if dense:
      model.add(tf.keras.layers.Flatten())
      model.add(tf.keras.layers.Dense(1))
    else:
      model.add(tf.keras.layers.Conv2D(1, self.disc_kernel_size, strides=(2, 2), padding='valid'))

    return model
