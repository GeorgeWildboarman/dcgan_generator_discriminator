import tensorflow as tf
from tensorflow.keras import layers

class Initial_Block(layers.Layer):
  def __init__(self,
          latent_dim = 100, # Dimension of random noise (latent space vectors)
          image_size = (64, 64), # Image size
          num_filters = 1024, # Number of filters
          kernel_size = (4, 4), # Kernel size for the generator
          activation='LeakyReLU', # Activation for the generator
          dense=True,
          ):
    super(Initial_Block, self).__init__()
    # Project size
    project_size = [x//16 for x in image_size]
    # Project shape
    project_shape = project_size + [num_filters]
    dense_units = project_size[0] * project_size[1] * num_filters

    if dense:
      self.block=[
          layers.Dense(dense_units, use_bias=False, input_dim=latent_dim)
      ]
    else:
      self.block=[
          layers.Reshape((1,1,latent_dim)),
          layers.Conv2DTranspose(num_filters, project_size, strides=(1, 1), padding='valid', use_bias=False),
      ]

    self.block.append(layers.BatchNormalization())

    if not activation:
      pass
    elif 'leakyrelu' == activation.lower():
      self.block.append(layers.LeakyReLU())
    else:
      self.block.append(layers.ReLU())

    self.block.append(layers.Reshape(project_shape))

  def call(self, x):
    for layer in self.block:
      x = layer(x)
    return x

class TransConv_Block(layers.Layer):
  def __init__(self, num_filters=1024, kernel_size=(4,4), activation='LeakyReLU', batchnorm=True):
    super(TransConv_Block, self).__init__()
    # Transposed Conv block
    self.block =[tf.keras.layers.Conv2DTranspose(num_filters, kernel_size, strides=(2, 2), padding='same', use_bias=False)]

    if batchnorm:
      self.block.append(tf.keras.layers.BatchNormalization())
    
    if not activation:
      pass
    elif 'leakyrelu' == activation.lower():
      self.block.append(layers.LeakyReLU())
    elif 'relu' == activation.lower():
      self.block.append(layers.Activation('relu'))
    elif 'tanh':
      self.block.append(layers.Activation('tanh'))

  def call(self, x):
    for layer in self.block:
      x = layer(x)
    return x

class Conv_Block(layers.Layer):
  def __init__(self, num_filters, kernel_size, strides, pooling='max', batchnorm=False, dropout=False, dropout_rate=0.4):
    super().__init__()

    # Conv2D
    self.conv_block = [tf.keras.layers.Conv2D(num_filters, kernel_size, strides=strides, padding='same')]
    # Pooling
    if pooling == 'max':
      self.conv_block.append(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    elif pooling:
      self.conv_block.append(tf.keras.layers.AveragePooling2D(pool_size=(2, 2)))
    # Batch Normalization
    if batchnorm:
      self.conv_block.append(tf.keras.layers.BatchNormalization())
    # Leaky ReLU Activaton
    self.conv_block.append(tf.keras.layers.LeakyReLU())
    # Dropout
    if dropout:
      self.conv_block.append(tf.keras.layers.Dropout(dropout_rate))

  def call(self, x):
    for layer in self.conv_block:
      x = layer(x)
    return x

class Header_Block(layers.Layer):
  def __init__(self, dense=True, kernel_size=(4,4)):
    super(Header_Block, self).__init__()
# Header block
    if dense:
      self.block = [tf.keras.layers.Flatten(), tf.keras.layers.Dense(1)]
    else:
      self.block = [tf.keras.layers.Conv2D(1, kernel_size, strides=(2, 2), padding='valid'), tf.keras.layers.GlobalAveragePooling2D()]

  def call(self, x):
    for layer in self.block:
      x = layer(x)
    return x

class ResNet_Block(layers.Layer):
  def __init__(self, filter_size=16, kernel_size=(3,3), strides=(1,1)):
    ''' args:
         filter_size: numbers of output filters.
         kernel_size: kernel_size. default is (3,3)
         strides: turple. If this is not (1,1), skip connection are replaced to convolution layer and it makes an output downsized.
    '''
    super(ResNet_Block, self).__init__()

    self.norm1 = layers.BatchNormalization()
    self.act1 = layers.Activation(tf.nn.relu)
    self.conv1 = layers.Conv2D(filter_size, kernel_size, strides=strides, padding='same')

    if strides == (1,1):
      self.shortcutblock = [layers.Identity()]
    else:
      self.shortcutblock = [
          layers.BatchNormalization(),
          layers.Activation(tf.nn.relu),
          layers.Conv2D(filter_size, (1,1), strides=strides, padding='same')
       ]

    self.norm2 = layers.BatchNormalization()
    self.act2 = layers.Activation(tf.nn.relu)
    self.conv2 = layers.Conv2D(filter_size, kernel_size, strides=(1,1), padding='same')

    self.add = layers.Add()

  def call(self, input):
    # Residual Block
    x = self.norm1(input, training=False)
    x = self.act1(x)
    x = self.conv1(x)

    x = self.norm2(x, training=False)
    x = self.act2(x)
    x = self.conv2(x)

    # Skip Connection Block
    skip = input
    for layer in self.shortcutblock:
      skip = layer(skip)

    x = self.add([x, skip])

    return x


class SE_Block(layers.Layer):
  def __init__(self, num_filters=16, ratio=16):
    ''' args:
         num_filters: numbers of output filters.
         kernel_size: kernel_size. default is (3,3)
         strides: turple. If this is not (1,1), skip connection are replaced to convolution layer and it makes an output downsized.
    '''
    super(SE_Block, self).__init__()

    self.num_filters = num_filters

    self.avepooling = layers.GlobalAveragePooling2D()
    self.dens1 = layers.Dense(num_filters//ratio, activation='relu')
    self.dens2 = layers.Dense(num_filters, activation='sigmoid')
    self.reshape = layers.Reshape((1, 1, num_filters))
    self.multiply = layers.Multiply()

  def call(self, input):
    x = self.avepooling(input)
    x = self.dens1(x)
    x = self.dens2(x)
    x = self.reshape(x)
    x = self.multiply([input, x])
    return x

class SEResNet_Block(layers.Layer):
  def __init__(self, filter_size=16, kernel_size=(3,3), strides=(1,1)):
    ''' args:
         filter_size: numbers of output filters.
         kernel_size: kernel_size. default is (3,3)
         strides: turple. If this is not (1,1), skip connection are replaced to convolution layer and it makes an output downsized.
    '''
    super(SEResNet_Block, self).__init__()

    self.norm1 = layers.BatchNormalization()
    self.act1 = layers.Activation(tf.nn.relu)
    self.conv1 = layers.Conv2D(filter_size, kernel_size, strides=strides, padding='same')

    if strides == (1,1):
      self.shortcutblock = [layers.Identity()]
    else:
      self.shortcutblock = [
          layers.BatchNormalization(),
          layers.Activation(tf.nn.relu),
          layers.Conv2D(filter_size, (1,1), strides=strides, padding='same')
       ]

    self.norm2 = layers.BatchNormalization()
    self.act2 = layers.Activation(tf.nn.relu)
    self.conv2 = layers.Conv2D(filter_size, kernel_size, strides=(1,1), padding='same')

    self.seblock = SE_Block(filter_size)

    self.add = layers.Add()

  def call(self, input):
    # Residual Block
    x = self.norm1(input, training=False)
    x = self.act1(x)
    x = self.conv1(x)

    x = self.norm2(x, training=False)
    x = self.act2(x)
    x = self.conv2(x)

    # SE block
    x = self.seblock(x)

    # Skip Connection Block
    skip = input
    for layer in self.shortcutblock:
      skip = layer(skip)

    x = self.add([x, skip])

    return x

class ResNet_Blocks(layers.Layer):
  def __init__(self, num_filters, resnet=None, depth=0):
    super().__init__()
    # depth: int. Number of blocks.

    # ResNet Block
    if not resnet:
      self.block = [tf.keras.layers.Identity()]
    elif 'se' in resnet.lower():
      self.block = [
          SEResNet_Block(num_filters) for _ in range(depth)
      ]
    else:
      self.block = [
          ResNet_Block(num_filters) for _ in range(depth)
      ]

  def call(self, x):
    for layer in self.block:
      x = layer(x)
    return x



def build_generator(
        latent_dim = 100, # Dimension of random noise (latent space vectors)
        image_size = (64, 64), # Image size
        channels = 3, # Number of channels of images
        num_filters = 64, # Number of filters for the last conv layer of the generator
        gen_kernel_size = (4, 4), # Kernel size for the generator
        activation='LeakyReLU', # Activation for the generator
        dense=True,
        resnet = '',
        ):

  '''Generator Model for DCGAN
  The generator uses tf.keras.layers.Conv2DTranspose (upsampling) layers to produce an image from a seed (random noise),
  which is a vector with the dimension of latent_dim.

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
    resnet: Str, ResNet is adopted when 'Res', SEResnet is adopted when 'SE', None is none adopt.
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

  # ResNet
  if 'se' in resnet.lower():
    model.add(SEResNet_Block(num_filters*16))
  elif 'res' in resnet.lower():
    model.add(ResNet_Block(num_filters*16))
  else:
    pass

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
        dense=True,
        pooling=None,
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

  if pooling:
    strides=(1,1)
  else:
    strides=(2,2)

  # conv1
  model.add(tf.keras.layers.Conv2D(num_filters, disc_kernel_size, strides=strides, padding='same', input_shape=input_shape))
  if pooling == 'max':
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
  elif pooling:
    model.add(tf.keras.layers.AveragePooling2D(pool_size=(2, 2)))

  model.add(tf.keras.layers.LeakyReLU())

  if dropout:
    model.add(tf.keras.layers.Dropout(dropout_rate))

  # conv2
  model.add(tf.keras.layers.Conv2D(num_filters*2, disc_kernel_size, strides=strides, padding='same'))
  if pooling == 'max':
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
  elif pooling:
    model.add(tf.keras.layers.AveragePooling2D(pool_size=(2, 2)))

  if batchnorm:
    model.add(tf.keras.layers.BatchNormalization())

  model.add(tf.keras.layers.LeakyReLU())

  if dropout:
    model.add(tf.keras.layers.Dropout(dropout_rate))

  # conv3
  model.add(tf.keras.layers.Conv2D(num_filters*4, disc_kernel_size, strides=strides, padding='same'))
  if pooling == 'max':
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
  elif pooling:
    model.add(tf.keras.layers.AveragePooling2D(pool_size=(2, 2)))

  if batchnorm:
    model.add(tf.keras.layers.BatchNormalization())

  model.add(tf.keras.layers.LeakyReLU())

  if dropout:
    model.add(tf.keras.layers.Dropout(dropout_rate))

  # conv4
  model.add(tf.keras.layers.Conv2D(num_filters*8, disc_kernel_size, strides=strides, padding='same'))
  if pooling == 'max':
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
  elif pooling:
    model.add(tf.keras.layers.AveragePooling2D(pool_size=(2, 2)))

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
    model.add(tf.keras.layers.GlobalAveragePooling2D())

  return model

class SEResNet_generator(tf.keras.Model):
  '''Generator Model for DCGAN with SE-ResNet
  A Deep Convolutional(DC) generator uses tf.keras.layers.Conv2DTranspose (upsampling) layers to produce an image from a seed (random noise),
  which is a vector with the dimension of latent_dim. This generator is improved with adding SE-ResNet blocks.

  arguments:
    latent_dim: Int. Dimension of random noise (latent space vectors).
    image_size: Tuple. Image size with the sphape of (height, width).
    channels: Int. Number of channels of images
    num_filters: Int. Number of filters for the 1st conv layer of the discriminator.
    gen_kernel_size: Tuple. Kernel size for the generator
    activation: str: Str, the activation functions for each layer except the output layer. If you specify "LeakyReLU", the activations are LeakyReLU, otherwise ReLU.
    dense: Boolean, whether the 1st layer is a Dense layer. If False, the 1st layer is a Conv2DTranspose layer.
    resnet: Str, ResNet is adopted when 'Res', SEResnet is adopted when 'SE', None is none adopt.
  '''
  def __init__(self,
          latent_dim = 100, # Dimension of random noise (latent space vectors)
          image_size = (64, 64), # Image size
          channels = 3, # Number of channels of images
          num_filters = 64, # Number of filters for the last conv layer of the generator
          kernel_size = (4, 4), # Kernel size for the generator
          activation='LeakyReLU', # Activation for the generator
          dense=True,
          resnet = None,
          depths=[1,0,0], # List of number of blocks
          ):
    super().__init__()
    
    # Initial Block
    self.initial_block = Initial_Block(latent_dim, image_size, num_filters*16, kernel_size, activation, dense)
    # ResNet Block 1
    self.res_block1 = ResNet_Blocks(num_filters*16, resnet, depths[0])
    # conv block1
    self.conv_block1 = TransConv_Block(num_filters*8, kernel_size, activation)
    # ResNet Block 2
    self.res_block2 = ResNet_Blocks(num_filters*8, resnet, depths[1])
    # conv2
    self.conv_block2 = TransConv_Block(num_filters*4, kernel_size, activation)
    # ResNet Block 3
    self.res_block3 = ResNet_Blocks(num_filters*4, resnet, depths[2])
    # conv3
    self.conv_block3 = TransConv_Block(num_filters*2, kernel_size, activation)
    # conv4
    self.conv_block4 = TransConv_Block(channels, kernel_size, activation='tanh', batchnorm=False)

  def call(self, x):
    x = self.initial_block(x)
    x = self.res_block1(x)
    x = self.conv_block1(x)
    x = self.res_block2(x)
    x = self.conv_block2(x)
    x = self.res_block3(x)
    x = self.conv_block3(x)
    x = self.conv_block4(x)
    return x

class SEResNet_discriminator(tf.keras.Model):
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
  def __init__(self,
        image_size = (64, 64), # Image size
        channels = 3, # Number of channels of images
        num_filters = 64, # Number of filters for the 1st conv layer of the discriminator
        kernel_size = (4, 4), # Kernel size for the discriminator
        dropout_rate = 0.4, # Dropout rate for the discriminator
        batchnorm=False,
        dropout=True,
        dense=True,
        pooling=None,
        resnet = None,
        depths = [1,0,0,0] # Number of blocks
        ):
    super().__init__()

    # Input Shape
    input_shape = list(image_size) + [channels]


    if pooling:
      strides=(1,1)
    else:
      strides=(2,2)

    # conv1
    self.conv_block1 = Conv_Block(num_filters, kernel_size, strides, pooling, False, dropout, dropout_rate)
    # ResNet Block 1
    self.res_block1 = ResNet_Blocks(num_filters, resnet, depth=depths[0])
    # conv2
    self.conv_block2 = Conv_Block(num_filters*2, kernel_size, strides, pooling, batchnorm, dropout, dropout_rate)
    # ResNet Block 2
    self.res_block2 = ResNet_Blocks(num_filters*2, resnet, depth=depths[1])
    # conv3
    self.conv_block3 = Conv_Block(num_filters*4, kernel_size, strides, pooling, batchnorm, dropout, dropout_rate)
    # ResNet Block 3
    self.res_block3 = ResNet_Blocks(num_filters*4, resnet, depth=depths[2])
    # conv4
    self.conv_block4 = Conv_Block(num_filters*8, kernel_size, strides, pooling, batchnorm, dropout, dropout_rate)
    # ResNet Block 4
    self.res_block4 = ResNet_Blocks(num_filters*8, resnet, depth=depths[3])
    # Header block
    self.header_block = Header_Block(dense, kernel_size)
  
  def call(self, x):
    x = self.conv_block1(x)
    x = self.res_block1(x)
    x = self.conv_block2(x)
    x = self.res_block2(x)
    x = self.conv_block3(x)
    x = self.res_block3(x)
    x = self.conv_block4(x)
    x = self.res_block4(x)
    x = self.header_block(x)
    return x
