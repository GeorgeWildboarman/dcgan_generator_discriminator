import tensorflow as tf
from tensorflow.keras import layers

class SelfAttention_layer(layers.Layer):
  '''Self Attention.
    Args
    ----------
      dim : Int. The input and out dimension of per token features.
      num_heads: : Int. Number of attention heads.
      qkv_bias: Boolean, whether the dense layers use bias vectors/matrices in MultiHeadAttention.
      attn_p : Float. Dropout probability applied to the query, key and value tensors.
      proj_p : Float. Dropout probability applied to the output tensor.
  '''
  
  def __init__(self, dim, num_heads=4, qkv_bias=True, attn_p=0., proj_p=0.):
    super().__init__()
    self.num_heads = num_heads
    self.dim = dim
    self.head_dim = dim // num_heads
    self.scale = self.head_dim ** -0.5

    self.qkv_layer = tf.keras.layers.Dense(dim*3, use_bias=qkv_bias, input_shape=(None, dim))
    
    if attn_p > 0.0:
      self.attn_drop_layer = tf.keras.layers.Dropout(attn_p)
    else:
      self.attn_drop_layer = tf.keras.layers.Identity()
    
    self.proj_layer = tf.keras.layers.Dense(dim, input_shape=(None, dim))

    if proj_p > 0.0:
      self.proj_drop_layer = tf.keras.layers.Dropout(proj_p)
    else:
      self.proj_drop_layer = tf.keras.layers.Identity()
      

  def call(self, input):
    '''
      Args:
          input: Tensor whith the shape `(batch_size, num_patches, dim)`.

      Returns
          output: Tensor with the shape `(batch_size, num_patches, dim)`.
    '''

    b, n, c = input.shape
    qkv = self.qkv_layer(input)  # (batch_size, num_patches, dim*3)
    qkv = tf.reshape(qkv, (b, n, 3, self.num_heads, self.head_dim)) # (batch_size, num_patches, 3, num_heads, head_dim)
    qkv = tf.transpose(qkv, (2, 0, 3, 1, 4)) # (3, batch_size, num_heads, num_patches, head_dim)
    q, k, v = qkv[0], qkv[1], qkv[2] 

    # Scaled-dot product
    x = tf.matmul(q, k, transpose_b=True) * self.scale # (batch_size, num_heads, num_patches, num_patches)

    # Calc attention weight 
    x = tf.nn.softmax(x, axis=-1)

    # Drop out applied to attention weight
    x = self.attn_drop_layer(x)
    
    # Attention pooling
    x = tf.matmul(x, v) # (batch_size, num_heads, num_patches, head_dim)
    x = tf.transpose(x, (0, 2, 1, 3))
    x = tf.reshape(x, (b,n,c))  # (batch_size, num_patches, dim)
    
    x = self.proj_layer(x)  # (batch_size, num_patches, dim)
    x = self.proj_drop_layer(x)

    return x

class CrossAttention_layer(layers.Layer):
  '''Cross Attention.
    Args
    ----------
      que_dim : Int. The input and out dimension of query.
      key_dim : Int. The input dimension of key.
      num_heads: : Int. Number of attention heads.
      qkv_bias: Boolean, whether the dense layers use bias vectors/matrices in MultiHeadAttention.
      attn_p : Float. Dropout probability applied to the query, key and value tensors.
      proj_p : Float. Dropout probability applied to the output tensor.
  '''
  
  def __init__(self, que_dim, key_dim, num_heads=4, qkv_bias=True, attn_p=0., proj_p=0.):
    super().__init__()
    self.num_heads = num_heads
    self.que_dim = que_dim
    self.key_dim = key_dim
    self.head_dim = que_dim // num_heads
    self.scale = self.head_dim ** -0.5

    self.q_layer = tf.keras.layers.Dense(que_dim, use_bias=qkv_bias, input_shape=(None, que_dim))
    self.k_layer = tf.keras.layers.Dense(que_dim, use_bias=qkv_bias, input_shape=(None, key_dim))
    self.v_layer = tf.keras.layers.Dense(que_dim, use_bias=qkv_bias, input_shape=(None, key_dim))

    if attn_p > 0.0:
      self.attn_drop_layer = tf.keras.layers.Dropout(attn_p)
    else:
      self.attn_drop_layer = tf.keras.layers.Identity()

    self.proj_layer = tf.keras.layers.Dense(que_dim, input_shape=(None, que_dim))

    if proj_p > 0.0:
      self.proj_drop_layer = tf.keras.layers.Dropout(proj_p)
    else:
      self.proj_drop_layer = tf.keras.layers.Identity()
    
  def call(self, input, embedding):
    '''
      Args:
          input: query. Tensor whith the shape `(batch_size, num_patches, que_dim)`.
          embedding: key. Tensor whith the shape `(batch_size, num_patches, key_dim)`.

      Returns
          output: Tensor with the shape `(batch_size, num_patches, dim)`.
    '''
    
    b, n, c = input.shape
    b, e_n, e_c = embedding.shape

    
    q = self.q_layer(input)  # (batch_size, num_patches, que_dim)
    q = tf.reshape(q,(b, n, self.num_heads, self.head_dim))
    q = tf.transpose(q, (0, 2, 1, 3)) # (batch_size, num_heads, num_patches, head_dim)
    
    k = self.k_layer(embedding)  # (batch_size, num_patches, key_dim)
    k = tf.reshape(k,(b, e_n, self.num_heads, self.head_dim))
    k = tf.transpose(k, (0, 2, 1, 3)) # (batch_size, num_heads, num_embed, head_dim)
    
    v = self.k_layer(embedding)  # (batch_size, num_patches, key_dim)
    v = tf.reshape(v,(b, e_n, self.num_heads, self.head_dim))
    v = tf.transpose(v, (0, 2, 1, 3)) # (batch_size, num_heads, num_embed, head_dim)

    # Scaled-dot product
    x = tf.matmul(q, k, transpose_b=True) * self.scale # (batch_size, num_heads, num_patches, num_embed)

    # Calc attention weight 
    x = tf.nn.softmax(x, axis=-1)

    # Drop out applied to attention weight
    x = self.attn_drop_layer(x)
    
    # Attention pooling
    x = tf.matmul(x, v) # (batch_size, num_heads, num_patches, head_dim)
    x = tf.transpose(x, (0, 2, 1, 3))
    x = tf.reshape(x, (b, n, self.que_dim))  # (batch_size, num_patches, que_dim)
    
    x = self.proj_layer(x)  # (batch_size, num_patches, que_dim)
    x = self.proj_drop_layer(x)

    return input + x

class WindowPartition_layer(layers.Layer):
  """
  Args:
      window_size (int): window size
  Call Args:
      x: (B, H, W, C)
  Returns:
      windows: (num_windows*B, window_size*window_size, C)
  """
  def __init__(self, window_size):
    super().__init__()
    self.window_size = window_size

  def call(self,x):
    """
      Args:
        x: (B, H, W, C)
      Returns:
        windows: (num_windows*B, window_size*window_size, C)
    """
    B, H, W, C = x.shape
    x = tf.reshape(x, (B, H//self.window_size, self.window_size, W//self.window_size, self.window_size, C))
    x = tf.transpose(x,(0, 1, 3, 2, 4, 5))
    x = tf.reshape(x, (-1, self.window_size, self.window_size, C))
    x = tf.reshape(x, (-1, self.window_size*self.window_size, C))
    return x

class WindowReverse_layer(layers.Layer):
  def __init__(self, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size*window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    super().__init__()
    self.window_size = window_size
    self.H = H
    self.W = W

  def call(self, x):
    """
    Args:
      x: (num_windows*B, window_size*window_size, C)
    """
    C = x.shape[-1]
    num_windows = (self.H//self.window_size)*(self.W//self.window_size)
    B = x.shape[0] // num_windows

    x = tf.reshape(x, (-1, self.window_size, self.window_size, C))
    x = tf.reshape(x, (B, self.H//self.window_size, self.W//self.window_size, self.window_size, self.window_size, -1))
    x = tf.transpose(x, (0, 1, 3, 2, 4, 5))
    x = tf.reshape(x, (B, self.H, self.W, -1))
    return x

class MLP_layer(layers.Layer):
  r"""Implement multilayer perceptron (MLP)

    Args:
      hiddden_units: List of output dimension for each dense layer.
      activation: String. Activation function to use.
      dropout_rate: Float. Dropout rate.

  """
  def __init__(self, hidden_units, activation='gelu', dropout_rate=0.):
    super().__init__()
    self.dropout_rate = dropout_rate
    self.dense_layers = []

    for units in hidden_units:
      self.dense_layers.append(layers.Dense(units, activation=activation))
      if dropout_rate > 0.0:
        self.dense_layers.append(layers.Dropout(dropout_rate))
    
  def call(self, x):
    for layer in self.dense_layers:
        x = layer(x)
    return x

class MLP_layer_2x(layers.Layer):
  r"""Implement multilayer perceptron (MLP)

    Args:
      hiddden_units: List of output dimension for each dense layer.
      activation: String. Activation function to use.
      dropout_rate: Float. Dropout rate.

  """
  def __init__(self, hidden_units, activation='gelu', dropout_rate=0., in_dim=None, ):
    super().__init__()
    in_dim = in_dim or hidden_units[0]
    self.dropout_rate = dropout_rate
    self.dense_1 = layers.Dense(hidden_units[0], activation=activation, input_shape=(None, in_dim))
    if dropout_rate > 0.0:
      self.drop_1 = layers.Dropout(dropout_rate)
      self.drop_2 = layers.Dropout(dropout_rate)
    else:
      self.drop_1 = layers.Identity()
      self.drop_2 = layers.Identity()
    self.dense_2 = layers.Dense(hidden_units[1], activation=None, input_shape=(None, hidden_units[0]))
    
  def call(self, x):
    x = self.dense_1(x)
    x = self.drop_1(x)
    x = self.dense_2(x)
    x = self.drop_2(x)
    return x

class PixelShuffle_layer(layers.Layer):
  r"""Implementation of the PixelShuffle layer

  PixelShuffle is a technique that can be used for upsampling in deep learning, 
  particularly in computer vision applications.
  It was introduced in a paper titled 
  "Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network" 
  by Shi et al. in 2016.

  Arguments:
    scale : specifies the upsampling factor.
  """

  def __init__(self, scale, kernel_size = 3):
    super().__init__()
    self.scale = scale
    self.kernel_size = kernel_size
  
  def call(self, inputs):
    batch_size, height, width, in_channels = inputs.shape
    out_channels = in_channels // (self.scale**2)

    # Reshape to (batch_size, height, width, scale, scale, out_channels)
    x = tf.reshape(inputs, [batch_size, height, width, self.scale, self.scale, out_channels])

    # Transpose to (batch_size, height*scale, width*scale, out_channels)
    x = tf.transpose(x, [0, 1, 3, 2, 4, 5])
    x = tf.reshape(x, [batch_size, height * self.scale, width * self.scale, out_channels])

    return x

def drop_path(x, drop_prob=0., training=False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = tf.shape(x)
    random_tensor = keep_prob + tf.random.uniform(shape, dtype=x.dtype)
    random_tensor = tf.floor(random_tensor)  # binarize
    output = tf.math.divide(x, keep_prob) * random_tensor
    return output

class DropPath(tf.keras.layers.Layer):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def call(self, x, training=False):
        return drop_path(x, self.drop_prob, training)

class Augmentaion_layer(layers.Layer):
  r"""Augmentaion
  """
  def __init__(self, image_size=[256, 256]):
    super().__init__()
    self.image_size = image_size # The size of output images [heith, withd]
  
  def call(self, inputs):
    batch_size, height, width, in_channels = tf.shape(inputs)

    inputs = tf.cast(inputs, tf.float32)
    
    # Normalize the images to [-1, 1]
    inputs = inputs / 127.5 - 1

    # Standardization that linearly scales each image in image to have mean 0 and variance 1.
    # inputs = tf.image.per_image_standardization(inputs)

    data_augmentation = tf.keras.Sequential(
        [
            layers.Resizing(self.image_size[0], self.image_size[1]),
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(factor=0.02),
            layers.RandomZoom(
                height_factor=0.2, width_factor=0.2
            ),
        ],
        name="data_augmentation",
    )

    return data_augmentation(inputs)
