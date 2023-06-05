import tensorflow as tf
from tensorflow.keras import layers

from transgan.vit_helper import MLP_layer, WindowPartition_layer, WindowReverse_layer, SelfAttention_layer, CrossAttention_layer
  
class BlockGridTrans(layers.Layer):
  '''Transformer block for generator.

    Args:
        dim: Int. 
        embed_dim: Int. Embeddinig dimension.
        num_heads: Int. Number of attention heads.
        mlp_ratio: Int, the factor for determination of the hidden dimension size of the MLP module with respect
        to embed_dim.
        mlp_p: Float. Dropout probability applied to the MLP.
        qkv_bias: Boolean, whether the dense layers use bias vectors/matrices in Attention.
        attn_p : Float. Dropout probability applied to Attention.
        activation: String. Activation function to use in MLP.
        window_size: Int, grid size for grid transformer block.

  '''
  def __init__(self, que_dim, key_dim, num_heads=4, mlp_ratio=4, mlp_p=0., qkv_bias=False, attn_p=0., proj_p=0., activation='gelu', window_size=16):
    super().__init__()

    self.window_size = window_size
    self.norm1 = layers.LayerNormalization(epsilon=1e-6)
    self.norm2 = layers.LayerNormalization(epsilon=1e-6)
    hidden_units = [que_dim * mlp_ratio, que_dim]
    self.mlp = MLP_layer(hidden_units=hidden_units, dropout_rate=mlp_p, activation=activation)
 
    self.cross_attention = CrossAttention_layer(que_dim=que_dim, key_dim=key_dim, num_heads=num_heads, qkv_bias=qkv_bias)
    self.attention = SelfAttention_layer(dim=que_dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_p=attn_p, proj_p=proj_p)

  def call(self, inputs):
    x, embedding = inputs

    x = self.cross_attention(x, embedding)
    
    B, N, C = x.shape
    H = W = int(np.sqrt(N))

    x = layers.Reshape((H, W, C))(x)
    x = WindowPartition_layer(self.window_size)(x)

    x = x + self.attention(self.norm1(x))

    x = WindowReverse_layer(self.window_size, H, W)(x)
    x = layers.Reshape((N, C))(x)

    x = x + self.mlp(self.norm2(x))

    return [x, embedding]

class Block(layers.Layer):
  '''Transformer block for generator.

    Args:
        que_dim: Int. Embeddinig dimension.
        num_heads: Int. Number of attention heads.
        mlp_ratio: Int, the factor for determination of the hidden dimension size of the MLP module with respect
        to embed_dim.
        mlp_p: Float. Dropout probability applied to the MLP.
        qkv_bias: Boolean, whether the dense layers use bias vectors/matrices in Attention.
        attn_p : Float. Dropout probability applied to Attention.
        activation: String. Activation function to use in MLP.

  '''
  def __init__(self, que_dim, num_heads=4, mlp_ratio=4, mlp_p=0., qkv_bias=False, attn_p=0., proj_p=0., activation='gelu'):
    super().__init__()
    self.norm1 = layers.LayerNormalization(epsilon=1e-6)
    self.norm2 = layers.LayerNormalization(epsilon=1e-6)
    self.attention = SelfAttention_layer(dim=que_dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_p=attn_p, proj_p=proj_p)
    hidden_units = [que_dim * mlp_ratio, que_dim]
    self.mlp = MLP_layer(hidden_units=hidden_units, dropout_rate=mlp_p, activation=activation)
 
  def call(self, x):
    x = x + self.attention(self.norm1(x))
    x = x + self.mlp(self.norm2(x))
    return x
    
class StageBlock(layers.Layer):
  '''Transformer block Stage for generator.

    Args:
        depth: Int. The number of transformer blocks in the stage.
        que_dim: Int. Embeddinig dimension.
        num_heads: Int. Number of attention heads.
        mlp_ratio: Int, the factor for determination of the hidden dimension size of the MLP module with respect
        to embed_dim.
        mlp_p: Float. Dropout probability applied to the MLP.
        qkv_bias: Boolean, whether the dense layers use bias vectors/matrices in Attention.
        attn_p : Float. Dropout probability applied to Attention.
        activation: String. Activation function to use in MLP.

  '''

  def __init__(self, depth, que_dim, num_heads=4, mlp_ratio=4, mlp_p=0., qkv_bias=False, attn_p=0., proj_p=0., activation='gelu'):
    super().__init__()


    self.blocks = [
        Block(
            que_dim=que_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, mlp_p=mlp_p, qkv_bias=False, attn_p=attn_p, activation='gelu'
        ) for _ in range(depth)
    ]

  def call(self, x):
    for block in self.blocks:
      x = block(x)
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

class Generator(tf.keras.Model):

  r'''Generator with transformer blocks.

    Args:
        depth: Int. The number of transformer blocks in the stage.
        que_dim: Int. Embeddinig dimension.
        num_heads: Int. Number of attention heads.
        mlp_ratio: Int, the factor for determination of the hidden dimension size of the MLP module with respect
        to embed_dim.
        mlp_p: Float. Dropout probability applied to the MLP.
        qkv_bias: Boolean, whether the dense layers use bias vectors/matrices in Attention.
        attn_p : Float. Dropout probability applied to Attention.
        activation: String. Activation function to use in MLP.

  '''
  def __init__(self, latent_dim=100, depth=[5,4,2], que_dim=1024, num_heads=4, mlp_ratio=4, mlp_p=0., qkv_bias=False, attn_p=0., proj_p=0., activation='gelu', bottom_width=8):
    super().__init__()

    self.latent_dim = latent_dim
    self.bottom_width = bottom_width
    self.embed_dim = embed_dim = que_dim
    self.dence_1 = layers.Dense((self.bottom_width ** 2) * self.embed_dim, input_shape=(latent_dim,))

    self.pos_embed_1 = tf.Variable(tf.zeros((1, bottom_width**2, embed_dim), dtype=tf.dtypes.float32))
    self.pos_embed_2 = tf.Variable(tf.zeros((1, (bottom_width*2)**2, embed_dim//4), dtype=tf.dtypes.float32))
    self.pos_embed_3 = tf.Variable(tf.zeros((1, (bottom_width*4)**2, embed_dim//16), dtype=tf.dtypes.float32))
    self.pos_embed = [
        self.pos_embed_1,
        self.pos_embed_2,
        self.pos_embed_3
    ]

    self.blocks = StageBlock(
        depth=depth[0],
        que_dim = que_dim,
        num_heads = num_heads,
        mlp_ratio = mlp_ratio,
        mlp_p = mlp_p,
        qkv_bias = qkv_bias,
        attn_p = attn_p,
        activation = activation
    )

    self.upsample_blocks = [
        StageBlock(
            depth=depth[1],
            que_dim = que_dim//4,
            num_heads = num_heads,
            mlp_ratio = mlp_ratio,
            mlp_p = mlp_p,
            qkv_bias = qkv_bias,
            attn_p = attn_p,
            activation = activation
        ),
        StageBlock(
            depth=depth[2],
            que_dim = que_dim//16,
            num_heads = num_heads,
            mlp_ratio = mlp_ratio,
            mlp_p = mlp_p,
            qkv_bias = qkv_bias,
            attn_p = attn_p,
            activation = activation
        )
    ]

    self.conv_1 = layers.Conv2D(3, 1, 1, padding='valid', activation='tanh')

  def call(self, latent_vector):
    batch_size = latent_vector.shape[0]
    H = self.bottom_width
    W = self.bottom_width
    C = self.embed_dim

    x = tf.reshape(self.dence_1(latent_vector), (-1, H*W, C))
    x = x + self.pos_embed[0]
    x = self.blocks(x)

    for i, block in enumerate(self.upsample_blocks):
      x = tf.reshape(x, (batch_size, H, W, C))
      x = PixelShuffle_layer(2)(x)
      _, H, W, C = x.shape
      x = tf.reshape(x, (batch_size, -1, C))
      x = x + self.pos_embed[i+1]
      x = block(x)

    x = tf.reshape(x, (batch_size, H, W, C))
    x = self.conv_1(x)

    return x
