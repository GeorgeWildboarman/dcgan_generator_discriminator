import tensorflow as tf
from tensorflow.keras import layers

class Attention():

  '''Attention.
    Args
    ----------
      dim : Int. The input and out dimension of per token features.
      n_heads: : Int. Number of attention heads.
      qkv_bias: Boolean, whether the dense layers use bias vectors/matrices in MultiHeadAttention.
      attn_p : Float. Dropout probability applied to the query, key and value tensors.
      proj_p : Float. Dropout probability applied to the output tensor.
  '''
  
  def __init__(self, dim, n_heads=12, qkv_bias=True, attn_p=0., proj_p=0.):
    super().__init__()
    self.n_heads = n_heads
    self.dim = dim
    self.head_dim = dim // n_heads
    self.scale = self.head_dim ** -0.5

    self.qkv_layer = tf.keras.layers.Dense(dim * 3, use_bias=qkv_bias)
    self.attn_drop_layer = tf.keras.layers.Dropout(attn_p)
    self.proj_layer = tf.keras.layers.Dense(dim)
    self.proj_drop_layer = tf.keras.layers.Dropout(proj_p)
      

  def call(self, images):
    '''
      Args:
          x: Tensor whith the shape `(batch_size, num_patches, dim)`.

      Returns
        Tensor with the shape `(batch_size, num_patches, dim)`.
    '''

    b, n, c = x.shape
    qkv = self.qkv_layer(x)  # (batch_size, num_patches, dim*3)
    qkv = tf.reshape(qkv, (b, n, 3, self.n_heads, self.head_dim)) # (batch_size, num_patches, 3, n_heads, head_dim)
    qkv = tf.transpose(qkv, (2, 0, 3, 1, 4))
    q, k, v = qkv[0], qkv[1], qkv[2]

    k_t = tf.transpose(k, (0, 1, -2, -1)) # (batch_size, n_heads, head_dim, num_patches)
    dp = tf.matmul(q, k_t) * self.scale
    attn = tf.nn.softmax(dp, axis=-1)
    attn = self.attn_drop_layer(attn)
    
    weighted_avg = tf.matmut(attn, v)
    weighted_avg = tf.transpose(weighted_avg, (0, 2, 1, 3))
    weigthed_avg = tf.reshape(weighted_avg, (b,n,c))
    
    x = self.proj(weighted_avg)  # (batch_size, n_patches, dim)
    x = self.proj_drop(x)

    return x

class Block(layers.Layer):
  '''Transformer block.

    Args:
        embed_dim: Int. Embeddinig dimension.
        num_heads: Int. Number of attention heads.
        mlp_ratio: Int, the factor for determination of the hidden dimension size of the MLP module with respect
        to embed_dim.
        mlp_drop: Float. Dropout probability applied to the MLP.
        qkv_bias: Boolean, whether the dense layers use bias vectors/matrices in MultiHeadAttention.
        attn_drop : Float. Dropout probability applied to MultiHeadAttention.
        activation: String. Activation function to use in MLP.

  '''
  def __init__(self, embed_dim, num_heads=4, mlp_ratio=4, mlp_drop=0., qkv_bias=False, attn_drop=0., activation='gelu'):
    super().__init__()

    self.LN1 = layers.LayerNormalization(epsilon=1e-6)

    self.MHA =layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=embed_dim, use_bias=qkv_bias, dropout=attn_drop,
    )

    self.LN2 = layers.LayerNormalization(epsilon=1e-6)

    hidden_units = [embed_dim * mlp_ratio, embed_dim]
    self.MLP = MLP_layer(hidden_units=hidden_units, dropout_rate=mlp_drop, activation='gelu')

  def call(self, encoded_patches):
    # Layer normalization 1.
    x1 = self.LN1(encoded_patches)

    # Create a multi-head attention layer.
    attention_output = self.MHA(x1, x1)
    
    # Skip connection 1.
    x2 = layers.Add()([attention_output, encoded_patches])

    # Layer normalization 2.
    x3 = self.LN2(x2)
    
    # MLP.
    x3 = self.MLP(x3)
    
    # Skip connection 2.
    out = layers.Add()([x3, x2])

    return out

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
      self.dense_layers.append(layers.Dropout(dropout_rate))

  def call(self, x):
    for layer in self.dense_layers:
        x = layer(x)
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

class WindowPartition_layer(layers.Layer):
  """
  Args:
      window_size (int): window size
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

class PatchEmbed_layer(layers.Layer):
  '''Divide an image into patches

    As ViT operates on sequences, you need to tokenize the images into sequences of tokens.
    Each token represents a specific region or patch of the image. You can divide the image 
    into a grid of patches and flatten each patch into a single token. Additionally, 
    add a special "classification" token at the beginning to indicate the task.

    The approach to divide an image into patches is by using a convolutional layer (Conv2D) as part of the tokenization process. 
    This approach can be seen as a form of patch extraction using a sliding window technique.

     Sliding Window Patch Extraction: Use a Conv2D layer with a sliding window approach to extract patches from the image. 
     The Conv2D layer acts as a local feature extractor, similar to how it is used in convolutional neural networks.

      > Configure the Conv2D layer with appropriate kernel size, stride, and padding.
      > The kernel size determines the patch size you want to extract.
      > The stride determines the amount of overlap between patches.
      > Padding can be used to ensure that patches are extracted from the entire image.

    Positional Encoding: As ViT doesn't encode spatial information inherently, you need to include positional encodings for each token. 
    The positional encodings represent the position or order of the tokens within the sequence. You can use sine and cosine functions or 
    learnable embeddings to generate these positional encodings.


  Args:
    image_size: Tuple. Size of the image wiht the shape (Height, Width).
    patch_size: Int. Size of the patch.
    num_patches: the number of patches in one image
    embed_dim: the size of a vector that each patch is projected.
    kernel_size: Int or tuple/list of 2 integers, specifying the height and width of the 2D convolution window.
    
  Returns:
    A tf.tensor with the shape of (batches, num patches, embed dimension)

  '''

  def __init__(self, image_size, patch_size, embed_dim, kernel_size, padding='valid', add_pos=False):
    super().__init__()
    self.embed_dim = embed_dim
    self.num_patches = num_patches = (image_size[0]//patch_size)*(image_size[1]//patch_size)
    self.add_pos = add_pos

    # Split into patches and Project to a vector with embed dimensional. 
    self.projection = tf.keras.layers.Conv2D(filters=embed_dim, kernel_size=kernel_size, strides=patch_size, padding=padding)

    # Position Embed
    if add_pos:
      self.pos_embed = tf.Variable(tf.zeros((1, num_patches, embed_dim), dtype=tf.dtypes.float32))
    
  def call(self, images):
    # batch_size = tf.shape(images)[0]
    batch_size = images.shape[0]
    x = self.projection(images)
    x = tf.reshape(x, (batch_size, -1, self.embed_dim))
    if self.add_pos:
      x = x + self.pos_embed
    return x

class AddPositionEmbed_layer(layers.Layer):
  def __init__(self, num_patches, embed_dim):
    super().__init__()
    self.pos_embed = tf.Variable(tf.zeros((1, num_patches, embed_dim), dtype=tf.dtypes.float32))

  def call(self, x):
    return x + self.pos_embed

class AddCLSToken_layer(layers.Layer):
  def __init__(self, embed_dim):
    super().__init__()
    self.cls_token = tf.Variable(tf.zeros((1, 1, embed_dim), dtype=tf.dtypes.float32))

  def call(self, x):
    batch_size = x.shape[0]
    return tf.concat([tf.tile(self.cls_token, [batch_size, 1, 1]), x], axis=1)


class discriminator(tf.keras.Model):
  '''Discriminator with a Vision Transformer (ViT).
  Building a discriminator with a Vision Transformer (ViT) involves adapting 
  the transformer architecture to perform binary classification tasks on image data.

  In a Vision Transformer (ViT), the primary components include self-attention mechanisms, 
  transformer encoder layers, position embeddings, and classification heads. Each component:

  Self-Attention Mechanisms: Self-attention is a key component in transformers, including ViTs. 
    Self-attention allows the model to capture relationships between different tokens within 
    a sequence. It calculates attention weights for each token based on its relation to other tokens, 
    enabling the model to focus on relevant information during processing.

  Transformer Encoder Layers: ViTs consist of multiple transformer encoder layers stacked on 
    top of each other. Each transformer encoder layer consists of two sub-layers: 
    a multi-head self-attention mechanism and a feed-forward neural network. 
    The self-attention mechanism captures the dependencies between tokens, 
    while the feed-forward network applies non-linear transformations to each token independently.

  Position Embeddings: Position embeddings encode the spatial information of each token within 
    the image sequence. Since ViTs do not have built-in positional information like CNNs, position 
    embeddings are added to the input tokens to indicate their relative positions. Commonly, 
    sine and cosine functions or learnable embeddings are used to generate position embeddings.

  Classification Head: A classification head is added to the output of the ViT model to perform 
    the final task-specific prediction. For binary classification tasks like discrimination, 
    the classification head typically consists of a fully connected layer followed by 
    a sigmoid activation function to produce the binary classification output.
  
    Args:
      image_size: Turple.
      depth: Int. The number of transformer blocks.
      num_classes: Int. The number of classes.
      embed_dim: Int. Embeddinig dimension.
      patch_siz: Int.
      mlp_drop: Float. Dropout probability applied to the MLP.
      num_heads: Int. Number of attention heads.
      mlp_ratio: Int, the factor for determination of the hidden dimension size of the MLP module with respect to embed_dim.
      attn_drop : Float. Dropout probability applied to MultiHeadAttention.
      window_size: Int, grid size for grid transformer block.
        
        
  '''

  def __init__(
      self,
      image_size = (64, 64),
      depth = 3,
      num_classes = 1,
      embed_dim = 384,
      patch_size = 2,
      mlp_drop = 0.,
      num_heads = 4,
      mlp_ratio = 4,
      attn_drop = 0.,
      window_size = 4,
      ):
    
    super().__init__()

    self.patch_size = patch_size

    self.embed_dim = embed_dim
    self.embed_dim_1 = embed_dim_1 = embed_dim//4
    self.embed_dim_2 = embed_dim_2 = embed_dim//4
    self.embed_dim_3 = embed_dim_3 = embed_dim//2

    self.patch_size_1 = patch_size_1 = patch_size
    self.patch_size_2 = patch_size_2 = patch_size*2
    self.patch_size_3 = patch_size_3 = patch_size*4

    self.patches_1 = PatchEmbed_layer(image_size, patch_size = patch_size_1, embed_dim = embed_dim_1, kernel_size = patch_size_1*2, padding='same')
    self.patches_2 = PatchEmbed_layer(image_size, patch_size = patch_size_2, embed_dim = embed_dim_2, kernel_size = patch_size_2, padding='valid')
    self.patches_3 = PatchEmbed_layer(image_size, patch_size = patch_size_3, embed_dim = embed_dim_3, kernel_size = patch_size_3, padding='valid')

    num_patches_1 = self.patches_1.num_patches
    num_patches_2 = self.patches_2.num_patches
    num_patches_3 = self.patches_3.num_patches

    self.add_pos_embed_1 = AddPositionEmbed_layer(num_patches_1, embed_dim_1)
    self.add_pos_embed_2 = AddPositionEmbed_layer(num_patches_2, embed_dim_2)
    self.add_pos_embed_3 = AddPositionEmbed_layer(num_patches_3, embed_dim_3)

    self.window_size = window_size

    self.blocks_1 = [
        Block(
            embed_dim=embed_dim_1, num_heads=num_heads, mlp_ratio=mlp_ratio, mlp_drop=mlp_drop, qkv_bias=False, attn_drop=attn_drop, activation='gelu'
        ) for _ in range(depth)
    ]
    
    self.blocks_2 = [
        Block(
            embed_dim=embed_dim_1+embed_dim_2, num_heads=num_heads, mlp_ratio=mlp_ratio, mlp_drop=mlp_drop, qkv_bias=False, attn_drop=attn_drop, activation='gelu'
        ) for _ in range(depth-1)
    ]
    
    self.blocks_21 = [
        Block(
            embed_dim=embed_dim_1+embed_dim_2, num_heads=num_heads, mlp_ratio=mlp_ratio, mlp_drop=mlp_drop, qkv_bias=False, attn_drop=attn_drop, activation='gelu'
        ) for _ in range(1)
    ]
    
    self.blocks_3 = [
        Block(
            embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, mlp_drop=mlp_drop, qkv_bias=False, attn_drop=attn_drop, activation='gelu'
        ) for _ in range(depth)
    ]


    self.add_cls_token = AddCLSToken_layer(embed_dim=embed_dim)

    self.blocks_last = [
        Block(
            embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, mlp_drop=mlp_drop, qkv_bias=False, attn_drop=attn_drop, activation='gelue'
        )
    ]

    self.layer_norm = layers.LayerNormalization(epsilon=1e-6)

    self.head = layers.Dense(num_classes)

  def call(self, input):
    batch_size, height, width, channels = input.shape
    h = height//self.patch_size_1
    w = width//self.patch_size_1

    x_1 = self.patches_1(input)
    x_2 = self.patches_2(input)
    x_3 = self.patches_3(input)

    # B, H//2*W//2, embed_dim//4 if patch_size = 2
    x = self.add_pos_embed_1(x_1)
    c = x.shape[-1]

    # -- Grid Transformer Block --
    # B, H//2, W//2, embed_dim//4 if patch_size = 2
    x = layers.Reshape((h, w, c))(x)
    x = WindowPartition_layer(self.window_size)(x)
    for block in self.blocks_1:
      x = block(x)
    x = WindowReverse_layer(self.window_size, h, w)(x)

    # -- 2x AvePool --
    # B, H//4, W//4, embed_dim//4
    x = layers.AveragePooling2D(2)(x)
    _, h, w, c = x.shape

    #  -- Concatnate --
    # B, H//4*W//4, embed_dim//4
    x = layers.Reshape((-1, c))(x)
    # B, H//4*W//4, embed_dim//2
    x = layers.Concatenate(axis=-1)([x, x_2])
    c = x.shape[-1]

    # -- Grid Transformer Block --
    x = layers.Reshape((h, w, c))(x)
    x = WindowPartition_layer(self.window_size)(x)
    for block in self.blocks_2:
      x = block(x)
    x = WindowReverse_layer(self.window_size, h, w)(x)    
    # -- Transformer Block --
    x = layers.Reshape((h*w, c))(x)
    for block in self.blocks_21:
      x = block(x)

    # -- 2x AvePool --
    x = layers.Reshape((h, w, c))(x)
    # B, H//8, W//8, embed_dim//2
    x = layers.AveragePooling2D(2)(x)
    _, h, w, c = x.shape

    #  -- Concatnate --
    x = layers.Reshape((-1, c))(x)
    # B, H//8*W//8, embed_dim
    x = layers.Concatenate(axis=-1)([x, x_3])
    c = x.shape[-1]

    # -- Transformer Block --
    for block in self.blocks_3:
      x = block(x)

    # -- Add CLS token --
    # B, H//8*W//8+1, embed_dim
    x = self.add_cls_token(x)

    # -- Transformer Block --
    for block in self.blocks_last:
      x = block(x)

    x = self.layer_norm(x)
    x = self.head(x[:,0])

    return x
