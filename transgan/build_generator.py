import tensorflow as tf
from tensorflow.keras import layers

from transgan.vit_helper import MLP_layer, WindowPartition_layer, WindowReverse_layer

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
  
  def __init__(self, dim, num_heads=12, qkv_bias=True, attn_p=0., proj_p=0.):
    super().__init__()
    self.num_heads = num_heads
    self.dim = dim
    self.head_dim = dim // num_heads
    self.scale = self.head_dim ** -0.5

    self.qkv_layer = tf.keras.layers.Dense(dim*3, use_bias=qkv_bias, input_shape=(None, dim))
    self.attn_drop_layer = tf.keras.layers.Dropout(attn_p)
    self.proj_layer = tf.keras.layers.Dense(dim)
    self.proj_drop_layer = tf.keras.layers.Dropout(proj_p)
      

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
    dp = tf.matmul(q, k, transpose_b=True) * self.scale # (batch_size, num_heads, num_patches, num_patches)

    # Calc attention weight 
    attention_weight = tf.nn.softmax(dp, axis=-1)

    # Drop out applied to attention weight
    attention_weight = self.attn_drop_layer(attention_weight)
    
    # Attention pooling
    attention_ouptput = tf.matmul(attention_weight, v) # (batch_size, num_heads, num_patches, head_dim)
    attention_ouptput = tf.transpose(attention_ouptput, (0, 2, 1, 3))
    attention_ouptput = tf.reshape(attention_ouptput, (b,n,c))  # (batch_size, num_patches, dim)
    
    output = self.proj_layer(attention_ouptput)  # (batch_size, num_patches, dim)
    output = self.proj_drop_layer(output)

    return output

class CrossAttention_layer(layers.Layer):
  '''Cross Attention.
    Args
    ----------
      que_dim : Int. The input and out dimension of query.
      key_dim : Int. The input and out dimension of key.
      num_heads: : Int. Number of attention heads.
      qkv_bias: Boolean, whether the dense layers use bias vectors/matrices in MultiHeadAttention.
      attn_p : Float. Dropout probability applied to the query, key and value tensors.
      proj_p : Float. Dropout probability applied to the output tensor.
  '''
  
  def __init__(self, que_dim, key_dim, num_heads=12, qkv_bias=True, attn_p=0., proj_p=0.):
    super().__init__()
    self.num_heads = num_heads
    self.que_dim = que_dim
    self.key_dim = key_dim
    self.head_dim = que_dim // num_heads
    self.scale = self.head_dim ** -0.5

    self.q_layer = tf.keras.layers.Dense(que_dim, use_bias=qkv_bias, input_shape=(None, que_dim))
    self.k_layer = tf.keras.layers.Dense(que_dim, use_bias=qkv_bias, input_shape=(None, key_dim))
    self.v_layer = tf.keras.layers.Dense(que_dim, use_bias=qkv_bias, input_shape=(None, key_dim))

    self.attn_drop_layer = tf.keras.layers.Dropout(attn_p)
    self.proj_layer = tf.keras.layers.Dense(que_dim, input_shape=(None, que_dim))
    self.proj_drop_layer = tf.keras.layers.Dropout(proj_p)
      
  def call(self, input, embedding):
    '''
      Args:
          input: Tensor whith the shape `(batch_size, num_patches, que_dim)`.
          embedding: Tensor whith the shape `(batch_size, num_patches, key_dim)`.

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
    dp = tf.matmul(q, k, transpose_b=True) * self.scale # (batch_size, num_heads, num_patches, num_embed)

    # Calc attention weight 
    attention_weight = tf.nn.softmax(dp, axis=-1)

    # Drop out applied to attention weight
    attention_weight = self.attn_drop_layer(attention_weight)
    
    # Attention pooling
    attention_ouptput = tf.matmul(attention_weight, v) # (batch_size, num_heads, num_patches, head_dim)
    attention_ouptput = tf.transpose(attention_ouptput, (0, 2, 1, 3))
    attention_ouptput = tf.reshape(attention_ouptput, (b, n, self.que_dim))  # (batch_size, num_patches, que_dim)
    
    output = self.proj_layer(attention_ouptput)  # (batch_size, num_patches, que_dim)
    output = self.proj_drop_layer(output)

    return input + output
  
  class Block(layers.Layer):
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
    hidden_units = [embed_dim * mlp_ratio, embed_dim]
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


