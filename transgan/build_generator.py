import tensorflow as tf
from tensorflow.keras import layers

from transgan.vit_helper import MLP_layer, WindowPartition_layer, WindowReverse_layer, SelfAttention_layer, CrossAttention_layer
  
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


