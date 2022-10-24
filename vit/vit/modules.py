from typing import Callable, Optional, Any, Tuple 
import flax.linen as nn
import jax.numpy as jnp
import jax.random as jr
import jax
from functools import partial


def scaled_dot_product(q, k, v, mask=None):
    d_k = q.shape[-1]
    attn_logits = jnp.matmul(q, jnp.swapaxes(k, -2, -1))
    attn_logits = attn_logits / jnp.sqrt(d_k)
    if mask is not None:
        attn_logits = jnp.where(mask == 0, -jnp.inf, attn_logits)
    attention = nn.softmax(attn_logits, axis = -1)
    values = jnp.matmul(attention, v)
    return values, attention


class ResidualWrapper(nn.Module):
    fn: nn.Module
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return x + self.fn(x)     

class PreNormLayer(nn.Module):
    fn: nn.Module
    norm: Optional[nn.Module] = nn.LayerNorm

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        if isinstance(x, Tuple):
            return self.fn(self.norm(x[0])), x[1]  
        return self.fn(self.norm(x))      

class FeedForwardLayer(nn.Module):
    latent_dim: int
    dropout_rate: float
    activation: nn.activation
    training: bool
    use_bias: Optional[bool] = True

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        feature_dim = x.shape[-1]
        out = nn.Dense(self.latent_dim, use_bias = self.use_bias)(x)
        out = nn.Dropout(rate = self.dropout_rate, deterministic = self.training)(out)
        out = self.activation(out)
        out = nn.Dense(feature_dim, use_bias = self.use_bias)(out)
        return out



class MultiHeadSelfAttentionLayer(nn.Module):
    dropout_rate: float
    n_heads: int 
    training: bool
    use_bias: Optional[bool] = False
    attention_function: Optional[Callable] = scaled_dot_product


    def prepare_qkv(self, x: jnp.ndarray):
        batch_size, seq_length, hidden_dim = x.shape
        head_dim = hidden_dim // self.n_heads
        assert head_dim > 0
        qkv = nn.Dense(3 * hidden_dim, use_bias = self.use_bias)(x)
        qkv = jnp.swapaxes(qkv.reshape(batch_size, seq_length, head_dim, -1), 1, 2)
        q, k, v = jnp.array_split(qkv, 3, axis = -1)
        return q, k, v

    @nn.compact
    def __call__(self, x: jnp.ndarray, mask: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        batch_size, seq_length, hidden_dim = x.shape
        o_dropout = nn.Dropout(self.dropout_rate, deterministic = self.training)

        q, k, v = self.prepare_qkv(x)
        values, _ = self.attention_function(q, k, v, mask = mask)
        values = jnp.swapaxes(values, 1, 2)
        values =  o_dropout(values).reshape(batch_size, seq_length, hidden_dim)
        values = nn.Dense(hidden_dim, use_bias = self.use_bias)(values)
                           
        return values

    def get_attention_map(self, x, mask=None):
        q, k, v = self.prepare_qkv(x)
        return self.attention_function(q, k, v, mask = mask)[1]

        

class TransformerEncoderBlock(nn.Module):
    n_heads: int
    training: bool
    latent_ffd_dim: int
    dropout_rate_ffd: float
    dropout_rate_att: float
    use_bias_att: Optional[bool] = False
    use_bias_ffd: Optional[bool] = True
    attention_function: Optional[Callable] = scaled_dot_product
    
    def setup(self) -> None:
        self_attn = MultiHeadSelfAttentionLayer(
                            **dict(dropout_rate = self.dropout_rate_att,
                                    n_heads = self.n_heads,
                                    training = self.training,
                                    use_bias = self.use_bias_att,
                                    attention_function = self.attention_function)
                            )

        norm_attention = PreNormLayer(fn = self_attn,
                                      norm  = nn.LayerNorm())

        self.residual_norm_attention = ResidualWrapper(fn = norm_attention)
        
        ffd = FeedForwardLayer(
                                latent_dim= self.latent_ffd_dim,
                                dropout_rate = self.dropout_rate_ffd,
                                activation = nn.gelu,
                                training = self.training,
                                use_bias = self.use_bias_ffd
                            )

        norm_ffd = PreNormLayer(fn = ffd,
                                norm  = nn.LayerNorm())

        self.residual_norm_ffd = ResidualWrapper(fn = norm_ffd)

    def __call__(self, x: jnp.ndarray) ->  jnp.ndarray:
        return self.residual_norm_ffd(
            self.residual_norm_attention(x)
        )

        