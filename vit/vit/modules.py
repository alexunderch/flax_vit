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
    def apply(self, x: jnp.ndarray, fn: nn.Module) -> jnp.ndarray:
        return x + fn(x)            

class PreNormLayer(nn.Module):
    def apply(self, x: jnp.ndarray, 
                    fn: nn.Module, 
                    norm: Optional(nn.Module) = nn.LayerNorm) -> jnp.ndarray:
        return fn(norm(x))      

class FeedForwardLayer(nn.Module):
    dropout_rate: float
    latent_dim: int
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
    latent_dim: int
    head_dim: int
    training: bool
    use_bias: Optional[bool] = False
    attention_function: Optional[Callable] = scaled_dot_product
    
    def setup(self) -> None:
        assert 3 * self.latent_dim // self.head_dim

        self.qkv_proj = nn.Dense(3 * self.latent_dim, use_bias = self.use_bias)
        self.o_proj = nn.Dense(self.latent_dim, use_bias = self.use_bias)
        self.o_dropout = nn.Dropout(self.dropout_rate, deterministic = self.training)

    def __call__(self, x: jnp.ndarray, mask: Optional[jnp.ndarray]) -> jnp.ndarray:
        batch_size, seq_length, hidden_dim = x.shape
        qkv = self.qkv_proj(x)
        qkv = jnp.swapaxes(qkv.reshape(batch_size, seq_length, self.head_dim, -1), 1, 2)
        q, k, v = jnp.array_split(qkv, 3, axis = -1)
        values, attention = self.attention_function(q, k, v, mask = mask)

        values = jnp.swapaxes(values, 1, 2)
        values = self.o_proj(self.o_dropout(values)).reshape(batch_size, seq_length, hidden_dim)
        return values, attention
        

class TransformerEncoderBlock(nn.Module):
    latent_dim: int
    head_dim: int
    training: bool
    dropout_rate_ffd: float
    dropout_rate_att: float
    use_bias_att: Optional[bool] = False
    use_bias_ffd: Optional[bool] = True
    attention_function: Optional[Callable] = scaled_dot_product
    
    def setup(self) -> None:
        self_attn = partial(MultiHeadSelfAttentionLayer, 
                            dropout_rate = self.dropout_rate_att,
                            head_dim = self.head_dim,
                            training = self.training,
                            use_bias = self.use_bias_att,
                            attention_function = self.attention_function)

        norm_attention = partial(PreNormLayer,
                                fn = self_attn,
                                norm  = nn.LayerNorm)

        self.residual_norm_attention = partial(ResidualWrapper, 
                                               fn = norm_attention)
        
        ffd = partial(FeedForwardLayer, 
                        dropout_rate = self.dropout_rate_ffd,
                        activation = nn.gelu,
                        training = self.training,
                        use_bias = self.use_bias_ffd
                     )

        norm_ffd = partial(PreNormLayer,
                            fn = ffd,
                            norm  = nn.LayerNorm)

        self.residual_norm_ffd = partial(ResidualWrapper, 
                                        fn = norm_ffd)

    def __call__(self, x: jnp.ndarray) ->  jnp.ndarray:
        return self.residual_norm_ffd(
            self.residual_norm_attention(x)
        )

        