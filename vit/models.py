from functools import partial
import flax.linen as nn
from typing import Dict, Optional
import jax.numpy as jnp
from modules import TransformerEncoderBlock
from pos_embeddings import OneDPositionalEmbedding, TwoDDPositionalEmbedding

class TransformerHead(nn.Module):
    output_dim: int
    training: bool
    dropout_rate: Optional[float] = .1
    use_bias: Optional[bool] = True

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        out = nn.Dropout(rate = self.dropout_rate, deterministic = self.training)(out)
        out = nn.Dense(self.output_dim, use_bias = self.use_bias)(out)
        return out

class VisualTransormer(nn.Module):
    n_blocks: int
    block_config: Dict
    output_dim: int
    cls_index: int = 0

    def setup(self) -> None:
        #TODO: pos_embeddings

        self.pos_embedding = OneDPositionalEmbedding

        self.encoder_layers = [
            partial(TransformerEncoderBlock, **self.block_config) \
            for _ in self.n_blocks
        ]

        self.head = partial(TransformerHead,
                            output_dim = self.output_dim,
                            training = self.block_config["training"])


    def __call__(self,  x: jnp.ndarray) -> jnp.ndarray:

        #TODO: images
        out = self.encoder_layers[0](x)
        for layer in self.encoder_layers[1:]:
            out = layer(out)
        out = self.head(out[:, self.cls_index, :])
        return out
