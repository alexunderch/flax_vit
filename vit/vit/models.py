from functools import partial
import flax.linen as nn
from typing import Dict, Optional, Tuple
import jax.numpy as jnp
from modules import TransformerEncoderBlock
from pos_embeddings import TransformerEmbeddings

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
    dropout_embedding: float
    img_params: Tuple[int, int]
    cls_index: int = 0

    def setup(self) -> None:

        self.apply_embedding = TransformerEmbeddings(
            dropout_rate = self.dropout_embedding,
            latent_dim = self.block_config["latent_dim"],
            image_size = self.img_params[0],
            patch_size = self.img_params[1],
            training = self.block_config["training"]
        )

        self.encoder_layers = [
            partial(TransformerEncoderBlock, **self.block_config) \
            for _ in self.n_blocks
        ]

        self.head = TransformerHead(output_dim = self.output_dim,
                                    training = self.block_config["training"])

    def __call__(self,  x: jnp.ndarray) -> jnp.ndarray:

        out = self.apply_embedding(x)
        out = self.encoder_layers[0](out)
        for layer in self.encoder_layers[1:]:
            out = layer(out)
        out = self.head(out[:, self.cls_index, :])
        return out
