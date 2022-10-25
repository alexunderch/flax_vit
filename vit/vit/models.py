
import sys
sys.path.append(".")
from functools import partial
import flax.linen as nn
from typing import Dict, List, Optional, Tuple
import jax.numpy as jnp
try:
    from .modules import TransformerEncoderBlock
    from .pos_embeddings import TransformerEmbeddings
except:
    from modules import TransformerEncoderBlock
    from pos_embeddings import TransformerEmbeddings
# https://huggingface.co/flax-community/vit-gpt2/tree/main/vit_gpt2
# https://github.com/google/flax/blob/main/examples/nlp_seq/train.py
# https://github.com/google/flax/blob/main/examples/imagenet/train.py
# https://github.com/google/flax/blob/main/examples/wmt/models.py
class TransformerHead(nn.Module):
    output_dim: int
    training: bool
    dropout_rate: Optional[float] = .1
    use_bias: Optional[bool] = True

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        out = nn.Dropout(rate = self.dropout_rate, deterministic = self.training)(x)
        out = nn.Dense(self.output_dim, use_bias = self.use_bias)(out)
        return out

class VisualTransformer(nn.Module):
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
            TransformerEncoderBlock(**self.block_config.pop("latent_dim")[0]) \
            for _ in range(self.n_blocks)
        ]

        self.head = TransformerHead(output_dim = self.output_dim,
                                    training = self.block_config["training"])

    def __call__(self, x: jnp.ndarray, mask: Optional[jnp.ndarray] = None) -> jnp.ndarray:

        out = self.apply_embedding(x)
        
        out = self.encoder_layers[0](out, mask)
        for layer in self.encoder_layers[1:]:
            out = layer(out, mask)

        out = self.head(out[:, self.cls_index, :])
        return out

    def get_attention_maps(self, x: jnp.ndarray, mask: Optional[jnp.ndarray] = None) -> List[jnp.ndarray]:
        """"""
        # out = self.apply_embedding(x)
        # out = self.encoder_layers[0](out, mask)
        # attention_maps = [self.encoder_layers[0].get_attention_map(out, mask)]
        # for layer in self.encoder_layers[1:]:
        #     out = layer(out, mask)
        #     attention_maps.append(layer.get_attention_map(out, mask))
        # return attention_maps
