import flax.linen as nn
import jax.numpy as jnp
import jax.random as jr
import jax

class PatchEmbeddings(nn.Module):

    image_size: int
    patch_size: int
    latent_dim: int
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.num_patches =  (self.image_size // self.patch_size) **2
        self.projection = nn.Conv(
            self.latent_dim,
            kernel_size=(self.patch_size, self.patch_size),
            strides=(self.patch_size, self.patch_size),
            padding="VALID",
            dtype=self.dtype,
        )

    def __call__(self, pixel_values):
        x = self.projection(pixel_values)
        batch_size, _, _, channels = x.shape
        return jnp.reshape(x, (batch_size, -1, channels))


class ViTEmbeddings(nn.Module):
    """Construct the CLS token, position and patch embeddings."""

    dropout_rate: float
    latent_dim: int
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.cls_token = self.param("cls_token", nn.initializers.zeros, (1, 1, self.latent_dim))
        self.patch_embeddings = PatchEmbeddings(self.latent_dim, dtype=self.dtype)
        num_patches = self.patch_embeddings.num_patches
        self.position_embeddings = self.param(
            "position_embeddings", nn.initializers.zeros, (1, num_patches + 1, self.latent_dim)
        )
        self.dropout = nn.Dropout(rate=self.dropout_rate)

    def __call__(self, pixel_values, deterministic=True):
        batch_size = pixel_values.shape[0]

        embeddings = self.patch_embeddings(pixel_values)

        cls_tokens = jnp.broadcast_to(self.cls_token, (batch_size, 1, self.latent_dim))
        embeddings = jnp.concatenate((cls_tokens, embeddings), axis=1)
        embeddings = embeddings + self.position_embeddings
        embeddings = self.dropout(embeddings, deterministic=deterministic)
        return embeddings


class OneDPositionalEmbedding(nn.Module):
    """"""

class TwoDDPositionalEmbedding(nn.Module):
    """"""