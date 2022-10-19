import flax.linen as nn
import jax.numpy as jnp
import jax.random as jr
import jax

class PatchEmbeddings(nn.Module):

    image_size: int
    patch_size: int
    latent_dim: int
    dtype: jnp.dtype = jnp.float32  

    def setup(self):
        self.num_patches =  (self.image_size // self.patch_size) ** 2
        self.projection = nn.Conv(
            self.latent_dim,
            kernel_size = (self.patch_size, self.patch_size),
            strides = (self.patch_size, self.patch_size),
            padding = "VALID",
            dtype = self.dtype,
        )

    def __call__(self, x):
        x = self.projection(x)
        batch_size, _, _, channels = x.shape
        return jnp.reshape(x, (batch_size, -1, channels))


class TransformerEmbeddings(nn.Module):
    """Construct the CLS token, position and patch embeddings."""

    dropout_rate: float
    latent_dim: int
    image_size: int
    patch_size: int
    training: bool
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.cls_token = self.param("cls_token", 
                                    nn.initializers.zeros, 
                                    (1, 1, self.latent_dim))
        self.patch_embeddings = PatchEmbeddings(self.latent_dim,
                                                image_size = self.image_size,
                                                patch_size= self.patch_size,
                                                dtype = self.dtype)
        num_patches = self.patch_embeddings.num_patches
        self.position_embeddings = self.param(
            "position_embeddings", 
            nn.initializers.zeros, 
            (1, num_patches + 1, self.latent_dim)
        )
        self.dropout = nn.Dropout(rate=self.dropout_rate, deterministic = self.training)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        batch_size = x.shape[0]

        embeddings = self.patch_embeddings(x)

        cls_tokens = jnp.broadcast_to(self.cls_token, (batch_size, 1, self.latent_dim))
        embeddings = jnp.concatenate((cls_tokens, embeddings), axis=1)
        embeddings = embeddings + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings
    