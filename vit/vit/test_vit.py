
from models import VisualTransformer
import jax
import jax.numpy as jnp

def test_transformer():
    rng = jax.random.PRNGKey(0)
    x = jnp.ones(shape = (2, 32, 32, 3))
    model = VisualTransformer(  n_blocks = 3,
                                block_config = {
                                    "latent_dim": 512 ,
                                    "latent_ffd_dim": 1203,
                                    "n_heads": 8 ,
                                    "training": True ,
                                    "dropout_rate_ffd": .1 ,
                                    "dropout_rate_att": .5
                                },
                                output_dim = 2,
                                dropout_embedding = .5,
                                img_params = (32, 16))
    params = model.init(rng, x, mask = None)
    # print(model.get_attention_maps(x)[0].shape)
    print(jax.tree_map(lambda x: x.shape, params))


if __name__ == "__main__":
    test_transformer()
