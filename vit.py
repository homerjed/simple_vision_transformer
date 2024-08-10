from typing import List, Tuple, Optional
import jax
import jax.numpy as jnp
import jax.random as jr 
import equinox as eqx
from jaxtyping import Key, Array, Float
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


class MultiHeadAttentionLayer(eqx.Module):
    embed_dim: int
    n_heads: int
    head_dim: int
    _q: eqx.nn.Linear
    _k: eqx.nn.Linear
    _v: eqx.nn.Linear # Q, K, V projections
    _o: eqx.nn.Linear # Output layer
    dropout: eqx.nn.Dropout

    def __init__(
        self, 
        embed_dim: int, 
        n_heads: int, 
        dropout_rate: float, 
        *, 
        key: Key
    ):
        self.embed_dim = embed_dim # Embed dimension
        self.n_heads = n_heads 
        self.head_dim = int(embed_dim / n_heads) 
        keys = jr.split(key, 4)
        self._q = eqx.nn.Linear(embed_dim, embed_dim, key=keys[0]) # Query 
        self._k = eqx.nn.Linear(embed_dim, embed_dim, key=keys[1]) # Key 
        self._v = eqx.nn.Linear(embed_dim, embed_dim, key=keys[2]) # Value 
        self._o = eqx.nn.Linear(embed_dim, embed_dim, key=keys[3]) # Output 
        self.dropout = eqx.nn.Dropout(dropout_rate)

    def __call__(
        self, 
        query: Float[Array, "pc self.embed_dim"],
        _key: Float[Array, "pc self.embed_dim"], 
        value: Float[Array, "pc self.embed_dim"], 
        *, 
        key: Optional[Key] = None
    ) -> Tuple[Array, Array]: 
 
        Q = jax.vmap(self._q)(query) # [query_len, embed_dim] 
        K = jax.vmap(self._k)(_key) # [key_len, embed_dim]
        V = jax.vmap(self._v)(value) # [value_len, embed_dim]

        # Embed_dim = n_heads * head_dim
        Q = rearrange(Q, 'l (h d) -> h l d', h=self.n_heads) # [n_heads, query_len, head_dim]
        K = rearrange(K, 'l (h d) -> h l d', h=self.n_heads) # [n_heads, key_len, head_dim]
        V = rearrange(V, 'l (h d) -> h l d', h=self.n_heads) # [n_heads, value_len, head_dim]

        # Scaled Dot-Product Attention
        weight = Q @ rearrange(K, 'h l d -> h d l') / jnp.sqrt(self.head_dim) # [n_heads, query_len, key_len] 
        attention = jax.nn.softmax(weight, axis=-1) # [n_heads, query_len, key_len]

        # Class token (ViT regresses with this into output layer)
        c = self.dropout(attention, key=key) @ V # [n_heads, query_len, head_dim] 

        # Reshape & stack 
        c = rearrange(c, 'h l d -> l (h d)') # [query_len, embed_dim] 

        output = jax.vmap(self._o)(c)

        return output, attention # [query_len, embed_dim]


class TokenMLP(eqx.Module):
    linear1: eqx.nn.Linear
    linear2: eqx.nn.Linear
    dropout: eqx.nn.Dropout

    def __init__(
        self, 
        embed_dim: int, 
        hidden_dim: int, 
        dropout_rate: float, 
        *, 
        key: Key
    ):
        keys = jr.split(key)
        self.linear1 = eqx.nn.Linear(embed_dim, hidden_dim, key=keys[0])
        self.linear2 = eqx.nn.Linear(hidden_dim, embed_dim, key=keys[1])
        self.dropout = eqx.nn.Dropout(dropout_rate)

    def __call__(self, x: Array, key: Optional[Key] = None) -> Array:
        # x: [seq_len, embed_dim]
        x = jax.nn.gelu(jax.vmap(self.linear1)(x))
        x = self.dropout(x, key=key)
        x = jax.vmap(self.linear2)(x) # [seq_len, hidden_dim]
        return x # [seq_len, embed_dim]


class EncoderLayer(eqx.Module):
    embed_dim: int
    layernorm1: eqx.nn.LayerNorm
    layernorm2: eqx.nn.LayerNorm
    multihead_attention_layer: MultiHeadAttentionLayer
    token_mlp: TokenMLP
    dropout: eqx.nn.Dropout

    def __init__(
        self, 
        embed_dim: int, 
        n_heads: int, 
        hidden_dim: int, 
        dropout_rate: float, 
        *, 
        key: Key
    ):
        self.embed_dim = embed_dim
        keys = jr.split(key)
        self.layernorm1 = eqx.nn.LayerNorm(embed_dim)
        self.layernorm2 = eqx.nn.LayerNorm(embed_dim)
        self.multihead_attention_layer = MultiHeadAttentionLayer(
            embed_dim, n_heads, dropout_rate, key=keys[0]
        )
        self.token_mlp = TokenMLP(
            embed_dim, hidden_dim, dropout_rate, key=keys[1]
        )
        self.dropout = eqx.nn.Dropout(dropout_rate)

    def __call__(
        self, 
        x: Float[Array, "s self.embed_dim"], 
        *, 
        key: Optional[Key] = None
    ) -> Tuple[Float[Array, "s self.embed_dim"], Float[Array, "s self.embed_dim"]]:
        # x: [x_len, embed_dim] 
        if key is not None:
            keys = jr.split(key, 4)

        # Layernorm first
        _x = jax.vmap(self.layernorm1)(x)

        # Self attention (K=Q=V)
        _x, attention = self.multihead_attention_layer(
            _x, _x, _x, key=keys[0] if key is not None else None
        )

        # Residual connections after every attention block
        x = x + self.dropout(
            _x, key=keys[1] if key is not None else None
        )
        _x = jax.vmap(self.layernorm2)(x)

        _x = self.token_mlp(
            _x, key=keys[2] if key is not None else None
        ) 
        x = x + self.dropout(
            _x, key=keys[3] if key is not None else None
        ) # [x_len, embed_dim]
        return x, attention # x: [x_len, embed_dim]



class Encoder(eqx.Module):
    embed_dim: int
    layers: list[EncoderLayer]

    def __init__(
        self, 
        embed_dim: int, 
        n_layers: int, 
        n_heads: int, 
        hidden_dim: int, 
        dropout_rate: float, 
        *, 
        key: Key
    ):
        self.embed_dim = embed_dim
        self.layers = [
            EncoderLayer(
                embed_dim, n_heads, hidden_dim, dropout_rate, key=_key
            ) 
            for _key in jr.split(key, n_layers)
        ]

    def __call__(
        self, 
        x: Float[Array, "s self.embed_dim"], 
        key: Optional[Key] = None
    ) -> Tuple[Float[Array, "s self.embed_dim"], List[Float[Array, "s a"]]]:
        # x: [x_len]

        attentions = []
        for i, layer in enumerate(self.layers):
            x, attention = layer(
                x, key=jr.fold_in(key, i) if key is not None else None
            )
            attentions.append(attention)

        return x, attentions # x: [x_len, embed_dim]


class ImageEmbedding(eqx.Module):
    patch_size: int
    linear: eqx.nn.Linear
    cls_token: Array

    def __init__(
        self, 
        channel: int, 
        patch_size: int, 
        embed_dim: int, 
        *, 
        key: Key
    ):
        keys = jr.split(key)
        self.patch_size = patch_size
        # [patch, patch_size * patch_size * channel] -> [patch, embed_dim]
        self.linear = eqx.nn.Linear(
            channel * patch_size * patch_size, embed_dim, key=keys[0]
        )
        # Class token
        self.cls_token = jr.normal(keys[1], (1, embed_dim))

    def __call__(self, x: Float[Array, "c h w"]) -> Float[Array, "p self.embed_dim"]:
        # x: [channel, width, height]
        c, *_ = image.shape

        flatten_patches = rearrange(
            image, 
            'c (n_w p1) (n_h p2) -> (n_w n_h) (p1 p2 c) ', 
            c=c,
            p1=self.patch_size, 
            p2=self.patch_size
        ) # [patch, patch_size * patch_size * channel]

        embedded_patches = jax.vmap(self.linear)(flatten_patches) # [patch, embed_dim]
        
        # Learnable embedding to the sequence of embedded patches for regression
        embedded_patches = jnp.concatenate([self.cls_token, embedded_patches])

        return embedded_patches # [1 + patch, embed_dim]


class PatchEmbedding(eqx.Module):
    linear: eqx.nn.Embedding
    patch_size: int

    def __init__(
        self,
        input_channels: int,
        output_shape: int,
        patch_size: int,
        key: Key,
    ):
        self.patch_size = patch_size
        self.linear = eqx.nn.Linear(
            self.patch_size ** 2 * input_channels,
            output_shape,
            key=key,
        )

    def __call__(self, x: Array) -> Array:
        x = rearrange(
            x,
            "c (h ph) (w pw) -> (h w) (c ph pw)",
            ph=self.patch_size,
            pw=self.patch_size,
        )
        x = jax.vmap(self.linear)(x)
        return x


class TokenPositionalEmbedding(eqx.Module):
    embed_dim: int
    token_embedding: ImageEmbedding
    position_embedding: eqx.nn.Embedding
    dropout: eqx.nn.Dropout
    
    def __init__(
        self, 
        c: int, 
        p: int, 
        embed_dim: int, 
        dropout_rate: float, 
        *, 
        key: Key
    ):
        self.embed_dim = embed_dim
        keys = jr.split(key)
        self.token_embedding = ImageEmbedding(c, p, embed_dim, key=keys[0])
        # Replace this with sin/cos embedding? Max number of patches = 100 here?
        self.position_embedding = eqx.nn.Embedding(100, embed_dim, key=keys[1]) # Replace 100 with image_size ** 2 / patch_size ** 2
        self.dropout = eqx.nn.Dropout(dropout_rate)

    def __call__(
        self, 
        x: Float[Array, "w h c"], # ?
        *, 
        key: Optional[Key] = None
    ) -> Float[Array, "s self.embed_dim"]:
        # x: [width, height, channel]
        x = self.token_embedding(x) # [x_len, embed_dim]

        # Positional embedding
        pos = jnp.arange(0, x.shape[0]) # [x_len]

        x = x * jnp.sqrt(self.embed_dim) + jax.vmap(self.position_embedding)(pos)
        x = self.dropout(x, key=key)
        return x # [x_len, embed_dim]


class VisionTransformer(eqx.Module):
    embedding: TokenPositionalEmbedding
    encoder: Encoder
    layernorm: eqx.nn.LayerNorm
    linear: eqx.nn.Linear

    def __init__(
        self, 
        c, 
        p, 
        embed_dim, 
        n_layers, 
        n_heads, 
        hidden_dim, 
        dropout_rate, 
        output_dim, 
        *, 
        key
    ):
        keys = jr.split(key, 3)
        self.embedding = TokenPositionalEmbedding(c, p, embed_dim, dropout_rate, key=keys[0])
        self.encoder = Encoder(embed_dim, n_layers, n_heads, hidden_dim, dropout_rate, key=keys[1])
        self.layernorm = eqx.nn.LayerNorm(embed_dim)
        self.linear = eqx.nn.Linear(embed_dim, output_dim, key=keys[2])

    def __call__(
        self, 
        x: Float[Array, "c h w"], 
        *, 
        key: Optional[Key] = None
    ) -> Tuple[Float[Array, "o"], List[Float[Array, "s a"]]]:
        # x: [x_len]
        if key is not None:
            keys = jr.split(key)

        x = self.embedding(
            x, key=keys[0] if key is not None else None
        )

        # Encoded x: [x_len, embed_dim]
        x_embedded, attentions = self.encoder(
            x, key=keys[1] if key is not None else None
        )

        # Classification head
        cls_token = x_embedded[0, :] # [embed_dim]
        cls_token = self.layernorm(cls_token)

        output = self.linear(cls_token) # [output_dim]
        return output, attentions