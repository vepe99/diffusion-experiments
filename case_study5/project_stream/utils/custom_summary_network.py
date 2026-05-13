import inspect
from collections.abc import Mapping
import keras
from keras import ops
import jax

from bayesflow.networks.summary import SummaryNetwork
from bayesflow.utils.serialization import deserialize, serializable, serialize
from bayesflow.types import Tensor, Shape

from bayesflow.utils import check_lengths_same

from bayesflow.networks.summary.transformers.transformer import Transformer
from bayesflow.networks.summary.transformers.attention import SetAttention, InducedSetAttention, PoolingByMultiHeadAttention
MASK_BACKBONE_KEY = "input_a"


def _accepts_attention_mask(layer: keras.Layer) -> bool:
    """Returns True if the layer's call() signature accepts an attention_mask argument."""
    try:
        sig = inspect.signature(layer.call)
        return "attention_mask" in sig.parameters
    except (ValueError, TypeError):
        return False


@serializable("bayesflow.networks")
class FusionNetwork(SummaryNetwork):
    def __init__(
        self,
        backbones: Mapping[str, keras.Layer],
        head: keras.Layer | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.backbones = backbones
        self.head = head
        self._ordered_keys = sorted([k for k in self.backbones.keys() if k != "attention_mask"])

        # Pre-compute which backbones accept attention_mask so we don't
        # call inspect on every forward pass.
        self._accepts_mask = {k: _accepts_attention_mask(v) for k, v in self.backbones.items()}

    def build(self, inputs_shape: Mapping[str, Shape]):
        if self.built:
            return
        output_shapes = []
        for k, shape in inputs_shape.items():
            if k == "attention_mask":
                pass
            else:
                if not self.backbones[k].built:
                    self.backbones[k].build(shape)
                output_shapes.append(self.backbones[k].compute_output_shape(shape))
        if self.head and not self.head.built:
            fusion_input_shape = (*output_shapes[0][:-1], sum(shape[-1] for shape in output_shapes))
            self.head.build(fusion_input_shape)
        self.built = True

    def compute_output_shape(self, inputs_shape: Mapping[str, Shape]):
        output_shapes = []
        for k, shape in inputs_shape.items():
            if k == "attention_mask":
                pass
            else:
                output_shapes.append(self.backbones[k].compute_output_shape(shape))
        output_shape = (*output_shapes[0][:-1], sum(shape[-1] for shape in output_shapes))
        if self.head:
            output_shape = self.head.compute_output_shape(output_shape)
        return output_shape

    def call(
        self,
        inputs: Mapping[str, Tensor],
        training: bool = False,
    ) -> Tensor:
        """
        Parameters
        ----------
        inputs : dict[str, Tensor]
            Each value in the dictionary is the input to the summary network
            with the corresponding key.
        training : bool, optional
            Whether the model is in training mode. Default is False.
        """
        attention_mask = inputs.get("attention_mask", None)

        if attention_mask is not None:
            print(f"[FusionNetwork.call] attention_mask shape {attention_mask.shape} -> '{MASK_BACKBONE_KEY}' only")

        outputs = []
        for k in self._ordered_keys:
            if k == MASK_BACKBONE_KEY and attention_mask is not None:
                out = self.backbones[k](inputs[k], training=training, attention_mask=attention_mask)
            else:
                out = self.backbones[k](inputs[k], training=training)
            outputs.append(out)
 
        outputs = ops.concatenate(outputs, axis=-1)
        if self.head is None:
            return outputs
        return self.head(outputs, training=training)

    def compute_metrics(
        self,
        inputs: Mapping[str, Tensor],
        stage: str = "training",
        **kwargs,
    ) -> dict[str, Tensor]:
        """
        Parameters
        ----------
        inputs : dict[str, Tensor]
            Each value in the dictionary is the input to the summary network
            with the corresponding key.
        stage : str, optional
            Training stage string. Default is ``"training"``.
        attention_mask : Tensor, optional
            Boolean mask forwarded to backbones that accept it.
        **kwargs
            Additional keyword arguments passed to backbone compute_metrics().
        """

        if not self.built:
            self.build(keras.tree.map_structure(keras.ops.shape, inputs))


        attention_mask = inputs.get("attention_mask", None)
        if attention_mask is not None:
            print(f"[FusionNetwork.compute_metrics] attention_mask shape {attention_mask.shape} -> '{MASK_BACKBONE_KEY}' only")


        metrics = {"loss": [], "outputs": []}
        is_training = stage == "training"

        for k in self._ordered_keys:
            backbone = self.backbones[k]
 
            if isinstance(backbone, SummaryNetwork):
                extra = {"attention_mask": attention_mask} if (k == MASK_BACKBONE_KEY and attention_mask is not None) else {}
                metrics_k = backbone.compute_metrics(inputs[k], stage=stage, **extra, **kwargs)
                metrics["outputs"].append(metrics_k["outputs"])
                if "loss" in metrics_k:
                    metrics["loss"].append(metrics_k["loss"])
            else:
                if k == MASK_BACKBONE_KEY and attention_mask is not None:
                    out = backbone(inputs[k], training=is_training, attention_mask=attention_mask)
                else:
                    out = backbone(inputs[k], training=is_training)
                metrics["outputs"].append(out)
 
        if len(metrics["loss"]) == 0:
            del metrics["loss"]
        else:
            metrics["loss"] = ops.sum(metrics["loss"])
 
        metrics["outputs"] = ops.concatenate(metrics["outputs"], axis=-1)
        if self.head is not None:
            metrics["outputs"] = self.head(metrics["outputs"], training=is_training)
 
        return metrics

    def get_config(self) -> dict:
        base_config = super().get_config()
        config = {
            "backbones": self.backbones,
            "head": self.head,
        }
        return base_config | serialize(config)

    @classmethod
    def from_config(cls, config: dict, custom_objects=None):
        config = deserialize(config, custom_objects=custom_objects)
        return cls(**config)




@serializable("bayesflow.networks")
class SetTransformer(Transformer):
    """(SN) Implements the set transformer architecture from [1] which ultimately represents
    a learnable permutation-invariant function. Designed to naturally model interactions in
    the input set, which may be hard to capture with the simpler ``DeepSet`` architecture.

    [1] Lee, J., Lee, Y., Kim, J., Kosiorek, A., Choi, S., & Teh, Y. W. (2019).
        Set transformer: A framework for attention-based permutation-invariant neural networks.
        In International conference on machine learning (pp. 3744-3753). PMLR.
    """

    def __init__(
        self,
        summary_dim: int = 16,
        embed_dims: tuple = (64, 64),
        num_heads: tuple = (4, 4),
        num_seeds: int = 4,
        dropout: float = 0.05,
        expansion_factor: float = 4.0,
        glu_variant: str = "swiglu",
        kernel_initializer: str = "glorot_uniform",
        use_bias: bool = False,
        layer_norm: bool = True,
        num_inducing_points: int = None,
        seed_dim: int = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        check_lengths_same(embed_dims, num_heads)

        shared_kwargs = dict(
            dropout=dropout,
            expansion_factor=expansion_factor,
            glu_variant=glu_variant,
            kernel_initializer=kernel_initializer,
            use_bias=use_bias,
            layer_norm=layer_norm,
        )

        self.attention_blocks = []
        for i in range(len(embed_dims)):
            block_kwargs = shared_kwargs | dict(num_heads=num_heads[i], embed_dim=embed_dims[i])
            if num_inducing_points is None:
                block = SetAttention(**block_kwargs)
            else:
                block = InducedSetAttention(num_inducing_points=num_inducing_points, **block_kwargs)
            self.attention_blocks.append(block)

        self.pooling_by_attention = PoolingByMultiHeadAttention(
            num_heads=num_heads[-1],
            embed_dim=embed_dims[-1],
            num_seeds=num_seeds,
            seed_dim=seed_dim,
            **shared_kwargs,
        )
        self.output_projector = keras.layers.Dense(units=summary_dim)
        self.summary_dim = summary_dim

    def call(self, x: Tensor, training: bool = False, attention_mask: Tensor = None) -> Tensor:
        """Compresses the input set into a summary vector of size ``summary_dim``.

        Parameters
        ----------
        x : Tensor
            Input of shape ``(batch_size, set_size, input_dim)``.
        training : bool, optional
            Passed to dropout and norm layers, by default False.
        attention_mask : Tensor, optional
            Boolean mask of shape ``(B, 1, S)`` where 1 = attend, 0 = mask.
            Broadcast to ``(B, T, S)`` inside MultiHeadAttention automatically.

        Returns
        -------
        Tensor
            Output of shape ``(batch_size, summary_dim)``.
        """
        if attention_mask is not None:
            print(
                f"[SetTransformer.call] attention_mask ARRIVED — shape: "
                f"{attention_mask.shape}, dtype: {attention_mask.dtype}"
            )
            # print(f"[SetTransformer.call] attention_mask sum values: {attention_mask.sum()}")
            jax.debug.print("[SetTransformer.call] attention_mask sum: {mask}", mask=attention_mask.sum())
        else:
            print("[SetTransformer.call] No attention_mask provided (None).")

        for layer in self.attention_blocks:
            x = layer(x, training=training, attention_mask=attention_mask)

        x = self.pooling_by_attention(x, training=training)
        x = self.output_projector(x)
        return x