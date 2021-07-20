import collections
from typing import Iterable, Optional, Type

from torch import nn
import torch as th
import gym
from stable_baselines3.common import preprocessing

class ActObsMLP(nn.Module):
    """Simple MLP that takes an action and observation and produces a single
    output."""

    def __init__(
        self, action_space: gym.Space, observation_space: gym.Space, **mlp_kwargs
    ):
        super().__init__()

        in_size = preprocessing.get_flattened_obs_dim(
            observation_space
        ) + preprocessing.get_flattened_obs_dim(action_space)
        self.mlp = build_mlp(
            **{"in_size": in_size, "out_size": 1, **mlp_kwargs}
        )

    def forward(self, obs: th.Tensor, acts: th.Tensor) -> th.Tensor:
        cat_inputs = th.cat((obs, acts), dim=1)
        outputs = self.mlp(cat_inputs)
        return outputs.squeeze(1)

class SqueezeLayer(nn.Module):
    """Torch module that squeezes a B*1 tensor down into a size-B vector."""

    def forward(self, x):
        assert x.ndim == 2 and x.shape[1] == 1
        new_value = x.squeeze(1)
        assert new_value.ndim == 1
        return new_value


def build_mlp(
    in_size: int,
    hid_sizes: Iterable[int],
    out_size: int = 1,
    name: Optional[str] = None,
    activation: Type[nn.Module] = nn.ReLU,
    squeeze_output=False,
    flatten_input=False,
) -> nn.Module:
    """Constructs a Torch MLP.

    Args:
        in_size: size of individual input vectors; input to the MLP will be of
            shape (batch_size, in_size).
        hid_sizes: sizes of hidden layers.
        out_size: required size of output vector.
        activation: activation to apply after hidden layers.
        squeeze_output: if out_size=1, then squeeze_input=True ensures that MLP
            output is of size (B,) instead of (B,1).
        flatten_input: should input be flattened along axes 1, 2, 3, …? Useful
            if you want to, e.g., process small images inputs with an MLP.

    Returns:
        nn.Module: an MLP mapping from inputs of size (batch_size, in_size) to
            (batch_size, out_size), unless out_size=1 and squeeze_output=True,
            in which case the output is of size (batch_size, ).

    Raises:
        ValueError: if squeeze_output was supplied with out_size!=1."""
    layers = collections.OrderedDict()

    if name is None:
        prefix = ""
    else:
        prefix = f"{name}_"

    if flatten_input:
        layers[f"{prefix}flatten"] = nn.Flatten()

    # Hidden layers
    prev_size = in_size
    for i, size in enumerate(hid_sizes):
        layers[f"{prefix}dense{i}"] = nn.Linear(prev_size, size)
        prev_size = size
        if activation:
            layers[f"{prefix}act{i}"] = activation()

    # Final layer
    layers[f"{prefix}dense_final"] = nn.Linear(prev_size, out_size)

    # sigmoid; hehua 20210719 20:59;
    layers[f"{prefix}act_final"] = nn.Sigmoid()

    if squeeze_output:
        if out_size != 1:
            raise ValueError("squeeze_output is only applicable when out_size=1")
        layers[f"{prefix}squeeze"] = SqueezeLayer()

    model = nn.Sequential(layers)

    return model
