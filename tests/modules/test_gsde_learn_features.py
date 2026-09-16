# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Regression test: gSDE sampling with differentiable features after rollout-time weight sampling."""

import torch

from rsl_rl.modules import MLP
from rsl_rl.modules.distribution import GSDEGaussianDistribution


def test_sample_backward_after_inference_mode_weights() -> None:
    """``learn_features=True`` must survive a ``sample()`` inside an autograd update.

    The runner draws the exploration tensors under ``torch.inference_mode()`` during the rollout.
    The PPO update then forwards with ``stochastic_output=True``, which calls ``sample()`` while
    ``latent_sde`` is differentiable; saving the inference tensors for backward raised
    ``RuntimeError: Inference tensors cannot be saved for backward``.
    """
    torch.manual_seed(0)
    mlp = MLP(8, 4, [16, 8], "elu")
    dist = GSDEGaussianDistribution(4, init_std=1.0, learn_features=True)
    dist.init_mlp_weights(mlp)

    with torch.inference_mode():
        dist.sample_weights(batch_size=6)

    obs = torch.randn(6, 8)
    mean, latent = mlp.forward_with_features(obs)
    dist.update(mean, latent_sde=latent)
    dist.sample().sum().backward()

    assert mlp[0].weight.grad is not None
    assert dist.log_std_param.grad is None  # noise carries no gradient
