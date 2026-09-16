# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the distillation runner."""

from __future__ import annotations

import torch

from rsl_rl.runners import DistillationRunner
from tests.runners.test_on_policy_runner import DummyEnv


def _make_distillation_cfg() -> dict:
    """Return a minimal training configuration for Distillation."""
    return {
        "num_steps_per_env": 8,
        "save_interval": 100,
        "obs_groups": {"student": ["policy"], "teacher": ["policy"]},
        "algorithm": {
            "class_name": "Distillation",
            "num_learning_epochs": 1,
            "gradient_length": 4,
        },
        "student": {
            "class_name": "MLPModel",
            "hidden_dims": [32, 32],
            "activation": "elu",
            "distribution_cfg": {"class_name": "GaussianDistribution"},
        },
        "teacher": {"class_name": "MLPModel", "hidden_dims": [32, 32], "activation": "elu"},
    }


def test_distillation_learn_runs_without_error() -> None:
    """The shared rollout loop must not require the PPO-only gSDE hooks."""
    runner = DistillationRunner(DummyEnv(), _make_distillation_cfg(), log_dir=None, device="cpu")
    assert not hasattr(runner.alg, "reset_sde_noise")
    runner.alg.teacher_loaded = True
    params_before = {n: p.clone() for n, p in runner.alg.student.named_parameters()}
    runner.learn(num_learning_iterations=2)
    assert any(not torch.equal(params_before[n], p) for n, p in runner.alg.student.named_parameters())
