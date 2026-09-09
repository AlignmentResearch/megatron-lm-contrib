# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
import copy
import random

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, Adam

try:
    from transformer_engine.pytorch.optimizers import FusedAdam as GPUAdam
    from transformer_engine.pytorch.optimizers import FusedSGD as GPUSGD
except:
    # Handle environment where transformer_engine is not installed
    from torch.optim import SGD as GPUSGD
    from torch.optim import Adam as GPUAdam

from megatron.core.optimizer.cpu_offloading import HybridDeviceOptimizer


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class BigNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 2048)
        self.fc2 = nn.Linear(2048, 8192)
        self.fc3 = nn.Linear(8192, 2048)
        self.fc4 = nn.Linear(2048, 100)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


def setup_seed(seed):
    random.seed(seed)  # Set Python's built-in random seed
    np.random.seed(seed)  # Set NumPy's random seed
    torch.manual_seed(seed)  # Set PyTorch's CPU seed
    torch.cuda.manual_seed(seed)  # Set PyTorch's GPU seed (if using CUDA)
    torch.cuda.manual_seed_all(seed)  # Set seed for all GPUs
    torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior
    torch.backends.cudnn.benchmark = False  # Disable auto-tuner for reproducibility


@pytest.mark.skipif(
    torch.__version__ < '2.3.0',
    reason=(
        "Requires PyTorch 2.3.0 or higher, lower versions of pytorch have "
        "misaligned optimizer accuracy for CPU and GPU."
    ),
)
@pytest.mark.parametrize('n_steps', [1, 10])
@pytest.mark.parametrize('overlap_cpu_optimizer_d2h_h2d', [False, True])
@pytest.mark.parametrize('offload_fraction', [0, 0.5, 1.0])
@pytest.mark.parametrize('optimizer', ['sgd', 'adam'])
@pytest.mark.parametrize('with_param_groups', [False, True])
def test_multi_device_hybrid_optimizer(
    with_param_groups, optimizer, offload_fraction, overlap_cpu_optimizer_d2h_h2d, n_steps
):
    setup_seed(42)
    net1 = Net().cuda()
    net2 = Net().cuda()
    net2.load_state_dict(net1.state_dict())
    base_lr = 1e-3
    params = list(net1.parameters())
    ref_params = list(net2.parameters())
    if with_param_groups:
        param_groups = [
            {"params": params[: len(params) // 2], "wd_mult": 1.0, "lr_mult": 1e-4},
            {"params": params[len(params) // 2 :], "wd_mult": 0.0, "lr_mult": 2e-4},
        ]
        params = param_groups
        ref_param_groups = [
            {"params": ref_params[: len(ref_params) // 2], "wd_mult": 1.0, "lr_mult": 1e-4},
            {"params": ref_params[len(ref_params) // 2 :], "wd_mult": 0.0, "lr_mult": 2e-4},
        ]
        ref_params = ref_param_groups

    if optimizer == 'adam':
        cls_kwargs = dict(cpu_optimizer_cls=Adam, gpu_optimizer_cls=GPUAdam)
    else:
        cls_kwargs = dict(cpu_optimizer_cls=SGD, gpu_optimizer_cls=GPUSGD)

    hdo = HybridDeviceOptimizer(
        params,
        offload_fraction=offload_fraction,
        lr=base_lr,
        overlap_cpu_optimizer_d2h_h2d=overlap_cpu_optimizer_d2h_h2d,
        **cls_kwargs,
    )

    ref_optimizer = cls_kwargs['gpu_optimizer_cls'](ref_params, lr=base_lr)

    # 1. run step on optimizer, make sure there is state generated
    assert len(hdo.state_dict()["state"]) == 0  # state is empty
    input = torch.randn(1, 3, 32, 32).cuda()
    output = net1(input)
    output.sum().backward()
    hdo.step()
    output = net2(input)
    output.sum().backward()
    ref_optimizer.step()
    # PyTorch SGD will not generate state
    if optimizer != 'sgd':
        assert len(hdo.state_dict()["state"]) != 0

    # 2. check the state is on right device
    if optimizer == 'adam':
        first_param_id = hdo.state_dict()["param_groups"][0]["params"][0]
        last_param_id = hdo.state_dict()["param_groups"][-1]["params"][-1]
        if offload_fraction > 0:
            assert not hdo.state_dict()["state"][first_param_id]["exp_avg"].is_cuda
        if offload_fraction < 1:
            assert hdo.state_dict()["state"][last_param_id]["exp_avg"].is_cuda

    # 3. check parameters allclose
    for _ in range(1, n_steps):
        input = torch.randn(1, 3, 32, 32).cuda()
        output = net1(input)
        output.sum().backward()
        hdo.step()
        output = net2(input)
        output.sum().backward()
        ref_optimizer.step()

    params = net1.state_dict()
    ref_params = net2.state_dict()
    for k, v in params.items():
        assert (v.isnan() == ref_params[k].isnan()).all()
        torch.nan_to_num_(v, 0)
        torch.nan_to_num_(ref_params[k], 0)
        assert torch.allclose(
            v, ref_params[k], atol=1e-03
        ), f"Weight {k} value mismatch, max error: {(v - ref_params[k]).abs().max()}"


@pytest.mark.skipif(
    torch.__version__ < '2.3.0',
    reason=(
        "Requires PyTorch 2.3.0 or higher, lower versions of pytorch have "
        "misaligned optimizer accuracy for CPU and GPU."
    ),
)
@pytest.mark.parametrize('n_steps', [1, 10])
@pytest.mark.parametrize('offload_fraction', [1, 0.5, 0])
@pytest.mark.parametrize('optimizer', ['adam', 'sgd'])
@pytest.mark.parametrize('with_param_groups', [False, True])
def test_overlap_cpu_optimizer_d2h_h2d_sync_correctness(
    with_param_groups, optimizer, offload_fraction, n_steps
):
    setup_seed(42)
    net1 = BigNet().cuda()
    net2 = BigNet().cuda()
    net2.load_state_dict(net1.state_dict())
    base_lr = 1e-3
    params = list(net1.parameters())
    ref_params = list(net2.parameters())
    if with_param_groups:
        param_groups = [
            {"params": params[: len(params) // 2], "wd_mult": 1.0, "lr_mult": 1e-4},
            {"params": params[len(params) // 2 :], "wd_mult": 0.0, "lr_mult": 2e-4},
        ]
        params = param_groups
        ref_param_groups = [
            {"params": ref_params[: len(ref_params) // 2], "wd_mult": 1.0, "lr_mult": 1e-4},
            {"params": ref_params[len(ref_params) // 2 :], "wd_mult": 0.0, "lr_mult": 2e-4},
        ]
        ref_params = ref_param_groups

    if optimizer == 'adam':
        cls_kwargs = dict(cpu_optimizer_cls=Adam, gpu_optimizer_cls=GPUAdam)
    else:
        cls_kwargs = dict(cpu_optimizer_cls=SGD, gpu_optimizer_cls=GPUSGD)

    hdo = HybridDeviceOptimizer(
        params,
        offload_fraction=offload_fraction,
        lr=base_lr,
        overlap_cpu_optimizer_d2h_h2d=True,
        **cls_kwargs,
    )

    ref_optimizer = cls_kwargs['gpu_optimizer_cls'](ref_params, lr=base_lr)

    # 1. run step on optimizer, make sure there is state generated
    assert len(hdo.state_dict()["state"]) == 0  # state is empty
    input = torch.randn(1, 3, 32, 32).cuda()
    output = net1(input)
    output.sum().backward()
    hdo.step()
    output = net2(input)
    output.sum().backward()
    ref_optimizer.step()
    # PyTorch SGD will not generate state
    if optimizer != 'sgd':
        assert len(hdo.state_dict()["state"]) != 0

    # 2. check the state is on right device
    if optimizer == 'adam':
        first_param_id = hdo.state_dict()["param_groups"][0]["params"][0]
        last_param_id = hdo.state_dict()["param_groups"][-1]["params"][-1]
        if offload_fraction > 0:
            assert not hdo.state_dict()["state"][first_param_id]["exp_avg"].is_cuda
        if offload_fraction < 1:
            assert hdo.state_dict()["state"][last_param_id]["exp_avg"].is_cuda

    inputs = [torch.randn(1, 3, 32, 32).cuda() for _ in range(1, n_steps)]
    for i in range(1, n_steps):
        output = net1(inputs[i - 1])
        output.sum().backward()
        hdo.step()

    for i in range(1, n_steps):
        output = net2(inputs[i - 1])
        output.sum().backward()
        ref_optimizer.step()

    params = net1.state_dict()
    ref_params = net2.state_dict()
    for k, v in params.items():
        assert (v.isnan() == ref_params[k].isnan()).all()
        torch.nan_to_num_(v, 0)
        torch.nan_to_num_(ref_params[k], 0)
        assert torch.allclose(
            v, ref_params[k], atol=1e-03
        ), f"Weight {k} value mismatch, max error: {(v - ref_params[k]).abs().max()}"


def _hybrid_adam(params, offload_fraction):
    return HybridDeviceOptimizer(
        params,
        offload_fraction=offload_fraction,
        lr=1e-3,
        overlap_cpu_optimizer_d2h_h2d=True,
        param_update_in_fp32=True,
        cpu_optimizer_cls=Adam,
        gpu_optimizer_cls=GPUAdam,
    )


def _train_steps(net, hdo, x, n_steps):
    for _ in range(n_steps):
        hdo.zero_grad()
        net(x).float().sum().backward()
        hdo.step()


@pytest.mark.skipif(
    torch.__version__ < '2.3.0',
    reason=(
        "Requires PyTorch 2.3.0 or higher, lower versions of pytorch have "
        "misaligned optimizer accuracy for CPU and GPU."
    ),
)
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
@pytest.mark.parametrize('offload_fraction', [0.5, 1.0])
def test_load_state_dict_reuses_the_cpu_copies_and_resumes_exactly(dtype, offload_fraction):
    """A loaded optimizer keeps its sub-optimizers, pinned CPU copies and gradient
    buffers (no second set is allocated), points every `master_param` at the live
    inner parameter, and then steps exactly like the optimizer it was saved from.

    Natively fp32 parameters used to raise a KeyError in the post-load hook (no fp32
    clone), and a bf16 parameter's fp32 master carries precision the bf16 weight lost,
    so an exact continuation needs the loaded master copied into the live clone.
    """
    setup_seed(42)
    net1 = Net().cuda().to(dtype)
    hdo1 = _hybrid_adam(list(net1.parameters()), offload_fraction)
    x = torch.randn(4, 3, 32, 32, device='cuda', dtype=dtype)
    _train_steps(net1, hdo1, x, 3)
    # state_dict() hands out the live tensors; a checkpoint round trip yields fresh ones
    saved = copy.deepcopy(hdo1.state_dict())

    # a fresh optimizer over the same (bf16-rounded) weights, as a resume builds one
    net2 = Net().cuda().to(dtype)
    net2.load_state_dict(net1.state_dict())
    hdo2 = _hybrid_adam(list(net2.parameters()), offload_fraction)
    inner_before = dict(hdo2.param_to_inner_param)
    cpu_optimizers_before = list(hdo2.cpu_optimizers)
    grad_buffers_before = hdo2.cpu_copy_map_grad

    hdo2.load_state_dict(saved)

    # 1. nothing was rebuilt: the same inner parameters, sub-optimizers and gradient buffer map
    assert hdo2.param_to_inner_param.keys() == inner_before.keys()
    for param, inner in inner_before.items():
        assert hdo2.param_to_inner_param[param] is inner
    assert len(hdo2.cpu_optimizers) == len(cpu_optimizers_before)
    for a, b in zip(hdo2.cpu_optimizers, cpu_optimizers_before):
        assert a is b
    assert hdo2.cpu_copy_map_grad is grad_buffers_before

    # 2. the state refers to the live inner parameters and lives where they live
    for param, state in hdo2.state.items():
        assert state["master_param"] is hdo2.param_to_inner_param[param]
        expect_cpu = param in hdo2.gpu_params_map_cpu_copy
        assert state["exp_avg"].is_cuda != expect_cpu
        assert state["exp_avg_sq"].is_cuda != expect_cpu

    # 3. the continuation is exact: same masters, same moments, same steps
    _train_steps(net1, hdo1, x, 2)
    _train_steps(net2, hdo2, x, 2)
    for (name, p1), p2 in zip(net1.named_parameters(), net2.parameters()):
        torch.testing.assert_close(p1, p2, rtol=0, atol=0, msg=lambda m: f"{name}: {m}")
