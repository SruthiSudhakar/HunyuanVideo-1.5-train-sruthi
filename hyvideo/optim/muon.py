import math
import torch
try:
    from torch.distributed.tensor import DTensor
    from torch.distributed.tensor.placement_types import (
        Partial,
        Placement,
        Replicate,
        Shard,
    )
except ImportError:
    # handle old pytorch versions
    DTensor = None


def compute_average_gradnorm_by_group(
    module,
    prefix_groups=None,
    substring_groups=None,
    require_grad_only=True,
):
    """Compute RMS grad per named-parameter group.

    Returns per group:
        sqrt(sum_i g_i^2 / numel)

    where `numel` is the total number of gradient elements assigned to the group.

    Priority rule:
    - If the same group name exists in both substring_groups and prefix_groups,
      substring match takes priority for that parameter/group (no double count).
    """
    prefix_groups = prefix_groups or {}
    substring_groups = substring_groups or {}

    # Accumulate sum of squared grads and total element count per group
    group_sumsq = {k: 0.0 for k in prefix_groups}
    group_numel = {k: 0 for k in prefix_groups}
    for k in substring_groups:
        group_sumsq.setdefault(k, 0.0)
        group_numel.setdefault(k, 0)

    # Pre-normalize patterns (drop empty strings)
    prefix_items = [(g, tuple(p for p in pats if p)) for g, pats in prefix_groups.items()]
    substr_items = [(g, tuple(s for s in pats if s)) for g, pats in substring_groups.items()]

    for name, param in module.named_parameters():
        if require_grad_only and not param.requires_grad:
            continue

        grad = param.grad
        if grad is None:
            continue
        if hasattr(grad, "to_local"):
            grad = grad.to_local()

        # Detach/read-only; does not affect autograd or loss computation
        g = grad.detach().float()
        grad_sumsq = float((g * g).sum().item())
        grad_numel = g.numel()

        # Track which group names already matched via substring (higher priority)
        substring_matched_groups = set()

        # 1) Substring groups first (higher priority)
        for group_name, substrings in substr_items:
            if substrings and any(sub in name for sub in substrings):
                group_sumsq[group_name] += grad_sumsq
                group_numel[group_name] += grad_numel
                substring_matched_groups.add(group_name)

        # 2) Prefix groups second, but skip same group if substring already matched
        for group_name, prefixes in prefix_items:
            if group_name in substring_matched_groups:
                continue
            if prefixes and any(name.startswith(prefix) for prefix in prefixes):
                group_sumsq[group_name] += grad_sumsq
                group_numel[group_name] += grad_numel

    return {
        group_name: (math.sqrt(group_sumsq[group_name] / group_numel[group_name]) if group_numel[group_name] > 0 else 0.0)
        for group_name in group_sumsq
    }

# This code is modified from the GitHub repository of KellerJordan:
# https://github.com/KellerJordan/Muon/blob/master/muon.py
def zeropower_via_newtonschulz5(G, steps = 5):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    if isinstance(G, DTensor):
        device_mesh = G.device_mesh
        G = G.full_tensor()
    else:
        device_mesh = None
        
    assert len(G.shape) >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G
    if G.size(-2) > G.size(-1):
        X = X.mT
    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A  # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    
    if device_mesh != None:
        return DTensor.from_local(X, device_mesh) 
    else:
        return X

class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    Arguments:
        muon_params: The parameters to be optimized by Muon.
        lr: The learning rate. The updates will have spectral norm of `lr`. (0.02 is a good default)
        momentum: The momentum used by the internal SGD. (0.95 is a good default)
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        ns_steps: The number of Newton-Schulz iterations to run. (6 is probably always enough)
        adamw_params: The parameters to be optimized by AdamW. Any parameters in `muon_params` which are
        {0, 1}-D or are detected as being the embed or lm_head will be optimized by AdamW as well.
        adamw_lr: The learning rate for the internal AdamW.
        adamw_betas: The betas for the internal AdamW.
        adamw_eps: The epsilon for the internal AdamW.
        adamw_wd: The weight decay for the internal AdamW.
    """

    def __init__(
        self,
        lr=1e-3,
        wd=0.1,
        muon_params=None,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
        adamw_params=None,
        adamw_betas=(0.95, 0.95),
        adamw_eps=1e-8,
    ):

        defaults = dict(
            lr=lr,
            wd=wd,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
        )

        muon_params = list(muon_params) if muon_params is not None else []
        adamw_params = list(adamw_params) if adamw_params is not None else []

        param_groups = []
        if muon_params and isinstance(muon_params[0], dict):
            param_groups.extend(muon_params)
        elif muon_params:
            param_groups.append({"params": muon_params})
        if adamw_params and isinstance(adamw_params[0], dict):
            param_groups.extend(adamw_params)
        elif adamw_params:
            param_groups.append({"params": adamw_params})

        super().__init__(param_groups, defaults)

        # Sort parameters into those for which we will use Muon, and those for which we will not
        if muon_params and isinstance(muon_params[0], dict):
            for group in muon_params:
                for p in group["params"]:
                    assert p.ndim >= 2, p.ndim
                    self.state[p]["use_muon"] = True
        else:
            for p in muon_params:
                # Use Muon for every parameter in muon_params which is >= 2D and doesn't look like an embedding or head layer
                assert p.ndim >= 2, p.ndim
                self.state[p]["use_muon"] = True

        if adamw_params and isinstance(adamw_params[0], dict):
            for group in adamw_params:
                for p in group["params"]:
                    self.state[p]["use_muon"] = False
        else:
            for p in adamw_params:
                # Do not use Muon for parameters in adamw_params
                self.state[p]["use_muon"] = False

    def adjust_lr_for_muon(self, lr, param_shape):
        A, B = param_shape[:2]
        # We adjust the learning rate and weight decay based on the size of the parameter matrix
        # as describted in the paper
        adjusted_ratio = 0.2 * math.sqrt(max(A, B))
        adjusted_lr = lr * adjusted_ratio
        return adjusted_lr

    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:

            ############################
            #           Muon           #
            ############################

            params = [p for p in group["params"] if self.state[p]["use_muon"]]

            lr = group["lr"]
            wd = group["wd"]
            momentum = group["momentum"]

            # generate weight updates in distributed fashion
            for p in params:
                # sanity check
                g = p.grad
                if g is None:
                    # # For i2v task, vision_in does not have gradients.
                    # # Creating a dummy gradient to allow `full_tensor` computation.
                    # if isinstance(p, DTensor):
                    #     g = DTensor.from_local(torch.zeros_like(p.to_local()), p.device_mesh, placements=p.placements)
                    # else:
                    #     g = torch.zeros_like(p)
                    continue
                if g.ndim > 2:
                    g = g.view(g.size(0), -1)
                assert g is not None

                # calc update
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                if group["nesterov"]:
                    g = g.add(buf, alpha=momentum)
                else:
                    g = buf
                g = g.bfloat16()
                u = zeropower_via_newtonschulz5(g, steps=group["ns_steps"])

                # scale update
                adjusted_lr = self.adjust_lr_for_muon(lr, p.shape)

                # apply weight decay
                p.data.mul_(1 - lr * wd)

                # apply update
                p.data.add_(u.view(p.shape), alpha=-adjusted_lr)

            ############################
            #       AdamW backup       #
            ############################

            params = [p for p in group["params"] if not self.state[p]["use_muon"]]
            lr = group['lr']
            beta1, beta2 = group["adamw_betas"]
            eps = group["adamw_eps"]
            weight_decay = group["wd"]

            for p in params:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    state["moment1"] = torch.zeros_like(g)
                    state["moment2"] = torch.zeros_like(g)
                state["step"] += 1
                step = state["step"]
                buf1 = state["moment1"]
                buf2 = state["moment2"]
                buf1.lerp_(g, 1 - beta1)
                buf2.lerp_(g.square(), 1 - beta2)

                g = buf1 / (eps + buf2.sqrt())

                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                scale = bias_correction1 / bias_correction2**0.5
                p.data.mul_(1 - lr * weight_decay)
                p.data.add_(g, alpha=-lr / scale)

        return loss

# help function to create the Muon optimizer
def get_muon_optimizer(
    model,
    lr=1e-3,
    weight_decay=0.1,
    momentum=0.95,
    adamw_betas=(0.95, 0.95),
    adamw_eps=1e-8,
    adamw_name_substrings=["lora_",],
    lr_overrides=None,
    lora_name_substring="lora_",
    save_param_info=True,
    param_info_path="muon_param_lrs_rank0.txt",
):
    def use_adamw_for_name(name):
        return any(substring in name for substring in adamw_name_substrings)

    lr_overrides = lr_overrides or {}

    def resolve_lr(name):
        # Priority: LoRA > action_xattn > action_encoding > feature_transformer > transformer > base lr
        if lr_overrides.get("lora") is not None and lora_name_substring in name:
            return lr_overrides["lora"]
        if lr_overrides.get("action_xattn") is not None and "action_xattn" in name:
            return lr_overrides["action_xattn"]
        if lr_overrides.get("action_encoding") is not None and ("action_encoding_init" in name or "action_encode_blocks" in name):
            return lr_overrides["action_encoding"]
        if lr_overrides.get("feature_transformer") is not None and name.startswith("feature_transformer."):
            return lr_overrides["feature_transformer"]
        if lr_overrides.get("transformer") is not None and name.startswith("transformer."):
            return lr_overrides["transformer"]
        return lr

    muon_group_map = {}
    adamw_group_map = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        target_lr = resolve_lr(name)
        if p.ndim >= 2 and (not use_adamw_for_name(name)):
            muon_group_map.setdefault(target_lr, []).append(p)
        else:
            adamw_group_map.setdefault(target_lr, []).append(p)

    muon_params = [{"params": params, "lr": group_lr} for group_lr, params in muon_group_map.items()]
    adamw_params = [{"params": params, "lr": group_lr} for group_lr, params in adamw_group_map.items()]

    if save_param_info:
        is_rank0 = True
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            is_rank0 = (torch.distributed.get_rank() == 0)

        if is_rank0:
            lines = ["name\tshape\tstatus\tlr\toptimizer"]
            for name, p in model.named_parameters():
                shape = "x".join(str(dim) for dim in p.shape)
                if p.requires_grad:
                    target_lr = resolve_lr(name)
                    if p.ndim >= 2 and (not use_adamw_for_name(name)):
                        optimizer_name = "muon"
                    else:
                        optimizer_name = "adamw"
                    lines.append(f"{name}\t{shape}\ttrainable\t{target_lr}\t{optimizer_name}")
                else:
                    lines.append(f"{name}\t{shape}\tfrozen\t-\tfrozen")

            with open(param_info_path, "w") as f:
                f.write("\n".join(lines) + "\n")

    return Muon(
        lr=lr,
        wd=weight_decay,
        muon_params=muon_params,
        momentum=momentum,
        adamw_params=adamw_params,
        adamw_betas=adamw_betas,
        adamw_eps=adamw_eps,
    )
