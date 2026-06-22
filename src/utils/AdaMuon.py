import torch

def zeropower_via_newtonschulz5(G, steps: int):
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


def adamuon_update(grad, momentum, second_momentum, beta=0.95, ns_steps=5, eps=1e-8):
    # Update first momentum
    momentum.mul_(beta).add_(grad)

    update = momentum
    if update.ndim == 4:  # for the case of conv filters
        update = update.view(len(update), -1)

    # Sign-stabilized input to Newton-Schulz
    O = zeropower_via_newtonschulz5(update.sign(), steps=ns_steps)

    # Update second momentum (elementwise on orthogonalized direction)
    second_momentum.mul_(beta).add_(O * O, alpha=1 - beta)

    # Apply second momentum normalization
    O_hat = O / (second_momentum.sqrt() + eps)

    # RMS-aligned scaling factor
    m, n = O_hat.shape[-2], O_hat.shape[-1]
    gamma = 0.2 * (m * n) ** 0.5 / (O_hat.norm() + 1e-7)

    return gamma * O_hat


def adam_update(grad, buf1, buf2, step, betas, eps):
    buf1.lerp_(grad, 1 - betas[0])
    buf2.lerp_(grad.square(), 1 - betas[1])
    buf1c = buf1 / (1 - betas[0]**step)
    buf2c = buf2 / (1 - betas[1]**step)
    return buf1c / (buf2c.sqrt() + eps)


class AdaMuon_optimizer(torch.optim.Optimizer):
    def __init__(self, param_groups):
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                # defaults
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0)
                group["eps"] = group.get("eps", 1e-8)
                assert set(group.keys()) == set(["params", "lr", "momentum", "weight_decay", "use_muon", "eps"])
            else:
                # defaults
                group["lr"] = group.get("lr", 3e-4)
                group["betas"] = group.get("betas", (0.9, 0.95))
                group["eps"] = group.get("eps", 1e-10)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(["params", "lr", "betas", "eps", "weight_decay", "use_muon"])
        super().__init__(param_groups, dict())

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                for p in group["params"]:
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)  # Force synchronization
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                        # second momentum is shaped like the (possibly flattened)
                        # 2D update used inside Newton-Schulz
                        flat_shape = p.view(len(p), -1).shape if p.ndim == 4 else p.shape
                        state["second_momentum_buffer"] = torch.zeros(flat_shape, dtype=p.dtype, device=p.device)

                    update = adamuon_update(
                        p.grad,
                        state["momentum_buffer"],
                        state["second_momentum_buffer"],
                        beta=group["momentum"],
                        eps=group["eps"],
                    )
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update.reshape(p.shape), alpha=-group["lr"])
            else:
                for p in group["params"]:
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)  # Force synchronization
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    update = adam_update(p.grad, state["exp_avg"], state["exp_avg_sq"],
                                          state["step"], group["betas"], group["eps"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])

        return loss