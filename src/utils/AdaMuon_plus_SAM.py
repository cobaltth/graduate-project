import torch
from AdaMuon import zeropower_via_newtonschulz5, adamuon_update, adam_update


class AdaMuon_plus_SAM(torch.optim.Optimizer):
    """
    AdaMuon + Sharpness-Aware Minimization.

    SAM.py의 2-step 절차를 그대로 따르되, base_optimizer.step() 대신
    AdaMuon의 update 로직(momentum / second_momentum / Newton-Schulz)을 사용한다.

    사용 방식:

        optimizer = AdaMuon_plus_SAM(param_groups, rho=0.05, adaptive=False)

        # --- 1st forward-backward: w 지점의 gradient로 perturbation 계산 ---
        loss = criterion(model(inputs), targets)
        loss.backward()
        optimizer.first_step(zero_grad=True)   # w -> w + e(w)

        # --- 2nd forward-backward: w + e(w) 지점에서 gradient 재계산 ---
        criterion(model(inputs), targets).backward()
        optimizer.second_step(zero_grad=True)  # w + e(w) -> w, 이후 AdaMuon update

    흐름:
        1. first_step  : 현재 grad로 e(w) 계산 후 w -> w + e(w) 로 이동 (SAM.py와 동일)
        2. (사용자가 직접) w + e(w) 지점에서 forward/backward 재수행
                         -> p.grad 에는 "perturbation이 적용된 gradient"가 담김
        3. second_step : w + e(w) -> w 로 복원한 뒤,
                         2번에서 얻은 perturbed gradient를 AdaMuon update에 사용

    BatchNorm을 쓰는 모델이라면 SAM 관례대로, 1st pass는
    enable_running_stats(model), 2nd pass는 disable_running_stats(model)를
    호출한 뒤 진행하는 것을 권장한다 (SAM.py에 정의된 헬퍼 사용 가능).
    """

    def __init__(self, param_groups, rho=0.05, adaptive=False):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        for group in param_groups:
            assert "use_muon" in group
            group["rho"] = group.get("rho", rho)
            group["adaptive"] = group.get("adaptive", adaptive)

            if group["use_muon"]:
                # AdaMuon (muon-side) defaults
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0)
                group["eps"] = group.get("eps", 1e-8)
                assert set(group.keys()) == set(
                    ["params", "lr", "momentum", "weight_decay", "use_muon", "eps", "rho", "adaptive"]
                )
            else:
                # Adam-side defaults (non-muon params, e.g. embeddings/bias/norm)
                group["lr"] = group.get("lr", 3e-4)
                group["betas"] = group.get("betas", (0.9, 0.95))
                group["eps"] = group.get("eps", 1e-10)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(
                    ["params", "lr", "betas", "eps", "weight_decay", "use_muon", "rho", "adaptive"]
                )

        super().__init__(param_groups, dict())

    # ---------------------------------------------------------------
    # Step 1: climb to w + e(w)   (SAM.py의 first_step과 동일)
    # ---------------------------------------------------------------
    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # w -> w + e(w)

        if zero_grad:
            self.zero_grad()

    # ---------------------------------------------------------------
    # Step 2: w + e(w) -> w 복원 후, perturbed gradient로 AdaMuon 업데이트
    # ---------------------------------------------------------------
    @torch.no_grad()
    def second_step(self, zero_grad=False):
        # restore w  (perturbed gradient는 p.grad에 이미 담겨있음)
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data = self.state[p]["old_p"]

        for group in self.param_groups:
            if group["use_muon"]:
                for p in group["params"]:
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)  # force sync
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(p)
                        flat_shape = p.view(len(p), -1).shape if p.ndim == 4 else p.shape
                        state["second_momentum_buffer"] = torch.zeros(
                            flat_shape, dtype=p.dtype, device=p.device
                        )

                    update = adamuon_update(
                        p.grad,  # perturbed gradient (w + e(w) 지점에서 계산됨)
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
                        p.grad = torch.zeros_like(p)
                    state = self.state[p]
                    if "exp_avg" not in state:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    update = adam_update(
                        p.grad, state["exp_avg"], state["exp_avg_sq"],
                        state["step"], group["betas"], group["eps"],
                    )
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])

        if zero_grad:
            self.zero_grad()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack([
                ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                for group in self.param_groups for p in group["params"]
                if p.grad is not None
            ]),
            p=2,
        )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)