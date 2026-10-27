from typing import List
import math

class WSDLR:
    """Warmup-Stable-Decay (step 단위).
    warmup/stable 구간 lr은 총 학습 길이와 무관 → stable 체크포인트 공유 가능.
    decay_start_epoch=None이면 stable 유지 (trunk run).
    """
    def __init__(self, steps_per_epoch, warmup_epochs,
                 decay_start_epoch=None, decay_epochs=0,
                 decay_type="1-sqrt", min_lr_ratio=0.0):
        self.warmup = warmup_epochs * steps_per_epoch
        self.decay_start = (None if decay_start_epoch is None
                            else decay_start_epoch * steps_per_epoch)
        self.decay = decay_epochs * steps_per_epoch
        self.decay_type = decay_type
        self.min_lr_ratio = min_lr_ratio

    def factor(self, step):
        if step < self.warmup:
            return (step + 1) / self.warmup
        if self.decay_start is None or step < self.decay_start:
            return 1.0
        p = min((step - self.decay_start) / self.decay, 1.0)
        if self.decay_type == "linear":
            f = 1.0 - p
        elif self.decay_type == "1-sqrt":
            f = 1.0 - math.sqrt(p)
        elif self.decay_type == "cosine":
            f = 0.5 * (1.0 + math.cos(math.pi * p))
        else:
            raise ValueError(self.decay_type)
        return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * f

    def __call__(self, optimizer, step):
        f = self.factor(step)
        for g in optimizer.param_groups:
            g.setdefault("initial_lr", g["lr"])
            g["lr"] = g["initial_lr"] * f
        return optimizer.param_groups[0]["lr"]

class StepLRforWRN:
    def __init__(self, learning_rate: float, total_epochs: int, first_decay: float = 0.2, second_decay: float = 0.2, milestones=(3/10, 6/10)):
        """Step LR for WideResNet with configurable decay factors and epoch fractions.

        Args:
            learning_rate (float): base learning rate
            total_epochs (int): total number of epochs
            first_decay (float): multiplicative factor applied at first milestone (e.g. 0.2)
            second_decay (float): multiplicative factor applied at second milestone (applied after first_decay)
            milestones (tuple): two fractions of total_epochs for first and second decay (e.g. (0.3, 0.6))
        """
        self.total_epochs = total_epochs
        self.base = learning_rate
        self.first_decay = first_decay
        self.second_decay = second_decay
        self.milestone_fracs = milestones

    def __call__(self, optimizer, epoch):
        first_cut = self.total_epochs * self.milestone_fracs[0]
        second_cut = self.total_epochs * self.milestone_fracs[1]

        if epoch < first_cut:
            lr = self.base
        elif epoch < second_cut:
            lr = self.base * self.first_decay
        else:
            lr = self.base * self.first_decay * self.second_decay

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr


class MultiStepLR:
    def __init__(self, learning_rate: float, milestones: List[int], gamma: float):
        """_summary_

        Args:
            learning_rate (float): _description_
            milestones (List[int]): _description_
            gamma (float): _description_
        """
        self.milestones = milestones
        self.base = learning_rate
        self.gamma = gamma
        
    def __call__(self, optimizer, epoch):
        lr = self.base
        for milestone in self.milestones:
            if epoch >= milestone - 1:
                lr *= self.gamma

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

class CosineWarmupLR:
    def __init__(self, learning_rate: float, total_epochs: int, warmup_epochs: int):
        """
        Args:
            learning_rate (float): 피크 학습률 (AdaMuon은 1e-3 또는 6e-4 추천)
            total_epochs (int): 총 에폭 수
            warmup_epochs (int): 웜업을 진행할 에폭 수 (보통 총 에폭의 5% 내외)
        """
        self.base_lr = learning_rate
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs

    def __call__(self, optimizer, epoch):
        # 1) Linear Warmup 구간
        if epoch < self.warmup_epochs:
            # 0에서 base_lr까지 선형적으로 증가
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
            
        # 2) Cosine Annealing 구간
        else:
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            # 코사인 공식에 따라 부드럽게 0에 가깝게 감소 (최저 lr 한선은 1e-5로 설정)
            lr = 1e-5 + 0.5 * (self.base_lr - 1e-5) * (1.0 + math.cos(math.pi * progress))

        # 옵티마이저에 학습률 주입
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
