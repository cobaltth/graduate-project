from typing import List
import math

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