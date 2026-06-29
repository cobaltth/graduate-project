from typing import List
import math

class StepLRforWRN:
    def __init__(self, learning_rate: float, total_epochs: int):
        """_summary_

        Args:
            learning_rate (float): _description_
            total_epochs (int): _description_
        """
        self.total_epochs = total_epochs
        self.base = learning_rate

    def __call__(self, optimizer, epoch):
        if epoch < self.total_epochs * 3/10:
            lr = self.base
        elif epoch < self.total_epochs * 6/10:
            lr = self.base * 0.2
        # elif epoch < self.total_epochs * 8/10:
        #     lr = self.base * 0.2 ** 2
        else:
            lr = self.base * 0.2 ** 2

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