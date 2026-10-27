import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import argparse
import copy
import math
import random
import torch
import wandb
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm

from data import prepare_dataset
from model import prepare_model
from utils import get_datetime, set_logger, get_logger, set_seed, set_device, \
    log_settings, save_current_src
from utils.step_lr import WSDLR
from utils.avgmeter import MetricTracker
from utils.tools import evaluation, get_weight_norm, get_grad_norm
from utils.SAM import disable_running_stats, enable_running_stats, smooth_crossentropy
from utils.AdaMuon_plus_SAM import AdaMuon_plus_SAM
from utils.sharpness import H_eigval


def epoch_seed(seed: int, epoch: int) -> int:
    """총 epoch 수와 무관하게 (seed, epoch)만으로 epoch별 seed 결정."""
    return random.Random(seed * 1_000_003 + epoch).randrange(10_000_000)


def parse_int_list(s) -> list:
    """'10,20,30' / '[10, 20, 30]' / [10, 20, 30] -> [10, 20, 30]"""
    if isinstance(s, (list, tuple)):
        return [int(x) for x in s]
    s = str(s).replace("[", "").replace("]", "").replace(" ", "")
    return [int(x) for x in s.split(",") if x]


def build_plan(totals: list, decay_ratio: float, warmup_epochs: int) -> dict:
    """총 epoch T -> (분기 epoch E, decay epochs D)."""
    plan = {}
    for T in sorted(set(totals)):
        D = max(1, round(T * decay_ratio))
        E = T - D
        assert E >= warmup_epochs, f"T={T}: 분기 epoch {E} < warmup {warmup_epochs}"
        plan[T] = (E, D)
    return plan


def build_optimizer(model, lr, momentum, weight_decay, rho, adaptive):
    params = [p for p in model.parameters() if p.requires_grad]
    param_groups = [
        {
            "params": [p for p in params if p.ndim >= 2],
            "use_muon": True,
            "lr": lr,
            "momentum": momentum,
            "weight_decay": weight_decay,
        },
        {
            "params": [p for p in params if p.ndim < 2],
            "use_muon": False,
            "lr": lr,
            "weight_decay": weight_decay,
        },
    ]
    return AdaMuon_plus_SAM(param_groups, rho=rho, adaptive=adaptive)


def run_epoch(model, optimizer, scheduler, trainset, batch_size, label_smoothing,
              device, epoch, seed, steps_per_epoch, tracker, logger):
    """1 epoch 학습. (마지막 lr, 평균 grad norm) 반환."""
    set_seed(epoch_seed(seed, epoch))
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    model.train()
    grad_norm_sum = torch.zeros((), device=device)
    cur_lr = optimizer.param_groups[0]["lr"]
    for batch_idx, (inputs, labels) in enumerate(tqdm(trainloader)):
        cur_lr = scheduler(optimizer, epoch * steps_per_epoch + batch_idx)
        inputs, labels = inputs.to(device), labels.to(device)

        # --- 1st forward-backward: w 지점의 gradient로 perturbation 계산 ---
        enable_running_stats(model)
        outputs = model(inputs)
        loss = smooth_crossentropy(outputs, labels, smoothing=label_smoothing).mean()

        optimizer.zero_grad()
        loss.backward()
        grad_norm_sum += get_grad_norm(model)
        optimizer.first_step(zero_grad=True)  # w -> w + e(w)

        # --- 2nd forward-backward: w + e(w) 지점에서 gradient 재계산 ---
        disable_running_stats(model)
        smooth_crossentropy(
            model(inputs), labels, smoothing=label_smoothing
        ).mean().backward()
        optimizer.second_step(zero_grad=True)  # w + e(w) -> w, AdaMuon update 적용

        tracker.update({
            "train_loss": loss.item(),
            "train_acc": (outputs.max(1)[1] == labels).float().mean().item()
        }, n=inputs.size(0))

    logger.info(tracker)
    return cur_lr, (grad_norm_sum / len(trainloader)).item()


def eval_epoch(device, model, testloader, subset, batch_size, logger) -> dict:
    test_loss, test_acc, _ = evaluation(device, model, testloader)
    weight_norm = get_weight_norm(model)

    model.eval()
    logger.info("Computing top-2 Hessian eigenvalue...")
    top_eigenvalues = H_eigval(
        device=device,
        model=model,
        dataset=subset,
        loss_fn=nn.CrossEntropyLoss(reduction='sum'),
        neigs=2,
        physical_batch_size=batch_size
    )
    metrics = {
        "test_loss": test_loss,
        "test_acc": test_acc,
        "weight_norm": weight_norm,
        "eig_1": top_eigenvalues[0].item(),
        "eig_2": top_eigenvalues[1].item(),
    }
    logger.info(f"test_loss: {test_loss:.4f}, test_acc: {test_acc:.4f}, "
                f"eig_1: {metrics['eig_1']:.4f}, eig_2: {metrics['eig_2']:.4f}")
    return metrics


def train(save_path: str,
          device: torch.device,
          model: nn.Module,
          trainset: Dataset,
          testset: Dataset,
          lr: float,
          batch_size: int,
          weight_decay: float,
          momentum: float,
          rho: float,
          adaptive: bool,
          label_smoothing: float,
          warmup_epochs: int,
          plan: dict,
          decay_type: str,
          min_lr_ratio: float,
          seed: int,
          save_trunk_ckpt: bool) -> None:
    """trunk(warmup + stable) 학습 중 분기 epoch마다 decay branch를 수행.

    Args:
        plan: {총 epoch T: (분기 epoch E, decay epochs D)}
        save_trunk_ckpt: 분기 시점 trunk 체크포인트를 디스크에 저장할지 여부
    """
    logger = get_logger(__name__)
    os.makedirs(save_path, exist_ok=True)

    model = model.to(device)
    optimizer = build_optimizer(model, lr, momentum, weight_decay, rho, adaptive)
    steps_per_epoch = math.ceil(len(trainset) / batch_size)  # drop_last=False 기준

    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    subset_indices = torch.randperm(
        len(trainset), generator=torch.Generator().manual_seed(42)
    )[:512].tolist()
    subset = Subset(trainset, subset_indices)

    branch_points = {}  # E -> [T, ...]
    for T, (E, _) in plan.items():
        branch_points.setdefault(E, []).append(T)
    trunk_end = max(branch_points)
    longest_T = max(plan)
    logger.info(f"WSD plan (T: (E, D)): {plan}, trunk_end={trunk_end}")

    # T별 곡선을 각자의 epoch 축으로 기록
    for T in plan:
        wandb.define_metric(f"T{T}/epoch")
        wandb.define_metric(f"T{T}/*", step_metric=f"T{T}/epoch")

    def do_epoch(scheduler, tracker, epoch, log_Ts, tag):
        logger.info(f"######[{tag}] Epoch - {epoch}")
        cur_lr, grad_norm = run_epoch(model, optimizer, scheduler, trainset, batch_size,
                                      label_smoothing, device, epoch, seed,
                                      steps_per_epoch, tracker, logger)
        metrics = eval_epoch(device, model, testloader, subset, batch_size, logger)
        metrics.update({"grad_norm": grad_norm, "lr": cur_lr})
        tracker.track({**metrics, "epoch": epoch})

        payload = {}
        for T in log_Ts:
            payload[f"T{T}/epoch"] = epoch
            payload.update({f"T{T}/{k}": v for k, v in metrics.items()})
        if payload:
            wandb.log(payload)
        return metrics

    def run_branch(T, E, D):
        scheduler = WSDLR(steps_per_epoch, warmup_epochs,
                          decay_start_epoch=E, decay_epochs=D,
                          decay_type=decay_type, min_lr_ratio=min_lr_ratio)
        tracker = MetricTracker()
        metrics = None
        for epoch in range(E, T):
            metrics = do_epoch(scheduler, tracker, epoch, [T], f"T{T} decay")

        torch.save(model.state_dict(), os.path.join(save_path, f"T{T}_model_final.pt"))
        tracker.save_to_csv(os.path.join(save_path, f"T{T}_decay.csv"))
        wandb.run.summary[f"T{T}/final_test_acc"] = metrics["test_acc"]
        if T == longest_T:
            wandb.run.summary["final_test_acc"] = metrics["test_acc"]
        logger.info(f"[T={T}] final test_acc: {metrics['test_acc']:.4f}")

    # ---- trunk: warmup + stable ----
    trunk_scheduler = WSDLR(steps_per_epoch, warmup_epochs)  # decay 없음
    trunk_tracker = MetricTracker()
    for epoch in range(trunk_end):
        # trunk 구간은 이 epoch을 공유하는 모든 T 곡선에 기록
        log_Ts = [T for T, (E, _) in plan.items() if epoch < E]
        do_epoch(trunk_scheduler, trunk_tracker, epoch, log_Ts, "trunk")

        E = epoch + 1
        if E in branch_points:
            snap = {"model": copy.deepcopy(model.state_dict()),
                    "optimizer": copy.deepcopy(optimizer.state_dict())}
            if save_trunk_ckpt:
                torch.save({"epoch": E, "seed": seed, "batch_size": batch_size, **snap},
                           os.path.join(save_path, f"trunk_ckpt_ep{E}.pt"))

            for T in branch_points[E]:
                run_branch(T, E, plan[T][1])
                # trunk 상태 복원 (optimizer state가 snap과 메모리를 공유하지 않도록 deepcopy)
                model.load_state_dict(snap["model"])
                optimizer.load_state_dict(copy.deepcopy(snap["optimizer"]))
            del snap

    trunk_tracker.save_to_csv(os.path.join(save_path, "trunk.csv"))


def add_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AdaMuon_plus_SAM with shared-trunk WSD")
    ## the basic setting of exp
    parser.add_argument('--device', default=0, type=int,
                        help="set the device.")
    parser.add_argument("--seed", default=-1, type=int,
                        help="set the seed. -1이면 무작위")
    parser.add_argument("--save_root", default="../outs/tmp/", type=str,
                        help='the path of saving results.')
    parser.add_argument("--dataset", default="cifar10", type=str,
                        help='the dataset name.')
    parser.add_argument("--model", default="vgg16_bn", type=str,
                        help='the model name.')
    parser.add_argument("--lr", default=0.01, type=float,
                        help="set the peak learning rate.")
    parser.add_argument("--bs", default=128, type=int,
                        help="set the batch size")
    parser.add_argument("--wd", default=1e-4, type=float,
                        help="set the weight decay")
    parser.add_argument("--momentum", default=0.9, type=float,
                        help="set the momentum rate")
    parser.add_argument("--rho", default=2.0, type=float,
                        help="set the rho for SAM")
    parser.add_argument("--adaptive", action="store_true", dest="adaptive",
                        help="set the adaptive for SAM")
    parser.add_argument("--label_smoothing", default=0.1, type=float,
                        help="set the label smoothing")
    ## WSD
    parser.add_argument("--total_epochs", default="40", type=str,
                        help="공유 trunk에서 분기할 총 epoch 목록. 예: '10,20,30,40'")
    parser.add_argument("--decay_ratio", default=0.2, type=float,
                        help="총 epoch 대비 decay 비율")
    parser.add_argument("--warmup_epochs", default=1, type=int,
                        help="warmup epochs (총 epoch과 무관한 고정값)")
    parser.add_argument("--decay_type", default="1-sqrt", type=str,
                        choices=["linear", "1-sqrt", "cosine"],
                        help="decay shape")
    parser.add_argument("--min_lr_ratio", default=0.0, type=float,
                        help="decay 종료 시 lr / peak lr")
    parser.add_argument("--save_trunk_ckpt", action="store_true",
                        help="분기 시점 trunk 체크포인트 저장")
    # set if using debug mod
    parser.add_argument("-v", "--verbose", action="store_true", dest="verbose",
                        help="enable debug info output.")
    return parser.parse_args()


def main():
    # sweep agent는 값을 커맨드라인 인자로 넘기므로 argparse 값을 그대로 사용
    args = add_args()
    wandb.init(project="AdaMuon_plus_SAM", config=vars(args))

    if args.seed < 0:
        args.seed = random.SystemRandom().randint(0, 2**31 - 1)  # true random
    args.total_epochs = parse_int_list(args.total_epochs)
    plan = build_plan(args.total_epochs, args.decay_ratio, args.warmup_epochs)
    wandb.config.update({"seed": args.seed,
                         "plan": {str(T): list(v) for T, v in plan.items()}},
                        allow_val_change=True)

    # set the save_path (wandb run id로 병렬 agent 간 경로 충돌 방지)
    exp_name = "-".join([get_datetime(),
                         wandb.run.id,
                         f"seed{args.seed}",
                         f"{args.dataset}",
                         f"{args.model}",
                         f"T{'_'.join(map(str, sorted(plan)))}",
                         f"lr{args.lr}",
                         f"bs{args.bs}",
                         f"rho{args.rho}",
                         f"adaptive{args.adaptive}",
                         f"wu{args.warmup_epochs}",
                         f"dr{args.decay_ratio}-{args.decay_type}"])
    args.save_path = os.path.join(args.save_root, exp_name)
    os.makedirs(args.save_path, exist_ok=True)

    # set the logger
    set_logger(args.save_path)
    logger = get_logger(__name__, args.verbose)
    # set the seed
    set_seed(args.seed)
    # set the device
    args.device = set_device(args.device)
    # save the current src
    save_current_src(save_path=args.save_path)

    # show the args.
    logger.info("#########parameters settings....")
    log_settings(args)

    # prepare the dataset
    logger.info("#########preparing dataset....")
    if args.dataset.startswith("cifar") and args.model.startswith("WideResNet"):
        trainset, testset = prepare_dataset(args.dataset, cutout=True)
    else:
        trainset, testset = prepare_dataset(args.dataset)

    # prepare the model
    logger.info("#########preparing model....")
    model = prepare_model(args.model, args.dataset, args.seed)
    logger.info(model)

    # train the model
    logger.info("#########training model....")
    train(save_path=os.path.join(args.save_path, "train"),
          device=args.device,
          model=model,
          trainset=trainset,
          testset=testset,
          lr=args.lr,
          batch_size=args.bs,
          weight_decay=args.wd,
          momentum=args.momentum,
          rho=args.rho,
          adaptive=args.adaptive,
          label_smoothing=args.label_smoothing,
          warmup_epochs=args.warmup_epochs,
          plan=plan,
          decay_type=args.decay_type,
          min_lr_ratio=args.min_lr_ratio,
          seed=args.seed,
          save_trunk_ckpt=args.save_trunk_ckpt)

    wandb.finish()


if __name__ == "__main__":
    main()
