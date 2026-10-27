import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import argparse
import random
import math
import torch
import wandb
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm

from data import prepare_dataset
from model import prepare_model
from utils import get_datetime, set_logger, get_logger, set_seed, set_device, \
    log_settings, save_current_src
from utils.step_lr import StepLRforWRN, MultiStepLR, CosineWarmupLR, WSDLR
from utils.avgmeter import MetricTracker
from utils.tools import evaluation, get_weight_norm, get_grad_norm
from utils.SAM import disable_running_stats, enable_running_stats, smooth_crossentropy
from utils.AdaMuon_plus_SAM import AdaMuon_plus_SAM
from utils.sharpness import H_eigval


def epoch_seed(seed: int, epoch: int) -> int:
    """총 epoch 수와 무관하게 (seed, epoch)만으로 epoch별 seed 결정."""
    return random.Random(seed * 1_000_003 + epoch).randrange(10_000_000)


def save_ckpt(path, model, optimizer, epoch, seed, batch_size):
    torch.save({
        "epoch": epoch,            # 완료된 epoch 수 = 재개 시 시작 epoch
        "seed": seed,
        "batch_size": batch_size,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }, path)


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
          start_epoch: int,
          end_epoch: int,
          warmup_epochs: int,
          decay_start,
          decay_epochs: int,
          decay_type: str,
          min_lr_ratio: float,
          branch_epochs: list,
          resume,
          seed: int) -> None:
    """train the model

    Args:
        start_epoch / end_epoch: 학습 구간 [start_epoch, end_epoch)
        warmup_epochs: warmup 길이 (총 epoch과 무관한 고정값)
        decay_start: decay 시작 epoch (None이면 stable 유지)
        decay_epochs: decay 길이
        branch_epochs: 완료 시점에 분기용 체크포인트를 저장할 epoch 목록
        resume: 불러온 체크포인트 dict (없으면 None)
    """
    logger = get_logger(__name__)
    os.makedirs(save_path, exist_ok=True)

    ## set up the basic component for training
    model = model.to(device)

    # 학습 전체 구간을 AdaMuon_plus_SAM 하나로 진행
    params = list(filter(lambda p: p.requires_grad, model.parameters()))
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
    optimizer = AdaMuon_plus_SAM(param_groups, rho=rho, adaptive=adaptive)

    if resume is not None:
        model.load_state_dict(resume["model"])
        optimizer.load_state_dict(resume["optimizer"])
        logger.info(f"resumed from checkpoint at epoch {start_epoch}")

    steps_per_epoch = math.ceil(len(trainset) / batch_size)  # drop_last=False 기준
    scheduler = WSDLR(steps_per_epoch, warmup_epochs,
                      decay_start_epoch=decay_start, decay_epochs=decay_epochs,
                      decay_type=decay_type, min_lr_ratio=min_lr_ratio)
    logger.info(f"WSD: epochs [{start_epoch}, {end_epoch}), warmup={warmup_epochs}, "
                f"decay_start={decay_start}, decay_epochs={decay_epochs}, "
                f"decay_type={decay_type}, branch_epochs={branch_epochs}")

    ## set up the data part
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    subset_indices = torch.randperm(
        len(trainset), generator=torch.Generator().manual_seed(42)
    )[:512].tolist()
    subset = Subset(trainset, subset_indices)

    tracker = MetricTracker()
    cur_lr = optimizer.param_groups[0]["lr"]
    for epoch in range(start_epoch, end_epoch):
        logger.info(f"######Epoch - {epoch}")
        set_seed(epoch_seed(seed, epoch))

        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
        model.train()
        grad_norm_sum = torch.zeros((), device=device)
        for batch_idx, (inputs, labels) in enumerate(tqdm(trainloader)):
            cur_lr = scheduler(optimizer, epoch * steps_per_epoch + batch_idx)
            inputs, labels = inputs.to(device), labels.to(device)

            # --- 1st forward-backward: w 지점의 gradient로 perturbation 계산 ---
            enable_running_stats(model)
            outputs = model(inputs)
            loss = smooth_crossentropy(
                outputs, labels, smoothing=label_smoothing
            ).mean()

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

        test_loss, test_acc, _ = evaluation(device, model, testloader)
        logger.info(f"test_loss: {test_loss:.4f}, test_acc: {test_acc:.4f}, lr: {cur_lr:.6f}")

        weight_norm = get_weight_norm(model)
        grad_norm = (grad_norm_sum / len(trainloader)).item()

        model.eval()

        logger.info("Computing top-2 Hessian eigenvalue...")
        loss_fn = nn.CrossEntropyLoss(reduction='sum')

        top_eigenvalues = H_eigval(
            device=device,
            model=model,
            dataset=subset,
            loss_fn=loss_fn,
            neigs=2,
            physical_batch_size=batch_size
        )
        eig_1 = top_eigenvalues[0].item()
        eig_2 = top_eigenvalues[1].item()
        logger.info(f"First Hessian Eigenvalue: {eig_1:.4f}")
        logger.info(f"Second Hessian Eigenvalue: {eig_2:.4f}")

        metrics = {
            "test_loss": test_loss,
            "test_acc": test_acc,
            "weight_norm": weight_norm,
            "grad_norm": grad_norm,
            "epoch": epoch,
            "lr": cur_lr,
            "eig_1": eig_1,
            "eig_2": eig_2,
        }
        tracker.track(metrics)
        wandb.log(metrics)

        if (epoch + 1) in branch_epochs:
            ckpt_path = os.path.join(save_path, f"ckpt_ep{epoch + 1}.pt")
            save_ckpt(ckpt_path, model, optimizer, epoch + 1, seed, batch_size)
            logger.info(f"saved branch checkpoint: {ckpt_path}")

        if epoch == end_epoch - 1:
            logger.info("save the final results")
            torch.save(model.state_dict(),
                       os.path.join(save_path, "model_final.pt"))
            tracker.save_to_csv(os.path.join(save_path, "train.csv"))


def add_args() -> argparse.Namespace:
    """get arguments from the program.

    Returns:
        return a dict containing all the program arguments
    """
    parser = argparse.ArgumentParser(
        description="simple verification")
    ## the basic setting of exp
    parser.add_argument('--device', default=0, type=int,
                        help="set the device.")
    parser.add_argument("--seed", default=0, type=int,
                        help="set the seed.")
    parser.add_argument("--save_root", default="../outs/tmp/", type=str,
                        help='the path of saving results.')
    parser.add_argument("--dataset", default="cifar10", type=str,
                        help='the dataset name.')
    parser.add_argument("--model", default="vgg16_bn", type=str,
                        help='the model name.')
    parser.add_argument('--epochs', default=160, type=int,
                        help="total epochs (trunk run / 단일 WSD run). branch run에서는 무시")
    parser.add_argument("--lr", default=0.01, type=float,
                        help="set the peak learning rate.")
    parser.add_argument("--bs", default=128, type=int,
                        help="set the batch size")
    parser.add_argument("--wd", default=1e-4, type=float,
                        help="set the weight decay")
    parser.add_argument("--momentum", default=0.9, type=float,
                        help="set the momentum rate")
    parser.add_argument("--start_SAM", default=150, type=int,
                        help="(unused)")
    parser.add_argument("--end_SAM", default=160, type=int,
                        help="(unused)")
    parser.add_argument("--rho", default=2.0, type=float,
                        help="set the rho for SAM")
    parser.add_argument("--adaptive", action="store_true", dest="adaptive",
                        help="set the adaptive for SAM")
    parser.add_argument("--label_smoothing", default=0.1, type=float,
                        help="set the label smoothing")
    parser.add_argument("--step_size", default=[80, 160], type=int, nargs="+",
                        help="(unused)")
    parser.add_argument("--step_saving", default=160, type=int,
                        help="(unused)")
    ## WSD
    parser.add_argument("--warmup_epochs", default=4, type=int,
                        help="warmup epochs (총 epoch과 무관한 고정값)")
    parser.add_argument("--decay_epochs", default=0, type=int,
                        help="decay epochs. trunk run은 0 (stable 유지)")
    parser.add_argument("--decay_type", default="1-sqrt", type=str,
                        choices=["linear", "1-sqrt", "cosine"],
                        help="decay shape")
    parser.add_argument("--min_lr_ratio", default=0.0, type=float,
                        help="decay 종료 시 lr / peak lr")
    parser.add_argument("--branch_epochs", default=[], type=int, nargs="*",
                        help="완료 시점에 분기용 체크포인트를 저장할 epoch 목록")
    parser.add_argument("--resume", default=None, type=str,
                        help="branch run: trunk 체크포인트 경로")
    # set if using debug mod
    parser.add_argument("-v", "--verbose", action="store_true", dest="verbose",
                        help="enable debug info output.")
    args = parser.parse_args()

    if not os.path.exists(args.save_root):
        os.makedirs(args.save_root)

    # set the save_path
    name_parts = [get_datetime(),
                  f"{args.dataset}",
                  f"{args.model}",
                  f"lr{args.lr}",
                  f"bs{args.bs}",
                  f"rho{args.rho}",
                  f"adaptive{args.adaptive}",
                  f"wu{args.warmup_epochs}"]
    if args.resume is not None:
        ckpt_name = os.path.splitext(os.path.basename(args.resume))[0]
        name_parts += [f"branch-{ckpt_name}", f"decay{args.decay_epochs}-{args.decay_type}"]
    else:
        name_parts += [f"epochs{args.epochs}", f"decay{args.decay_epochs}-{args.decay_type}"]
    args.save_path = os.path.join(args.save_root, "-".join(name_parts))
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    return args


def main():
    # get the args.
    args = add_args()

    # 같은 trunk에서 나온 run을 wandb group으로 묶음
    if args.resume is not None:
        trunk_dir = os.path.dirname(os.path.dirname(os.path.abspath(args.resume)))
        wandb_group = os.path.basename(trunk_dir)
    else:
        wandb_group = os.path.basename(os.path.normpath(args.save_path))

    wandb.init(project="AdaMuon_plus_SAM", config=args, group=wandb_group)
    config = wandb.config

    list_keys = ("step_size", "branch_epochs")
    for key, value in config.as_dict().items():
        if key in list_keys:
            continue
        setattr(args, key, value)
    # parsing needed
    for key in list_keys:
        if key in config:
            setattr(args, key, [int(x) for x in config[key]])

    # set the logger
    set_logger(args.save_path)
    # get the logger
    logger = get_logger(__name__, args.verbose)

    # set the schedule range / seed
    resume = None
    if args.resume is not None:  # branch (decay) run
        assert args.decay_epochs > 0, "branch run은 --decay_epochs > 0 필요"
        resume = torch.load(args.resume, map_location="cpu", weights_only=False)
        assert resume["batch_size"] == args.bs, \
            f"bs 불일치: trunk {resume['batch_size']} vs branch {args.bs}"
        args.seed = resume["seed"]
        start_epoch = resume["epoch"]
        end_epoch = start_epoch + args.decay_epochs
        decay_start = start_epoch
        branch_epochs = []
    else:  # trunk run (decay_epochs=0) 또는 단일 WSD run
        args.seed = random.SystemRandom().randint(0, 2**31 - 1)  # true random
        start_epoch, end_epoch = 0, args.epochs
        decay_start = args.epochs - args.decay_epochs if args.decay_epochs > 0 else None
        branch_epochs = args.branch_epochs
        assert decay_start is None or all(e <= decay_start for e in branch_epochs), \
            "branch_epochs는 decay 시작 이전이어야 함"
    wandb.config.update({"seed": args.seed,
                         "start_epoch": start_epoch,
                         "end_epoch": end_epoch},
                        allow_val_change=True)
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
          start_epoch=start_epoch,
          end_epoch=end_epoch,
          warmup_epochs=args.warmup_epochs,
          decay_start=decay_start,
          decay_epochs=args.decay_epochs,
          decay_type=args.decay_type,
          min_lr_ratio=args.min_lr_ratio,
          branch_epochs=branch_epochs,
          resume=resume,
          seed=args.seed)

    wandb.finish()


if __name__ == "__main__":
    main()
