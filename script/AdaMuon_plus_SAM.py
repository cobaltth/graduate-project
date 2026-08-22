import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import argparse
import random
import torch
import wandb
import math
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm
from torch.optim.lr_scheduler import LambdaLR

from data import prepare_dataset
from model import prepare_model
from utils import get_datetime, set_logger, get_logger, set_seed, set_device, \
    log_settings, save_current_src
from utils.step_lr import StepLRforWRN, MultiStepLR, CosineWarmupLR
from utils.avgmeter import MetricTracker
from utils.tools import evaluation, get_weight_norm
from utils.SAM import disable_running_stats, enable_running_stats, smooth_crossentropy
from utils.AdaMuon_plus_SAM import AdaMuon_plus_SAM
from utils.sharpness import H_eigval


def train(save_path: str,
          device: torch.device,
          model: nn.Module,
          trainset: Dataset,
          testset: Dataset,
          epochs: int,
          lr: float,
          batch_size: int,
          weight_decay: float,
          momentum: float,
          start_SAM: int,
          end_SAM: int,
          rho: float,
          adaptive: bool,
          label_smoothing: float,
          step_size: list,
          step_saving: int,
          seed: int) -> None:
    """train the model

    Args:
        save_path: the path to save results
        device: GPU or CPU
        model: the model to train
        trainset: the train dataset
        testset: the test dataset
        epochs: the epochs
        lr: the learning rate
        batch_size: the batch size
        weight_decay: the weight decay
        momentum: the momentum
        rho: the rho for AdaMuon_plus_SAM's SAM-style perturbation
        adaptive: the adaptive for AdaMuon_plus_SAM
        label_smoothing: the label smoothing
        step_size: the StepLR's step size
        steps_saving: the steps to save the model
        seed: the seed
    """
    logger = get_logger(__name__)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    ## set up the basic component for training
    # put the model to GPU or CPU
    model = model.to(device)

    # set the optimizer: 이제 학습 전체 구간(과거의 ERM phase + SAM phase)을
    # AdaMuon_plus_SAM 하나로 진행한다.
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

    # 기존 SAM_optimizer가 사용하던 rho, adaptive를 그대로 AdaMuon_plus_SAM에 전달
    optimizer = AdaMuon_plus_SAM(param_groups, rho=rho, adaptive=adaptive)
    scheduler = CosineWarmupLR(lr, epochs, int(epochs * 0.03))

    ## set up the data part
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    subset_indices = torch.randperm(
        len(trainset), generator=torch.Generator().manual_seed(42)
    )[:512].tolist()
    subset = Subset(trainset, subset_indices)

    seeds = random.Random(seed).sample(range(10000000), k=epochs)

    tracker = MetricTracker()
    for epoch in range(0, epochs):
        logger.info(f"######Epoch - {epoch}")
        set_seed(seeds[epoch])

        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
        model.train()
        for batch_idx, (inputs, labels) in enumerate(tqdm(trainloader)):
            inputs, labels = inputs.to(device), labels.to(device)

            # --- 1st forward-backward: w 지점의 gradient로 perturbation 계산 ---
            enable_running_stats(model)
            outputs = model(inputs)
            loss = smooth_crossentropy(
                outputs, labels, smoothing=label_smoothing
            ).mean()

            optimizer.zero_grad()
            loss.backward()
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
        logger.info(f"test_loss: {test_loss:.4f}, test_acc: {test_acc:.4f}")

        weight_norm = get_weight_norm(model)

        model.eval()
        
        logger.info("Computing top-2 Hessian eigenvalue...")
        loss_fn = nn.CrossEntropyLoss(reduction='sum')

        top_eigenvalues = H_eigval(
            device=device,
            model=model,
            dataset=subset,       # or dataset subset
            loss_fn=loss_fn,
            neigs=2,                # max eigenvalue만 필요하므로 2로 설정
            physical_batch_size=batch_size # OOM 방지 및 메모리에 맞게 조절
        )
        eig_1 = top_eigenvalues[0].item()
        eig_2 = top_eigenvalues[1].item()
        logger.info(f"First Hessian Eigenvalue: {eig_1:.4f}")
        logger.info(f"Second Hessian Eigenvalue: {eig_2:.4f}")

        tracker.track({
            "test_loss": test_loss,
            "test_acc": test_acc,
            "weight_norm": weight_norm,
            "epoch": epoch,
            "eig_1": eig_1,
            "eig_2": eig_2
        })

        wandb.log({
            "test_loss": test_loss,
            "test_acc": test_acc,
            "weight_norm": weight_norm,
            "epoch": epoch,
            "eig_1": eig_1,
            "eig_2": eig_2
        })

        scheduler(optimizer, epoch)

        if epoch == epochs - 1:
            logger.info("save the final results")
            torch.save(model.state_dict(),
                       os.path.join(save_path, f"model_final.pt"))
            tracker.save_to_csv(os.path.join(save_path, f"train.csv"))


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
                        help="set iteration number")
    parser.add_argument("--lr", default=0.01, type=float,
                        help="set the learning rate.")
    parser.add_argument("--bs", default=128, type=int,
                        help="set the batch size")
    parser.add_argument("--wd", default=1e-4, type=float,
                        help="set the weight decay")
    parser.add_argument("--momentum", default=0.9, type=float,
                        help="set the momentum rate")
    parser.add_argument("--start_SAM", default=150, type=int,
                        help="set the start epoch of SAM")
    parser.add_argument("--end_SAM", default=160, type=int,
                        help="set the end epoch of SAM")
    parser.add_argument("--rho", default=2.0, type=float,
                        help="set the rho for SAM")
    parser.add_argument("--adaptive", action="store_true", dest="adaptive",
                        help="set the adaptive for SAM")
    parser.add_argument("--label_smoothing", default=0.1, type=float,
                        help="set the label smoothing")
    parser.add_argument("--step_size", default=[80, 160], type=int, nargs="+",
                        help="set the StepLR stepsize")
    parser.add_argument("--step_saving", default=160, type=int,
                        help="set the steps to save the model")
    # set if using debug mod
    parser.add_argument("-v", "--verbose", action="store_true", dest="verbose",
                        help="enable debug info output.")
    args = parser.parse_args()

    if not os.path.exists(args.save_root):
        os.makedirs(args.save_root)

    # set the save_path
    exp_name = "-".join([get_datetime(),
                         f"seed{args.seed}",
                         f"{args.dataset}",
                         f"{args.model}",
                         f"epochs{args.epochs}",
                         f"lr{args.lr}",
                         f"bs{args.bs}",
                         f"rho{args.rho}",
                         f"adaptive{args.adaptive}",])
    args.save_path = os.path.join(args.save_root, exp_name)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    return args


def main():
    # get the args.
    args = add_args()

    wandb.init(project="AdaMuon_plus_SAM", config=args)
    config = wandb.config

    for key, value in config.as_dict().items():
        if key == "step_size":
            continue            
        setattr(args, key, value)
    
    # parsing needed
    if "step_size" in config:
        args.step_size = [int(x) for x in config.step_size]

    # set the logger
    set_logger(args.save_path)
    # get the logger
    logger = get_logger(__name__, args.verbose)
    # set the seed
    args.seed = random.SystemRandom().randint(0, 2**31 - 1) # true random
    set_seed(args.seed)
    # set the device
    args.device = set_device(args.device)
    # save the current src
    save_current_src(save_path = args.save_path)

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
    train(save_path = os.path.join(args.save_path, "train"),
          device = args.device,
          model = model,
          trainset = trainset,
          testset = testset,
          epochs = args.epochs,
          lr = args.lr,
          batch_size = args.bs,
          weight_decay = args.wd,
          momentum = args.momentum,
          start_SAM=args.start_SAM,
          end_SAM=args.end_SAM,
          rho = args.rho,
          adaptive = args.adaptive,
          label_smoothing = args.label_smoothing,
          step_size = args.step_size,
          step_saving = args.step_saving,
          seed = args.seed)

    wandb.finish()

if __name__ == "__main__":
    main()
