import os, json
import torch
import torch.nn as nn
from torch.optim import SGD, lr_scheduler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import pandas as pd
from tqdm import tqdm
from argparse import ArgumentParser

from .models.cnn import CNN
# from .models.cnn import resnet20

import torchvision
import torchvision.transforms as transforms

# python3 -m esgd-ws -a "esgd"

parser = ArgumentParser(description="An Evolutionary Stochastic Gradient Descent Trainer")
parser.add_argument("scheme", choices=["baseline", "esgd", "esgd_ws"])
parser.add_argument("--model", type=str, default="cnn")
parser.add_argument("-a", dest="data_augmentation", action="store_true")
parser.add_argument("--dataset", type=str, default="cifar")
args = parser.parse_args()

DATA_DIR = os.path.expanduser("~/esgd-ws/datasets")
# WEIGHTS_DIR = "./model_weights"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE = torch.device("mps")

DATASET_DICT = {"cifar": datasets.CIFAR10}
# MODEL_DICT = {"cnn": resnet20}
MODEL_DICT = {"cnn": CNN}
DATA_FOLDER = {"cifar": "CIFAR10"}

results_dir = os.path.expanduser(f"~/esgd-ws/results/{args.dataset}/{'DA' if args.data_augmentation else 'Non-DA'}")
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

download = not os.path.exists(os.path.join(DATA_DIR, DATA_FOLDER[args.dataset]))

if args.scheme == "esgd":
    from .esgd import ESGD, get_current_time


    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


    train_set = torchvision.datasets.CIFAR10(root=DATA_DIR, train=True,
                                        download=True, transform=transform)

    # train_set = torch.FloatTensor(train_set.data / 255), torch.LongTensor(train_set.targets)

    test_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)

    # import matplotlib.pyplot as plt
    # import numpy as np


    HPSET = {
        # "lr": (100, 10, 1),
        "lr": (0.01, 0.05, 0.1),
        "momentum": (0.8, 0.9, 0.99),
        "nesterov": (False, True)
    } ## not using these, using basic SGD optimizer without a schedule instead




    SEED = [5, 15, 24, 34, 49, 60, 74, 89, 97, 100]

    # learning_rates = [0.01, 0.05, 0.001]
    # iterations = [1, 3, 5]
    for seed in SEED:
        # for batch_s in batch_size:
        # for lr in learning_rates:
        # for it in iterations:
        esgd = ESGD(
            hpset=HPSET,
            model_class=MODEL_DICT[args.model],
            random_state=seed,
            n_generations=3
        )

        results = esgd.train(
            train_set=train_set,
            test_set=test_set,
            batch_size=8,
            input_lr=0.01
        )

        with open(f"{results_dir}/{args.scheme}.json", "w") as f:
            json.dump(results, f)



