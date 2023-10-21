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

# cd Documents
# cd ESGD
# source m1/bin/activate
# cd esgd-ws
# python3 -m esgd-ws -a "esgd"

parser = ArgumentParser(description="An Evolutionary Stochastic Gradient Descent Trainer")
parser.add_argument("scheme", choices=["baseline", "esgd", "esgd_ws"])
parser.add_argument("--model", type=str, default="cnn")
parser.add_argument("-a", dest="data_augmentation", action="store_true")
parser.add_argument("--dataset", type=str, default="fashion_mnist")
args = parser.parse_args()

DATA_DIR = os.path.expanduser("~/esgd-ws/datasets")
# WEIGHTS_DIR = "./model_weights"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATASET_DICT = {
    "mnist": datasets.MNIST,
    "fashion_mnist": datasets.FashionMNIST
}
MODEL_DICT = {"cnn": CNN}
DATA_FOLDER = {"mnist": "MNIST", "fashion_mnist": "FashionMNIST"}

results_dir = os.path.expanduser(f"~/esgd-ws/results/{args.dataset}/{'DA' if args.data_augmentation else 'Non-DA'}")
if not os.path.exists(results_dir):
    os.makedirs(results_dir)


download = not os.path.exists(os.path.join(DATA_DIR, DATA_FOLDER[args.dataset]))




if args.scheme == "esgd":
    from .esgd import ESGD, get_current_time

    transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])

    train_set = DATASET_DICT[args.dataset](root=DATA_DIR, train=True, download=True, transform=transform)
    # train_set.data = torch.FloatTensor(train_set.data / 255)
    # train_data = torch.FloatTensor(train_set.data / 255).unsqueeze(1)
    # train_targets = torch.LongTensor(train_set.targets)
    test_set = DATASET_DICT[args.dataset](root=DATA_DIR, train=False, download=True, transform=transform)
    # test_set.data = torch.FloatTensor(test_set.data / 255)
    # test_set = torch.FloatTensor(test_set.data / 255).unsqueeze(1), torch.LongTensor(test_set.targets)


    HPSET = {
        "lr": (0.01, 0.05, 0.1),
        "momentum": (0.8, 0.9, 0.99),
        "nesterov": (False, True)
    }

    SEED = [5, 15, 24, 34, 49, 60, 74, 89, 97, 100]
    for seed in SEED:
        esgd = ESGD(
            hpset=HPSET,
            model_class=MODEL_DICT[args.model],
            random_state=seed,
            n_generations=15
        )
        results = esgd.train(
            train_set=train_set,
            test_set=test_set,
            batch_size=64,
            input_lr=0.001
        )
        with open(f"{results_dir}/{args.scheme}.json", "w") as f:
            json.dump(results, f)


