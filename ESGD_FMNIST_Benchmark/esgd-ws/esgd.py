import os, sys, time
from datetime import datetime
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import DataLoader, SubsetRandomSampler
import itertools
import functools
import numpy as np

import torch.optim as optim
import torchvision
from torchvision import datasets, transforms

from statistics import mean

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from matplotlib.backends.backend_pdf import PdfPages
# import torchvision.transforms as transforms

import csv


# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("mps")



# cd Documents
# cd ESGD
# source m1/bin/activate
# cd esgd-ws
# python3 -m esgd-ws -a "esgd"




def get_current_time():
    return datetime.fromtimestamp(time.time()).isoformat()


class ESGD:

    def __init__(
            self,
            hpset,
            model_class,
            fitness_function=nn.CrossEntropyLoss(),
            n_generations=20,
            n_population=5,
            sgds_per_gen=1,
            evos_per_gen=1,
            reproductive_factor=4,
            m_elite=3,
            mixing_number=3,
            optimizer_class=SGD,
            mutation_length_init=0.001,
            random_state=97,
            device=DEVICE,
            verbose=True
    ):
        self.hpnames, self.hpvalues = self.extract_from_hpset(hpset)
        self.model_class = model_class
        self.fitness_function = fitness_function
        self.n_generations = n_generations
        self.n_population = n_population
        self.sgds_per_gen = sgds_per_gen
        self.evos_per_gen = evos_per_gen
        self.reproductive_factor = reproductive_factor
        self.m_elite = m_elite
        self.mixing_number = mixing_number
        self.optimizer_class = optimizer_class
        self.mutation_length = mutation_length_init
        self.random_state = random_state
        self.device = device
        self.verbose = verbose




    @staticmethod
    def extract_from_hpset(hpset):
        hpnames = []
        hpvalues = []
        for k, v in hpset.items():
            hpnames.append(k)
            hpvalues.append(v)
        return tuple(hpnames), tuple(itertools.product(*hpvalues))

    @staticmethod
    def get_data_loader(
            train_data,
            train_targets,
            shuffle=True,
            batch_size=1024,
            transform=None
    ):
        class Dataset:
            def __init__(self, train_data, train_targets, transform=None):
                self.train_data = train_data
                self.train_targets = train_targets
                self.length = len(self.train_targets)
                self.transform = transform

            def __getitem__(self, idx):
                if self.transform is None:
                    return self.train_data[idx], self.train_targets[idx]
                else:
                    return self.transform(self.train_data[idx]), self.train_targets[idx]

            def __len__(self):
                return self.length

        return DataLoader(
            Dataset(train_data, train_targets, transform=transform),
            shuffle=shuffle,
            batch_size=batch_size
        )

    class Logger:

        def __init__(self, log_file=None):
            self.log_file = log_file

        def logging(self, s):
            if self.log_file is None:
                sys.stdout.write(s)
                sys.stdout.write("\n")
            else:
                with open(self.log_file, "a+") as f:
                    f.write(s)
                    f.write("\n")

    def _sample_optimizer(self):
        hpval_indices = np.random.choice(len(self.hpvalues), size=self.n_population)
        return [self.hpvalues[idx] for idx in hpval_indices]

    # for graph
    history = []
    grad_step_counter = 0







    def train(
            self,
            train_set,
            test_set=None,
            log_file=None,
            batch_size=64,
            input_lr=0.01,
            transform=None
    ):

        def graph_history(history):
            integers = [i for i in range(1, (len(history))+1)]

            ema = []
            avg = history[0]

            ema.append(avg)

            for loss in history:
                avg = (avg * 0.9) + (0.1 * loss)
                ema.append(avg)


            x = [j * (100 * 5) for j in integers]
            y = history

            # plot line
            plt.plot(x, ema[:len(history)])
            # plot title/captions
            plt.title("ESGD FMNIST")
            plt.xlabel("Gradient Steps")
            plt.ylabel("Validation Loss")
            plt.tight_layout()
            
            # plt.savefig("TEST_DATA/PD_trial_%s.png" % trial)
            def save_image(filename):
                p = PdfPages(filename)
                fig = plt.figure(1)
                fig.savefig(p, format='pdf') 
                p.close()

            filename = "esgd_FMNIST_progress_without_reg_model4_line.pdf"
            # save_image(filename)

            # plot points too
            plt.scatter(x, history, s=20)

            print("ema:"), print(ema), print("")
            print("x:"), print(x), print("")
            print("history:"), print(history), print("")

            def save_image(filename):
                p = PdfPages(filename)
                fig = plt.figure(1)
                fig.savefig(p, format='pdf') 
                p.close()

            filename = "esgd_FMNIST_progress_without_reg_model4_with_points.pdf"
            # save_image(filename)

            # plt.show(block=True), plt.close()
            plt.close('all')


        # to track loss progress
        def get_best_test_loss(curr_gen):
            test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)

            test_losses = [0.0 for _ in range(int(self.n_population))]
            test_corrects = [0 for _ in range(int(self.n_population))]
            test_total = 0
            with torch.no_grad():
                for ind in curr_gen:
                    ind = ind.to(self.device)
                    ind.eval()

                for (x, y) in test_loader:
                    x = x.to(self.device)
                    y = y.to(self.device)
                    test_total += x.size(0)
                    for i, ind in enumerate(curr_gen):
                        out = ind(x)
                        test_losses[i] += self.fitness_function(out, y).item() * x.size(0)
                        test_corrects[i] += (out.max(dim=1)[1] == y).sum().item()


                for ind in curr_gen:
                    ind.train()

                test_losses = list(map(lambda x: x / test_total, test_losses))

            return min(test_losses)

        history = []
        grad_step_counter = 0

        import time
        start_time = time.time()


        # LOADING DATA FOR POPULATION DESCENT COMPARISON TEST
        FM_test_images = test_set[0]
        FM_test_labels = test_set[1]


        logger = self.Logger(log_file)


        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                  shuffle=True) ## new train_loader, use this

        if test_set is not None:
            test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                         shuffle=True)
            # test_loader = self.get_data_loader(*test_set, shuffle=False, batch_size=batch_size)
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        curr_gen = [self.model_class().to(self.device) for _ in range(self.n_population)]
        results = []
        for g in range(1, 1 + self.n_generations):
            curr_hpvals = self._sample_optimizer()
            optimizers = [optim.SGD(ind.parameters(), lr=input_lr, momentum=0.9) for ind in curr_gen] # custom optimizer, not ESGD give
            # optimizers = [self.optimizer_class(
            #     ind.parameters(), **dict(zip(self.hpnames, hpvs))
            # ) for ind, hpvs in zip(curr_gen, curr_hpvals)]
            running_losses = [0.0 for _ in range(self.n_population)]
            running_corrects = [0 for _ in range(self.n_population)]
            running_total = 0
            if self.verbose:
                logger.logging(f"Generation #{g}:")
                logger.logging(f"|___{get_current_time()}\tpre-SGD")
            for s in range(self.sgds_per_gen):
                # for j, data in enumerate(train_loader, 0):
                for data in train_loader:
                    x, y = data
                # for (x, y) in train_loader:
                    x = x.to(self.device)
                    y = y.to(self.device)
                    running_total += x.size(0)
                    # for i, (ind, opt) in enumerate(zip(curr_gen, optimizers)):
                    i = 0
                    for ind in curr_gen:
                        grad_step_counter += 1
                        if grad_step_counter % (100 * 5) == 0:
                            history.append(get_best_test_loss(curr_gen))
                        #     # print(history)

                        # opt = optim.SGD(ind.parameters(), lr=input_lr)
                        opt = optimizers[i]
                        opt.zero_grad()
                        # opt = opt.to(self.device)

                        ind = ind.to(self.device)
                        out = ind(x)

                        ## adding l2 regularization to loss
                        # l2_lambda = 0.001
                        # l2_norm = sum(p.pow(2.0).sum() for p in ind.parameters())

                        loss = self.fitness_function(out, y)

                        # loss_regularized = loss + (l2_norm * l2_lambda)

                        loss.backward()
                        opt.step()
                        
                        running_losses[i] += loss.item() * x.size(0)
                        running_corrects[i] += (out.max(dim=1)[1] == y).sum().item()
                        i += 1

            running_losses = list(map(lambda x: x / running_total, running_losses))
            running_accs = list(map(lambda x: x / running_total, running_corrects))
            if self.verbose:
                logger.logging(f"|___{get_current_time()}\tpost-SGD")
                logger.logging(f"\t|___population best fitness: {min(running_losses)}")
                logger.logging(f"\t|___population average fitness: {sum(running_losses) / len(running_losses)}")
                logger.logging(f"\t|___population best accuracy: {max(running_accs)}")
                logger.logging(f"\t|___population average accuracy: {sum(running_accs) / len(running_accs)}")
            if self.verbose:
                logger.logging(f"|___{get_current_time()}\tpre-evolution")
            curr_mix = [
                np.random.choice(self.n_population, size=self.mixing_number)
                for _ in range(int(self.reproductive_factor * self.n_population))
            ]



            # GENETIC STEP:

            offsprings = []
            for e in range(self.evos_per_gen):
                for mix in curr_mix:
                    model = self.model_class().to(self.device)
                    for p_child, *p_parents in zip(model.parameters(), *[curr_gen[idx].parameters() for idx in mix]):
                        p_child.data = functools.reduce(lambda x, y: x + y, p_parents) / self.mixing_number
                        # p_child.data.add_(1 / g * self.mutation_length * (2 * torch.rand_like(p_child) - 1))
                        p_child.data.add_(1 / g * self.mutation_length * torch.randn_like(p_child))
                    offsprings.append(model)
            train_losses = [0.0 for _ in range(int(self.n_population * (1 + self.reproductive_factor)))]
            train_corrects = [0 for _ in range(int(self.n_population * (1 + self.reproductive_factor)))]
            if test_set is not None:
                test_losses = [0.0 for _ in range(int(self.n_population * (1 + self.reproductive_factor)))]
                test_corrects = [0 for _ in range(int(self.n_population * (1 + self.reproductive_factor)))]
                test_total = 0
            curr_gen.extend(offsprings)
            with torch.no_grad():
                for ind in curr_gen:
                    ind = ind.to(self.device)
                    ind.eval()
                for (x, y) in train_loader:
                    x = x.to(self.device)
                    y = y.to(self.device)
                    for i, ind in enumerate(curr_gen):
                        out = ind(x)
                        train_losses[i] += self.fitness_function(out, y).item() * x.size(0)
                        train_corrects[i] += (out.max(dim=1)[1] == y).sum().item()
                if test_set is not None:
                    for (x, y) in test_loader:
                        x = x.to(self.device)
                        y = y.to(self.device)
                        test_total += x.size(0)
                        for i, ind in enumerate(curr_gen):
                            out = ind(x)
                            test_losses[i] += self.fitness_function(out, y).item() * x.size(0)
                            test_corrects[i] += (out.max(dim=1)[1] == y).sum().item()
                for ind in curr_gen:
                    ind.train()
            train_losses = list(map(lambda x: x / running_total, train_losses))
            train_accs = list(map(lambda x: x / running_total, train_corrects))
            if test_set is not None:
                test_losses = list(map(lambda x: x / test_total, test_losses))
                test_accs = list(map(lambda x: x / test_total, test_corrects))
            curr_rank = np.argsort(train_losses)
            elite = curr_rank[:self.m_elite]
            others = np.random.choice(len(curr_gen) - self.m_elite,
                                      size=self.n_population - self.m_elite) + self.m_elite
            others = curr_rank[others]
            curr_gen = [curr_gen[idx] for idx in np.concatenate([elite, others])]
            train_losses = [train_losses[idx] for idx in np.concatenate([elite, others])]
            train_accs = [train_accs[idx] for idx in np.concatenate([elite, others])]
            if test_set is not None:
                test_losses = [test_losses[idx] for idx in np.concatenate([elite, others])]
                test_accs = [test_accs[idx] for idx in np.concatenate([elite, others])]
            if self.verbose:
                logger.logging(f"|___{get_current_time()}\tpost-EVO")
                logger.logging(f"\t|___population best fitness: {min(train_losses)}")
                logger.logging(f"\t|___population average fitness: {sum(train_losses) / len(train_losses)}")
                logger.logging(f"\t|___population best accuracy: {max(train_accs)}")
                logger.logging(f"\t|___population average accuracy: {sum(train_accs) / len(train_accs)}")
                if test_set is not None:
                    logger.logging(f"\t|___(test) population best test fitness: {min(test_losses)}")
                    logger.logging(f"\t|___(test) population average test fitness: {sum(test_losses) / len(test_losses)}")
                    logger.logging(f"\t|___(test) population best accuracy: {max(test_accs)}")
                    logger.logging(f"\t|___(test) population average test accuracy: {sum(test_accs) / len(test_accs)}")



            results.append({
                "train_losses": train_losses,
                "train_accs": train_accs,
                "test_losses": test_losses,
                "test_accs": test_accs
            })


        time_lapsed = time.time() - start_time




        # # # EVALUATING LOSSES
        batch_size_evaluate = 64
        np.random.seed(0)
        eIndices = (np.random.choice(4999, size = (batch_size_evaluate*25, ), replace=False))

        evaluate_test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size_evaluate, sampler=SubsetRandomSampler(eIndices))
        evaluate_train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size_evaluate, sampler=SubsetRandomSampler(eIndices))

        train_losses_pd_comparison, test_losses_pd_comparison = [], []

        for ind in curr_gen:
            ind = ind.cpu()
            all_test_loss, all_train_loss = [], []
            counter, tl = 0, 0

            for data in evaluate_test_loader:
                counter += 1
                images, labels = data
                images = images.cpu()
                labels = labels.cpu()

                ind.eval()
                y_pred_test = ind(images)

                test_loss = self.fitness_function(y_pred_test, labels)
                # print(test_loss)
                tl += test_loss
                # all_test_loss.append(test_loss)

            print("test_loss: %s" % test_loss)
            test_avg = tl / counter
            print("test average: %s" % test_avg)

            test_losses_pd_comparison.append(test_avg)
            

            counter, tl = 0, 0

            for data in evaluate_train_loader:
                counter += 1
                images, labels = data
                ind.eval()
                y_pred_train = ind(images)

                train_loss = self.fitness_function(y_pred_train, labels)
                tl += train_loss
                # all_train_loss.append(train_loss)

            print("train_loss: %s" % train_loss), print("")
            train_avg = tl / counter
            train_losses_pd_comparison.append(train_avg)

        best_test_model_loss = min(test_losses_pd_comparison)
        best_train_model_loss = min(train_losses_pd_comparison)

        
        model_num = "4_no_reg"


        data = [[best_test_model_loss, best_train_model_loss, self.n_generations, self.n_population, self.sgds_per_gen, self.evos_per_gen, self.reproductive_factor, self.m_elite, model_num, time_lapsed, self.random_state]]

        with open('../ESGD_FMNIST_Benchmark', 'a', newline = '') as file:
            writer = csv.writer(file)
            writer.writerows(data)

        graph = False
        
        if graph:
            graph_history(history)


        return results


if __name__ == "__main__":
    from .models.cnn import CNN
    from torchvision import datasets, transforms

    HPSET = {
        "lr": (0.01, 0.05, 0.1),
        "momentum": (0.8, 0.9, 0.99),
        "nesterov": (False, True)
    }
    LOG_DIR = os.path.expanduser("~/esgd-ws/log/esgd")
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    esgd = ESGD(
        hpset=HPSET,
        model_class=CNN
    )
    esgd.train(
        train_data,
        train_targets,
        test_set=test_set,
        log_file=f"{LOG_DIR}/{get_current_time()}.log"
    )
