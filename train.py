import random
import time
import argparse

import numpy as np
import torch
import torch.optim as optim
from utils import accuracy
from models import GCN
from utils import load_data

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="cora",
                    help='dataset for training')
parser.add_argument('--times', type=int, default=1,
                    help='times of repeat training')
parser.add_argument('--seed', type=int, default=33, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')

args = parser.parse_args()

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
criterion = torch.nn.NLLLoss()
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def train(epoch, model, optimizer, adj, features, labels, idx_train, idx_val):
    t = time.time()
    model.train()
    optimizer.zero_grad()

    output = model(features, adj)
    loss_train = criterion(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    with torch.no_grad():
        model.eval()
        output = model(features, adj)

        loss_val = criterion(output[idx_val], labels[idx_val])
        acc_val = accuracy(output[idx_val], labels[idx_val])

    print(f'Epoch: {epoch + 1:04d}',
          f'loss_train: {loss_train.item():.4f}',
          f'acc_train: {acc_train:.4f}',
          f'loss_val: {loss_val.item():.4f}',
          f'acc_val: {acc_val:.4f}',
          f'time: {time.time() - t:.4f}s')
    return loss_val


@torch.no_grad()
def test(model, adj, features, labels, idx_test):
    model.eval()
    output = model(features, adj)
    loss_test = criterion(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print(f"Test set results:",
          f"loss= {loss_test.item():.4f}",
          f"accuracy= {acc_test:.4f}")
    return acc_test


def main(dataset, times):
    adj, features, labels, idx_train, idx_val, idx_test = load_data(dataset)
    features = features.to(device)
    adj = adj.to(device)
    labels = labels.to(device)
    idx_train = idx_train.to(device)
    idx_val = idx_val.to(device)
    idx_test = idx_test.to(device)

    nclass = labels.max().item() + 1

    acc_lst = list()
    for seed in random.sample(range(0, 100000), times):
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        # Model and optimizer
        model = GCN(nfeat=features.shape[1],
                    nhid=args.hidden,
                    nclass=nclass,
                    dropout=args.dropout)
        optimizer = optim.Adam(model.parameters(),
                               lr=args.lr, weight_decay=args.weight_decay)
        model.to(device)

        # Train model
        t_total = time.time()
        for epoch in range(args.epochs):
            train(epoch, model, optimizer, adj, features, labels, idx_train, idx_val)
        print(f"Total time elapsed: {time.time() - t_total:.4f}s")

        # Testing
        acc_lst.append(test(model, adj, features, labels, idx_test))

    print(acc_lst)
    print(np.mean(acc_lst))


if __name__ == '__main__':
    main(dataset=args.dataset, times=args.times)
