import lightning.pytorch as pl
from lightning_model import Encoder, Decoder, LitAutoEncoder, ImagenetTransferLearning
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader 
import os
import torch
import torch.utils.data as data
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from argparse import ArgumentParser

def train_and_save():

    # Load datasets
    transform = transforms.ToTensor()
    train_set = datasets.MNIST(root="MNIST", download=True, train=True, transform=transform)
    test_set = datasets.MNIST(root="MNIST", download=True, train=False, transform=transform)

    # use 20% of training data for validation
    train_set_size = int(len(train_set) * 0.8)
    valid_set_size = len(train_set) - train_set_size

    # split the train set into two
    seed = torch.Generator().manual_seed(42)
    train_set, valid_set = data.random_split(train_set, [train_set_size, valid_set_size], generator=seed)
    # model
    autoencoder = LitAutoEncoder(Encoder(), Decoder())

    # train with both splits
    trainer = pl.Trainer(max_epochs=1)
    trainer.fit(model=autoencoder, train_dataloaders=DataLoader(train_set), val_dataloaders=DataLoader(valid_set))
    # test the model
    # trainer.test(model=autoencoder, dataloaders=DataLoader(test_set))
    trainer.save_checkpoint("model.ckpt")


def load_and_test():

    transform = transforms.ToTensor()
    test_set = datasets.MNIST(root="MNIST", download=True, train=False, transform=transform)
    autoencoder = LitAutoEncoder.load_from_checkpoint("C:/Users/student/master-project/model.ckpt")
    # C:/Users/student/masters_project/lightning_logs/version_5/checkpoints/epoch=0-step=48000.ckpt
    trainer = pl.Trainer(max_epochs=1)
    trainer.test(model=autoencoder, dataloaders=test_set)
    checkpoint = torch.load("C:/Users/student/masters_project/model.ckpt")
    print(checkpoint["hyper_parameters"])

def early_stop(args):

    transform = transforms.ToTensor()
    train_set = datasets.MNIST(root="MNIST", download=True, train=True, transform=transform)

    # use 20% of training data for validation
    train_set_size = int(len(train_set) * args.percent_split)
    valid_set_size = len(train_set) - train_set_size

    # split the train set into two
    seed = torch.Generator().manual_seed(42)
    train_set, valid_set = data.random_split(train_set, [train_set_size, valid_set_size], generator=seed)

    autoencoder = LitAutoEncoder(Encoder(), Decoder())

    trainer = pl.Trainer(callbacks=[EarlyStopping(monitor="val_loss", mode="min")], fast_dev_run=args.fast_dev_run, max_epochs=args.max_epochs)

    trainer.fit(model=autoencoder, train_dataloaders=DataLoader(train_set, num_workers=7), val_dataloaders=DataLoader(valid_set, num_workers=7))

def image_net():

    transform = transforms.ToTensor()
    # Fit
    # train_set = datasets.CIFAR10(root="CIFAR10", download=True, train=True, transform=transform)
    model = ImagenetTransferLearning()
    trainer = pl.Trainer(max_epochs=1)
    trainer.fit(model)#, train_dataloaders=DataLoader(train_set))

def image_net_predict():

    # Predict
    transform = transforms.ToTensor()
    test_set = datasets.CIFAR10(root="CIFAR10", download=True, train=False, transform=transform)
    batch_size = 4
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=2)
    # C:\Users\student\master-project\lightning_logs\version_18\checkpoints\epoch=0-step=50000.ckpt
    model = ImagenetTransferLearning.load_from_checkpoint("C:/Users/student/master-project/lightning_logs/version_18/checkpoints/epoch=0-step=50000.ckpt")
    model.freeze()
    predictions = model(test_loader)
    
    

if __name__ == '__main__':
    parser = ArgumentParser()

    # trainer arguments
    parser.add_argument("--max_epochs", type=int, default=1)
    parser.add_argument("--percent_split", type=float, default=0.8)
    parser.add_argument("--fast_dev_run", type=int, default=0)

    args = parser.parse_args()
    early_stop(args)