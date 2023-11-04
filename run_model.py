import lightning.pytorch as pl
from lightning_model import Encoder, Decoder, LitAutoEncoder
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader 
import os
import torch
import torch.utils.data as data

def main():
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
    trainer.test(model=autoencoder, dataloaders=DataLoader(test_set))

if __name__ == '__main__':
    main()