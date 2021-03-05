import torchvision
import pathlib
import matplotlib.pyplot as plt
from torch.nn.modules.activation import ReLU
import utils
import torch
from torch import nn
import torch.optim as optim
from dataloaders_task4 import load_cifar10
from trainer_task4 import Trainer, compute_loss_and_accuracy

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torchvision.models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(512, 10) # No need to apply softmax,
        # as this is done in nn.CrossEntropyLoss
        for param in self.model.parameters(): # Freeze all parameters
            param.requires_grad = False
        for param in self.model.fc.parameters(): # Unfreeze the last fully-connected
            param.requires_grad = True # layer
        for param in self.model.layer4.parameters(): # Unfreeze the last 5 convolutional
            param.requires_grad = True # layers

    def forward(self, x):
        x = self.model(x)
        return x


def create_plots(trainer: Trainer, name: str):
    plot_path = pathlib.Path("plots")
    plot_path.mkdir(exist_ok=True)
    # Save plots and show them
    plt.figure(figsize=(20, 8))
    plt.subplot(1, 2, 1)
    plt.title("Cross Entropy Loss")
    utils.plot_loss(trainer.train_history["loss"], label="Training loss", npoints_to_average=10)
    utils.plot_loss(trainer.validation_history["loss"], label="Validation loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.title("Accuracy")
    utils.plot_loss(trainer.validation_history["accuracy"], label="Validation Accuracy")
    plt.legend()
    plt.savefig(plot_path.joinpath(f"{name}_plot.png"))
    plt.show()



if __name__ == "__main__":
    utils.set_seed(0)
    epochs = 10
    batch_size = 32
    learning_rate = 0.0005
    early_stop_count = 4
    dataloaders = load_cifar10(batch_size)
    optimizer = optim.Adam
    weight_decay = 0
    model = Model()
    trainer = Trainer(
        batch_size,
        learning_rate,
        early_stop_count,
        epochs,
        model,
        dataloaders,
        weight_decay,
        optimizer
    )
    trainer.train()
    create_plots(trainer, "task4")
    dataloader_train, dataloader_val, dataloader_test = dataloaders
    model.eval()
    loss, train_acc = compute_loss_and_accuracy(
        dataloader_train, model, torch.nn.CrossEntropyLoss()
    )
    loss, val_acc = compute_loss_and_accuracy(
        dataloader_val, model, torch.nn.CrossEntropyLoss()
    )
    loss, test_acc = compute_loss_and_accuracy(
        dataloader_test, model, torch.nn.CrossEntropyLoss()
    )
    print("train ", train_acc)
    print("val ", val_acc)
    print("test ", test_acc)