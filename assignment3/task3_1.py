import pathlib
import matplotlib.pyplot as plt
from torch.nn.modules.activation import ReLU
import utils
import torch
from torch import nn
import torch.optim as optim
from dataloaders import load_cifar10
from trainer import Trainer, compute_loss_and_accuracy


class Conv_with_pool(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding) -> None:
        super(Conv_with_pool, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2 ,stride=2)
        )
    
    def forward(self, x):
        return self.layer(x)

class Conv_conv_pool(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding) -> None:
        super(Conv_conv_pool, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),                               
            nn.ReLU(inplace=True),                                      
            nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
    
    def forward(self, x):
        return self.layer(x)



class ExampleModel(nn.Module):

    def __init__(self,
                 image_channels,
                 num_classes):
        """
            Is called when model is initialized.
            Args:
                image_channels. Number of color channels in image (3)
                num_classes: Number of classes we want to predict (10)
        """
        super().__init__()
        # TODO: Implement this function (Task  2a)
        num_filters = [16, 32, 64]  # Set number of filters in first conv layer
        kernel_size = 3
        self.padding = 1


        self.num_classes = num_classes

        conv_list = []
        in_channels = image_channels
        for filter_size in num_filters:
            module = Conv_conv_pool(
                in_channels, 
                filter_size, 
                kernel_size,
                stride=1,
                padding=self.padding)
            conv_list.append(module)
            in_channels = filter_size
        self.new_feature_extractor = nn.Sequential(*conv_list)

        self.num_output_features = num_filters[-1]*4*4
        # Initialize our last fully connected layer
        # Inputs all extracted features from the convolutional layers
        # Outputs num_classes predictions, 1 for each class.
        # There is no need for softmax activation function, as this is
        # included with nn.CrossEntropyLoss
        self.classifier = nn.Sequential(
            nn.Linear(in_features=self.num_output_features, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=num_classes),
        )

    def forward(self, x):
        """
        Performs a forward pass through the model
        Args:
            x: Input image, shape: [batch_size, 3, 32, 32]
        """
        # TODO: Implement this function (Task  2a)
        batch_size = x.shape[0]

        x = self.new_feature_extractor(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        out = x
        expected_shape = (batch_size, self.num_classes)
        assert out.shape == (batch_size, self.num_classes),\
            f"Expected output of forward pass to be: {expected_shape}, but got: {out.shape}"
        return out


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
    # Set the random generator seed (parameters, shuffling etc).
    # You can try to change this and check if you still get the same result! 
    utils.set_seed(0)
    epochs = 10
    batch_size = 64
    learning_rate = 5e-2
    early_stop_count = 4
    dataloaders = load_cifar10(batch_size)
    optimizer = optim.SGD
    weight_decay = 0
    model = ExampleModel(image_channels=3, num_classes=10)
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
    create_plots(trainer, "task3_1")
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

