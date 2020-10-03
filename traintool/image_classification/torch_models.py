import comet_ml
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import torchvision
from torch.optim.lr_scheduler import StepLR
from typing import Union

# from loguru import logger

from ..model_wrapper import ModelWrapper
from . import data_utils
from .. import utils


torchvision_models = [
    "alexnet",
    "vgg11",
    "vgg11_bn",
    "vgg13",
    "vgg13_bn",
    "vgg16",
    "vgg16_bn",
    "vgg19",
    "vgg19_bn",
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "squeezenet1_0",
    "squeezenet1_1",
    "densenet121",
    "densenet169",
    "densenet161",
    "densenet201",
    "inception_v3",
    "googlenet",
    "shufflenet_v2_x0_5",
    "shufflenet_v2_x1_0",
    "shufflenet_v2_x1_5",
    "shufflenet_v2_x2_0",
    "mobilenet_v2",
    "resnext50_32x4d",
    "resnext101_32x8d",
    "wide_resnet50_2",
    "wide_resnet101_2",
    "mnasnet0_5",
    "mnasnet0_75",
    "mnasnet1_0",
    "mnasnet1_3",
]


class SimpleCnn(nn.Module):
    """
    A simple CNN module for MNIST, similar to pytorch's MNIST example.
    
    See: https://github.com/pytorch/examples/blob/master/mnist/main.py
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


class TorchImageClassificationWrapper(ModelWrapper):
    """
    This wrapper handles torch models for image classification.

    It can either support models from torchvision.models (resnet18, alexnet, ...) or a 
    simple CNN for MNIST (simple-cnn, see SimpleCnn class).
    """

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.model = None

    def _create_model(self, config: dict) -> None:
        """Create the model based on self.model_name and store it in self.model."""
        if self.model_name == "simple-cnn":
            self.model = SimpleCnn()
        else:
            pretrained = config.get("pretrained", False)
            self.model = getattr(torchvision.models, self.model_name)(
                pretrained=pretrained
            )

    def _preprocess(
        self, data, config: dict, train: bool = False, use_cuda: bool = False
    ):
        # Handle empty dataset
        if data is None:
            return None

        # Convert to torch dataset.
        data = data_utils.to_torch(data)

        # Transform and normalize according to https://pytorch.org/docs/stable/torchvision/models.html
        # Note that inception_v3 requires 3 x 299 x 299
        # TODO

        # Wrap in data loader.
        kwargs = {"batch_size": config.get("batch_size", 128)}
        if train:
            kwargs["shuffle"] = True
        if use_cuda:
            kwargs["pin_memory"] = True
            kwargs["num_workers"] = 1
        data_loader = DataLoader(data, **kwargs)

        return data_loader

    def _create_optimizer(self, config: dict, params) -> optim.Optimizer:
        """Create the optimizer based on the config"""
        optimizer_name = config.get("optimizer", "adam")
        if optimizer_name == "adam":
            kwargs = utils.filter_dict(
                config, ["lr", "betas", "eps", "weight_decay", "amsgrad"]
            )
            return optim.Adadelta(params, **kwargs)
        elif optimizer_name == "adadelta":
            kwargs = utils.filter_dict(config, ["lr", "rho", "eps", "weight_decay"])
            return optim.Adadelta(params, **kwargs)
        elif optimizer_name == "adagrad":
            kwargs = utils.filter_dict(
                config,
                ["lr", "lr_decay", "weight_decay", "initial_accumulator_value", "eps"],
            )
            return optim.Adagrad(params, **kwargs)
        elif optimizer_name == "rmsprop":
            kwargs = utils.filter_dict(
                config, ["lr", "alpha", "eps", "weight_decay", "momentum", "centered"]
            )
            return optim.RMSprop(params, **kwargs)
        elif optimizer_name == "sgd":
            kwargs = utils.filter_dict(
                config, ["momentum", "dampening", "weight_decay", "nesterov"]
            )
            # In contrast to other optimizers, lr is a required param for SGD.
            return optim.SGD(params, lr=config.get("lr", 0.1), **kwargs)
        else:
            raise ValueError(f"Optimizer not known: {optimizer_name}")

        # TODO: Implement other optimizers.

    def train(
        self,
        train_data,
        val_data,
        test_data,
        config: dict,
        out_dir: Path,
        writer,
        experiment,
        dry_run: bool = False,
    ) -> None:
        """Trains the model, evaluates it on val/test data and saves it to file."""

        self.out_dir = out_dir

        use_cuda = torch.cuda.is_available()

        # Preprocess all datasets.
        train_loader = self._preprocess(
            train_data, config, train=True, use_cuda=use_cuda
        )
        val_loader = self._preprocess(val_data, config, use_cuda=use_cuda)
        test_loader = self._preprocess(test_data, config, use_cuda=use_cuda)

        # # Convert data to torch dataset.
        # train_data, test_data = data_utils.to_torch(train_data, test_data)

        # # TODO: Change kwargs if running on GPU.
        # # Create data loaders.
        # kwargs = {"batch_size": config["batch_size"]}
        # train_loader = torch.utils.data.DataLoader(train_data, **kwargs)
        # test_loader = torch.utils.data.DataLoader(test_data, **kwargs)

        # Set up everything for training and iterate through epochs.
        device = torch.device("cuda" if use_cuda else "cpu")
        self._create_model(config)
        optimizer = self._create_optimizer(config, self.model.parameters())
        loss_func = nn.CrossEntropyLoss()

        # TODO: Re-enable scheduler.
        # scheduler = StepLR(optimizer, step_size=1, gamma=config.get("gamma", 0.7))
        for epoch in range(1, config.get("epochs", 5) + 1):
            self._train_epoch(
                config=config,
                device=device,
                train_loader=train_loader,
                optimizer=optimizer,
                loss_func=loss_func,
                epoch=epoch,
                experiment=experiment,
                dry_run=dry_run,
            )
            if val_data is not None:
                self._evaluate(
                    device=device,
                    loader=val_loader,
                    loss_func=loss_func,
                    experiment=experiment,
                    is_val=True,
                )
            if test_data is not None:
                self._evaluate(
                    device=device,
                    loader=test_loader,
                    loss_func=loss_func,
                    experiment=experiment,
                )
            
            # scheduler.step()

            # TODO: Checkpoint model.
            torch.save(self.model, out_dir / "model.pt")

            if dry_run:
                break

    def _train_epoch(
        self,
        config: dict,
        device: torch.device,
        train_loader: DataLoader,
        optimizer: optim.Optimizer,
        loss_func,
        epoch: int,
        experiment,
        dry_run: bool,
    ) -> None:
        with experiment.train():
            self.model.train()
            for batch, (inputs, labels) in enumerate(train_loader):

                # Move data to cpu or gpu.
                inputs, labels = inputs.to(device), labels.to(device)

                # Reset optimizer.
                optimizer.zero_grad()

                # Run inputs through network.
                output = self.model(inputs)
                loss = loss_func(output, labels)
                pred_labels = output.argmax(dim=1, keepdim=True)
                correct = pred_labels.eq(labels.view_as(pred_labels)).sum().item()
                acc = 100 * correct / len(inputs)

                # Backprop and optimize.
                loss.backward()
                optimizer.step()  # this somehow tracks the loss already

                if batch % 10 == 0:
                    # Track metrics to comet.ml
                    # Loss is already tracked through optimizer.step above even though I don't
                    # really get how.
                    # experiment.log_metric("loss", loss.item(), step=batch)
                    experiment.log_metric("accuracy", acc, step=batch)

                    # Print metrics.
                    # TODO: This currently tracks the metrics only for the current batch.
                    #   Use running metrics instead, maybe using pytorch lightning or Ignite or fast.ai.
                    print(
                        f"Train epoch {epoch}, batch {batch}/{len(train_loader.dataset)}:\tLoss: {loss.item():.3f}\tAccuracy: {acc:.3f}"
                    )

                if dry_run:
                    break

    def _evaluate(
        self,
        device: torch.device,
        loader: DataLoader,
        loss_func,
        experiment,
        is_val: bool = False,
    ) -> None:
        cm = experiment.validate() if is_val else experiment.train()
        with cm:
            self.model.eval()
            running_loss = 0
            correct = 0
            with torch.no_grad():
                for inputs, labels in loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    output = self.model(inputs)
                    running_loss += (
                        loss_func(output, labels).mean().item()
                    )  # sum up batch loss
                    pred_labels = output.argmax(dim=1, keepdim=True)
                    correct += pred_labels.eq(labels.view_as(pred_labels)).sum().item()

            running_loss /= len(loader.dataset)
            acc = 100 * correct / len(loader.dataset)

            # Track metrics to comet.ml
            experiment.log_metric("loss", running_loss)
            experiment.log_metric("accuracy", correct / len(loader.dataset))

            print(
                "Val:" if is_val else "Val:",
                f"Loss: {running_loss:.3f}, Accuracy: {acc:.3f}",
            )

    # def save(self, out_dir: Union[Path, str]):
    #     """Saves the model to file."""
    #     out_dir = Path(out_dir)
    #     torch.save(self.model, out_dir / "model.pt")

    @classmethod
    def load(cls, out_dir: Path, model_name: str):
        """Loads the model from file."""
        wrapper = cls(model_name)
        wrapper.model = torch.load(out_dir / "model.pt")
        return wrapper

    def predict(self, data) -> dict:
        """Runs data through the model and returns output."""
        # TODO: Raise error if self.model is None.
        self.model.eval()
        with torch.no_grad():
            inp = torch.from_numpy(data).float().unsqueeze(dim=0)
            output = self.model(inp)[0]
            probabilities = torch.softmax(output, dim=0).numpy().tolist()
            predicted_class = output.argmax().item()
        return {"predicted_class": predicted_class, "probabilities": probabilities}

    def raw(self) -> dict:
        """Returns the raw model object."""
        return {"model": self.model}

    # @staticmethod
    # def default_config(model_name: str) -> dict:
    #     # TODO: Implement other models.
    #     if model_name == "simple-cnn":
    #         return {"lr": 0.1}
    #     else:
    #         raise NotImplementedError()
