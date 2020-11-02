import comet_ml  # noqa: F401
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
import numpy as np
from loguru import logger

from ..model_wrapper import ModelWrapper
from . import data_utils
from .. import utils

# TODO: This isn't actually used anymore. See if we still need it at some point or it
#   can be deleted.
# torchvision_models = [
#     "alexnet",
#     "vgg11",
#     "vgg11_bn",
#     "vgg13",
#     "vgg13_bn",
#     "vgg16",
#     "vgg16_bn",
#     "vgg19",
#     "vgg19_bn",
#     "resnet18",
#     "resnet34",
#     "resnet50",
#     "resnet101",
#     "resnet152",
#     "squeezenet1_0",
#     "squeezenet1_1",
#     "densenet121",
#     "densenet169",
#     "densenet161",
#     "densenet201",
#     "inception_v3",
#     "googlenet",
#     "shufflenet_v2_x0_5",
#     "shufflenet_v2_x1_0",
#     "shufflenet_v2_x1_5",
#     "shufflenet_v2_x2_0",
#     "mobilenet_v2",
#     "resnext50_32x4d",
#     "resnext101_32x8d",
#     "wide_resnet50_2",
#     "wide_resnet101_2",
#     "mnasnet0_5",
#     "mnasnet0_75",
#     "mnasnet1_0",
#     "mnasnet1_3",
# ]


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

    def _create_model(self, num_classes) -> None:
        """Create the model based on self.model_name and store it in self.model."""
        if self.model_name == "simple-cnn":
            self.model = SimpleCnn()
        else:
            # Raise error if pretrained model doesn't have 1000 classes.
            pretrained = self.config.get("pretrained", False)
            if pretrained and num_classes != 1000:
                raise ValueError(
                    "Using a pretrained model requires config parameter num_classes "
                    f"to be 1000, is: {num_classes}"
                )

            # Create model.
            # TODO: Pass additional kwargs on to model (is this possible?).
            self.model = getattr(torchvision.models, self.model_name)(
                pretrained=pretrained, num_classes=num_classes
            )

    def _create_optimizer(self) -> optim.Optimizer:
        """Create the optimizer based on the config"""
        model_params = self.model.parameters()
        optimizer_name = self.config.get("optimizer", "adam")
        if optimizer_name == "adam":
            kwargs = utils.filter_dict(
                self.config, ["lr", "betas", "eps", "weight_decay", "amsgrad"]
            )
            return optim.Adadelta(model_params, **kwargs)
        elif optimizer_name == "adadelta":
            kwargs = utils.filter_dict(
                self.config, ["lr", "rho", "eps", "weight_decay"]
            )
            return optim.Adadelta(model_params, **kwargs)
        elif optimizer_name == "adagrad":
            kwargs = utils.filter_dict(
                self.config,
                ["lr", "lr_decay", "weight_decay", "initial_accumulator_value", "eps"],
            )
            return optim.Adagrad(model_params, **kwargs)
        elif optimizer_name == "rmsprop":
            kwargs = utils.filter_dict(
                self.config,
                ["lr", "alpha", "eps", "weight_decay", "momentum", "centered"],
            )
            return optim.RMSprop(model_params, **kwargs)
        elif optimizer_name == "sgd":
            kwargs = utils.filter_dict(
                self.config, ["momentum", "dampening", "weight_decay", "nesterov"]
            )
            # In contrast to other optimizers, lr is a required param for SGD.
            return optim.SGD(model_params, lr=self.config.get("lr", 0.1), **kwargs)
        else:
            raise ValueError(f"Optimizer not known: {optimizer_name}")

        # TODO: Implement other optimizers.

    def _train(
        self,
        train_data,
        val_data,
        test_data,
        writer,
        experiment,
        dry_run: bool = False,
    ) -> None:

        use_cuda = torch.cuda.is_available()

        logger.info("Preprocessing datasets...")
        # Get number of classes from config or infer from train_data.
        if "num_classes" in self.config:
            num_classes = self.config["num_classes"]
        else:
            num_classes = data_utils.get_num_classes(train_data)

        # Preprocess all datasets.
        # TODO: mean and std normalization applies only to pretrained models. But
        #   doesn't this also make sense for not-pretrained? Or should I scale to mean
        #   0 and std 1 here? See also the code here:
        #   https://pytorch.org/docs/stable/torchvision/models.html
        train_data = data_utils.to_torch(
            train_data,
            resize=256,
            crop=224,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        val_data = data_utils.to_torch(
            train_data,
            resize=256,
            crop=224,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        test_data = data_utils.to_torch(
            train_data,
            resize=256,
            crop=224,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        # TODO: Maybe print some more stuff about the data.
        logger.info(f"Train data: {len(train_data)} samples")
        val_desc = "Not given" if val_data is None else f"{len(val_data)} samples"
        logger.info(f"Val data:   {val_desc}")
        test_desc = "Not given" if test_data is None else f"{len(test_data)} samples"
        logger.info(f"Test data:  {test_desc}")
        logger.info(f"Found {num_classes} different classes")
        logger.info("")

        kwargs = {"batch_size": self.config.get("batch_size", 128)}
        if use_cuda:
            kwargs["pin_memory"] = True
            kwargs["num_workers"] = 1

        train_loader = DataLoader(train_data, shuffle=True, **kwargs)
        val_loader = DataLoader(val_data, **kwargs) if val_data is not None else None
        test_loader = DataLoader(test_data, **kwargs) if test_data is not None else None

        # Set up model, optimizer, loss.
        logger.info("Creating model...")
        device = torch.device("cuda" if use_cuda else "cpu")
        self._create_model(num_classes)
        optimizer = self._create_optimizer()
        loss_func = nn.CrossEntropyLoss()

        # Configure trainer and metrics.
        trainer = create_supervised_trainer(
            self.model, optimizer, loss_func, device=device
        )
        val_metrics = {"accuracy": Accuracy(), "loss": Loss(loss_func)}
        evaluator = create_supervised_evaluator(
            self.model, metrics=val_metrics, device=device
        )

        @trainer.on(
            Events.ITERATION_COMPLETED(every=self.config.get("print_every", 100))
        )
        def log_training_loss(trainer):
            batch = (trainer.state.iteration - 1) % trainer.state.epoch_length + 1
            logger.info(
                f"Epoch {trainer.state.epoch}, "
                f"batch {batch} / {trainer.state.epoch_length}: "
                f"Loss: {trainer.state.output:.3f}"
            )

        # TODO: This iterates over complete train set again, maybe accumulate as in the
        #   example in the footnote here: https://pytorch.org/ignite/quickstart.html#
        @trainer.on(Events.EPOCH_COMPLETED)
        def log_training_results(trainer):
            evaluator.run(train_loader)
            metrics = evaluator.state.metrics
            logger.info(
                f"Training results - Epoch: {trainer.state.epoch}  "
                f"Avg accuracy: {metrics['accuracy']:.3f} "
                f"Avg loss: {metrics['loss']:.3f}"
            )
            experiment.log_metric("train_loss", metrics["loss"])
            experiment.log_metric("train_accuracy", metrics["accuracy"])
            writer.add_scalar("train_loss", metrics["loss"], trainer.state.epoch)
            writer.add_scalar(
                "train_accuracy", metrics["accuracy"], trainer.state.epoch
            )

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_validation_results(trainer):
            if val_loader:
                evaluator.run(val_loader)
                metrics = evaluator.state.metrics
                logger.info(
                    f"Validation results - Epoch: {trainer.state.epoch} "
                    f"Avg accuracy: {metrics['accuracy']:.3f} "
                    f"Avg loss: {metrics['loss']:.3f}"
                )
                experiment.log_metric("val_loss", metrics["loss"])
                experiment.log_metric("val_accuracy", metrics["accuracy"])
                writer.add_scalar("val_loss", metrics["loss"], trainer.state.epoch)
                writer.add_scalar(
                    "val_accuracy", metrics["accuracy"], trainer.state.epoch
                )

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_test_results(trainer):
            if test_loader:
                evaluator.run(test_loader)
                metrics = evaluator.state.metrics
                logger.info(
                    f"Test results - Epoch: {trainer.state.epoch} "
                    f"Avg accuracy: {metrics['accuracy']:.3f} "
                    f"Avg loss: {metrics['loss']:.3f}"
                )
                experiment.log_metric("test_loss", metrics["loss"])
                experiment.log_metric("test_accuracy", metrics["accuracy"])
                writer.add_scalar("test_loss", metrics["loss"], trainer.state.epoch)
                writer.add_scalar(
                    "test_accuracy", metrics["accuracy"], trainer.state.epoch
                )

        @trainer.on(Events.EPOCH_COMPLETED)
        def checkpoint_model(trainer):
            checkpoint_dir = (
                self.out_dir / "checkpoints" / f"epoch{trainer.state.epoch}"
            )
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            torch.save(self.model, checkpoint_dir / "model.pt")

        # Start training.
        max_epochs = 1 if dry_run else self.config.get("epochs", 5)
        epoch_length = 1 if dry_run else None
        logger.info(f"Training model on device {device}... (this may take a while)")
        trainer.run(train_loader, max_epochs=max_epochs, epoch_length=epoch_length)
        logger.info("Training finished!")

        # Save the trained model.
        torch.save(self.model, self.out_dir / "model.pt")

    def _load(self) -> None:
        """Loads the model from file."""
        self.model = torch.load(self.out_dir / "model.pt")

    def predict(self, image) -> dict:
        """Runs data through the model and returns output."""

        # Convert data format if required.
        image_format = data_utils.recognize_image_format(image)
        if image_format == "files":
            image = data_utils.load_image(
                image,
                to_numpy=False,
                resize=256,
                crop=224,
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
        elif image_format == "numpy":
            # TODO: This is almost the same code as in data_utils.numpy_to_torch,
            #   maybe refactor it.

            # Rescale images to 0-255 and convert to uint8.
            # TODO: This should probably be done with the same min/max values as train
            #   set, see note in data_utils.numpy_to_torch.
            image = (image - np.min(image)) / np.ptp(image) * 255
            image = image.astype(np.uint8)

            # If images are grayscale, convert to RGB by duplicating channels.
            if image.shape[0] == 1:
                image = np.stack((image[0],) * 3, axis=0)

            # Convert to channels last format (required for transforms.ToPILImage).
            image = image.transpose((1, 2, 0))

            # Set up transform to convert to PIL image, do manipulations, and convert
            # to tensor.
            # TODO: Converting to PIL and then to tensor is not super efficient, find
            #   a better method.
            transform = data_utils.create_transform(
                from_numpy=True,
                resize=256,
                crop=224,
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
            image = transform(image)
        else:
            raise RuntimeError()

        # Wrap image in batch.
        image_batch = image.unsqueeze(dim=0)

        self.model.eval()
        with torch.no_grad():
            output = self.model(image_batch)[0]
            probabilities = torch.softmax(output, dim=0).numpy()
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
