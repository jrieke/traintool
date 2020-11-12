"""
Wrapper around pytorch image classification models.
"""

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
from . import preprocessing, visualization
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
        logger.info(f"    optimizer: {optimizer_name}")
        if optimizer_name == "adam":
            kwargs = utils.filter_dict(
                self.config, ["lr", "betas", "eps", "weight_decay", "amsgrad"]
            )
            logger.info(f"    optimizer args: {kwargs}")
            return optim.Adadelta(model_params, **kwargs)
        elif optimizer_name == "adadelta":
            kwargs = utils.filter_dict(
                self.config, ["lr", "rho", "eps", "weight_decay"]
            )
            logger.info(f"    optimizer args: {kwargs}")
            return optim.Adadelta(model_params, **kwargs)
        elif optimizer_name == "adagrad":
            kwargs = utils.filter_dict(
                self.config,
                ["lr", "lr_decay", "weight_decay", "initial_accumulator_value", "eps"],
            )
            logger.info(f"    optimizer args: {kwargs}")
            return optim.Adagrad(model_params, **kwargs)
        elif optimizer_name == "rmsprop":
            kwargs = utils.filter_dict(
                self.config,
                ["lr", "alpha", "eps", "weight_decay", "momentum", "centered"],
            )
            logger.info(f"    optimizer args: {kwargs}")
            return optim.RMSprop(model_params, **kwargs)
        elif optimizer_name == "sgd":
            kwargs = utils.filter_dict(
                self.config, ["momentum", "dampening", "weight_decay", "nesterov"]
            )
            logger.info(f"    optimizer args: {kwargs}")
            # In contrast to other optimizers, lr is a required param for SGD.
            return optim.SGD(model_params, lr=self.config.get("lr", 0.1), **kwargs)
        else:
            raise ValueError(f"Optimizer not known: {optimizer_name}")

        # TODO: Implement other optimizers.

    def _preprocess_for_training(self, name, data, use_cuda=False):
        if data is None:  # val/test can be empty´
            logger.info(f"{name}: Not given")
            return None
        else:
            logger.info(f"{name}:")

            # Get number of classes from config or infer from train_data (needs to be
            # done before the conversion!).
            if "num_classes" in self.config:
                self.num_classes = self.config["num_classes"]
            elif name == "train":
                self.num_classes = preprocessing.get_num_classes(data)

            # Convert format.
            logger.info(f"    data format: {preprocessing.recognize_data_format(data)}")
            # TODO: Properly do mean/std normalization.
            # TODO: Give error if resize/crop are in config (we always need these values
            #   here!).
            data = preprocessing.to_torch(
                data,
                resize=256,
                crop=224,
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
            logger.info(f"    resized to 256 x 256")
            logger.info(f"    center-cropped to 224 x 224")
            logger.info(f"    samples: {len(data)}")
            logger.info(f"    image shape: {tuple(data[0][0].shape)}")

            # TODO: Either write num classes for each dataset here or
            # TODO: Raise error if num_classes diverges between datasets.
            logger.info(f"    classes: {self.num_classes}")

            # Wrap in data loader.
            batch_size = self.config.get("batch_size", 128)
            kwargs = {"batch_size": batch_size}
            logger.info(f"    batch size: {batch_size}")
            if use_cuda:
                kwargs["pin_memory"] = True
                kwargs["num_workers"] = 1
            if name == "train":
                kwargs["shuffle"] = True
                logger.info(f"    shuffled")
            loader = DataLoader(data, **kwargs)
            return loader

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

        # Preprocess all datasets.
        logger.info("Preprocessing datasets...")
        train_loader = self._preprocess_for_training("train", train_data, use_cuda)
        val_loader = self._preprocess_for_training("val", val_data, use_cuda)
        test_loader = self._preprocess_for_training("test", test_data, use_cuda)
        logger.info("")
        
        # Set up model, optimizer, loss.
        logger.info("Creating model...")
        device = torch.device("cuda" if use_cuda else "cpu")
        logger.info(f"    device: {device}")
        self._create_model(self.num_classes)
        optimizer = self._create_optimizer()
        loss_func = nn.CrossEntropyLoss()
        logger.info(f"    loss function: cross-entropy")
        logger.info("")

        # Dedicate a few images that will be plotted as samples to tensorboard.
        num_samples_to_plot = self.config.get("num_samples_to_plot", 5)

        def get_samples(loader):
            if loader is None:
                return None, None
            else:
                return next(
                    iter(DataLoader(loader.dataset, batch_size=num_samples_to_plot))
                )

        train_sample_images, train_sample_labels = get_samples(train_loader)
        val_sample_images, val_sample_labels = get_samples(val_loader)
        test_sample_images, test_sample_labels = get_samples(test_loader)

        # Configure trainer and metrics.
        trainer = create_supervised_trainer(
            self.model, optimizer, loss_func, device=device
        )
        val_metrics = {
            "accuracy": Accuracy(),
            "loss": Loss(loss_func),
            # "confusion_matrix": ConfusionMatrix(num_classes),
        }
        evaluator = create_supervised_evaluator(
            self.model, metrics=val_metrics, device=device
        )

        @trainer.on(
            Events.ITERATION_COMPLETED(every=self.config.get("print_every", 100))
        )
        def log_training_loss(trainer):
            batch = (trainer.state.iteration - 1) % trainer.state.epoch_length + 1
            logger.info(
                f"Epoch {trainer.state.epoch} / {num_epochs}, "
                f"batch {batch} / {trainer.state.epoch_length}: "
                f"Loss: {trainer.state.output:.3f}"
            )

        # TODO: This iterates over complete train set again, maybe accumulate as in the
        #   example in the footnote here: https://pytorch.org/ignite/quickstart.html#
        @trainer.on(Events.EPOCH_COMPLETED)
        def log_training_results(trainer):
            evaluator.run(train_loader)
            metrics = evaluator.state.metrics
            logger.info("")
            logger.info(f"Epoch {trainer.state.epoch} / {num_epochs} results: ")
            logger.info(
                f"Train: Average loss: {metrics['loss']:.3f}, "
                f"Average accuracy: {metrics['accuracy']:.3f}"
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
                    f"Val:   Average loss: {metrics['loss']:.3f}, "
                    f"Average accuracy: {metrics['accuracy']:.3f}"
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
                    f"Test:  Average loss: {metrics['loss']:.3f}, "
                    f"Average accuracy: {metrics['accuracy']:.3f}"
                )
                logger.info("")
                experiment.log_metric("test_loss", metrics["loss"])
                experiment.log_metric("test_accuracy", metrics["accuracy"])
                writer.add_scalar("test_loss", metrics["loss"], trainer.state.epoch)
                writer.add_scalar(
                    "test_accuracy", metrics["accuracy"], trainer.state.epoch
                )

        @trainer.on(Events.EPOCH_COMPLETED)
        def checkpoint_model(trainer):
            # TODO: Do not checkpoint at every step. 
            checkpoint_dir = (
                self.out_dir / "checkpoints" / f"epoch{trainer.state.epoch}"
            )
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            torch.save(self.model, checkpoint_dir / "model.pt")

        @trainer.on(Events.EPOCH_COMPLETED)
        def plot_samples(trainer):
            """Plot a few sample images and probabilites to tensorboard."""

            def write_samples_plot(name, sample_images, sample_labels):
                # TODO: This can be improved by just using the outputs already
                #   calculated in evaluator.state.output in the functions above.
                #   Problem: At least in the train evaluator, the batches are not equal,
                #   so the plotted images will differ from run to run.
                with torch.no_grad():
                    sample_output = self.model(sample_images)
                    sample_pred = torch.softmax(sample_output, dim=1)

                visualization.plot_samples(
                    writer,
                    f"{name}-samples",
                    trainer.state.epoch,
                    sample_images.numpy(),
                    sample_labels.numpy(),
                    sample_pred.numpy(),
                )

            write_samples_plot("train", train_sample_images, train_sample_labels)
            if val_data is not None:
                write_samples_plot("val", val_sample_images, val_sample_labels)
            if test_data is not None:
                write_samples_plot("test", test_sample_images, test_sample_labels)

        # Start training.
        num_epochs = 1 if dry_run else self.config.get("num_epochs", 5)
        epoch_length = 1 if dry_run else None
        logger.info(f"Training model on device {device}...")
        logger.info(
            "(show more steps by setting the config parameter 'print_every')"
        )
        logger.info("")
        trainer.run(train_loader, max_epochs=num_epochs, epoch_length=epoch_length)
        logger.info("Training finished!")

        # Save the trained model.
        torch.save(self.model, self.out_dir / "model.pt")

    def _load(self) -> None:
        """Loads the model from file."""
        self.model = torch.load(self.out_dir / "model.pt")

    def predict(self, image) -> dict:
        """Runs data through the model and returns output."""

        # Convert data format if required.
        image_format = preprocessing.recognize_image_format(image)
        if image_format == "files":
            image = preprocessing.load_image(
                image,
                to_numpy=False,
                resize=256,
                crop=224,
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
        elif image_format == "numpy":
            # TODO: This is almost the same code as in preprocessing.numpy_to_torch,
            #   maybe refactor it.

            # Rescale images to 0-255 and convert to uint8.
            # TODO: This should probably be done with the same min/max values as train
            #   set, see note in preprocessing.numpy_to_torch.
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
            transform = preprocessing.create_transform(
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
