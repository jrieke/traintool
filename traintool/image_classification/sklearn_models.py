"""
Wrapper around scikit-learn image classification models.
"""

import sklearn.preprocessing
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import (
    LogisticRegression,
    SGDClassifier,
    Perceptron,
    PassiveAggressiveClassifier,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.utils import shuffle
import joblib
import numpy as np
from loguru import logger

from ..model_wrapper import ModelWrapper
from . import preprocessing, visualization


classifier_dict = {
    "random-forest": RandomForestClassifier,
    "gradient-boosting": GradientBoostingClassifier,
    "gaussian-process": GaussianProcessClassifier,
    "logistic-regression": LogisticRegression,
    "sgd": SGDClassifier,
    "perceptron": Perceptron,
    "passive-aggressive": PassiveAggressiveClassifier,
    "gaussian-nb": GaussianNB,
    "k-neighbors": KNeighborsClassifier,
    "mlp": MLPClassifier,
    "svc": SVC,
    "linear-svc": LinearSVC,
    "decision-tree": DecisionTreeClassifier,
    "extra-tree": ExtraTreeClassifier,
}


class SklearnImageClassificationWrapper(ModelWrapper):
    """
    This wrapper handles sklearn models for image classification.
    """

    def _create_model(self) -> None:
        """Create the model based on self.model_name and store it in self.model."""
        # TODO: If there's anything else stored in config besides the classifier params,
        #   remove it here.
        # Some models need probability=True so that we can predict the probability
        # further down.
        try:
            self.model = classifier_dict[self.model_name](
                probability=True, **self.config
            )
        except TypeError:  # no probability arg
            self.model = classifier_dict[self.model_name](**self.config)

    # def _preprocess_for_prediction(self, images: np.ndarray):
    #     """Preprocess images for use in training and prediction."""

    #     # Flatten images.
    #     images = images.reshape(len(images), -1)

    #     # Scale mean and std.
    #     images = self.scaler.transform(images)
    #     return images

    def _preprocess_for_training(self, name, data):
        """Preprocess a dataset with images and labels for use in training."""

        if data is None:  # val/test can be empty´
            logger.info(f"{name}: Not given")
            return None, None
        else:
            # Convert format.
            logger.info(f"{name}:")
            logger.info(f"    data format: {preprocessing.recognize_data_format(data)}")
            data = preprocessing.to_numpy(data, resize=28, crop=28)
            images, labels = data

            # Describe dataset.
            num_classes = len(np.unique(labels))
            logger.info(f"    samples: {len(images)}")
            logger.info(f"    image shape: {images.shape[1:]}")
            logger.info(f"    classes: {num_classes}")

            # Flatten.
            self._original_image_size = images.shape[1:]
            images = images.reshape(len(images), -1)
            logger.info(f"    flattened images (to {images.shape[1]} features)")

            # Scale mean and std.
            if name == "train":
                self.scaler = sklearn.preprocessing.StandardScaler().fit(images)
            images = self.scaler.transform(images)
            logger.info("    scaled to mean 0, std 1 (based on train set)")

            # Shuffle train set.
            if name == "train":
                images, labels = shuffle(images, labels)
                logger.info("    shuffled")

            return images, labels

    def _train(
        self,
        train_data,
        val_data,
        test_data,
        writer,
        experiment,
        dry_run: bool = False,
    ) -> None:
        """Trains the model, evaluates it on val/test data and saves it to file."""

        # Preprocess all datasets and log some stats about them.
        logger.info("Preprocessing datasets...")
        train_images, train_labels = self._preprocess_for_training("train", train_data)
        val_images, val_labels = self._preprocess_for_training("val", val_data)
        test_images, test_labels = self._preprocess_for_training("test", test_data)
        logger.info("")

        # Create the model.
        logger.info("Creating model...")
        self._create_model()
        logger.info("")

        # Train the model
        if "num_samples" in self.config:
            logger.info(
                f"Training model... (using {self.config['num_samples']} of "
                f"{len(train_images)} samples)"
            )
            self.model.fit(
                train_images[: self.config["num_samples"]],
                train_labels[: self.config["num_samples"]],
            )
        else:
            logger.info("Training model...")
            logger.info(
                "(if this takes too long, train on less data with the config "
                "parameter 'num_samples')"
            )
            self.model.fit(train_images, train_labels)

        # Evaluate and log accuracy on all datasets.
        def log_accuracy(name, images, labels):
            if images is None:
                return
            acc = self.model.score(images, labels)
            logger.info(f"{name.capitalize()} accuracy:".ljust(18) + str(acc))
            writer.add_scalar(f"{name}_accuracy", acc)
            experiment.log_metric(f"{name}_accuracy", acc)

        log_accuracy("train", train_images, train_labels)
        log_accuracy("val", val_images, val_labels)
        log_accuracy("test", test_images, test_labels)

        # Plot a few samples from each dataset to tensorboard.
        num_samples_to_plot = self.config.get("num_samples_to_plot", 5)

        def plot_samples(name, images, labels):
            if images is None:
                return
            num = min(len(images), num_samples_to_plot)
            pred = self.model.predict_proba(images[:num])
            # TODO: Save sample images before shuffling train_data, and before
            #   preprocessing.
            original_images = images[:num].reshape(num, *self._original_image_size)
            visualization.plot_samples(
                writer, name, 1, original_images, labels[:num], pred,
            )

        plot_samples("train-samples", train_images, train_labels)
        plot_samples("val-samples", val_images, val_labels)
        plot_samples("test-samples", test_images, test_labels)

        # Save model.
        self._save()

    def _save(self):
        """Saves the model and scaler to file."""
        joblib.dump(self.model, self.out_dir / "model.joblib")
        joblib.dump(self.scaler, self.out_dir / "scaler.joblib")

    def _load(self):
        """Loads the model from the out dir."""
        self.model = joblib.load(self.out_dir / "model.joblib")
        self.scaler = joblib.load(self.out_dir / "scaler.joblib")

    def predict(self, image) -> dict:
        """Runs data through the model and returns output."""
        # TODO: This deals with single image right now, maybe extend for batch.

        # Convert data format if required.
        # TODO: Maybe refactor this with the code in torch_models.predict.
        image_format = preprocessing.recognize_image_format(image)
        if image_format == "files":
            # TODO: If the network was trained with numpy images,
            #   we need to convert to the same size and potentially convert to
            #   grayscale.
            image = preprocessing.load_image(image, to_numpy=True, resize=28, crop=28)
        elif image_format == "numpy":
            pass  # no conversion
        else:
            raise RuntimeError()

        # Wrap image in batch.
        image_batch = image[None]

        # Flatten dimensions.
        image_batch = image_batch.reshape(len(image_batch), -1)

        # Scale mean and std.
        image_batch = self.scaler.transform(image_batch)

        # Run through model and calculate most likely class.
        probabilities = self.model.predict_proba(image_batch)[0]
        predicted_class = int(np.argmax(probabilities))
        return {"predicted_class": predicted_class, "probabilities": probabilities}

    def raw(self) -> dict:
        """Returns the raw model object."""
        return {"model": self.model, "scaler": self.scaler}

    # @staticmethod
    # def default_config(model_name: str):
    #     # TODO: Implement other models.
    #     if model_name == "random-forest":
    #         return {"n_estimators": 10}
    #     else:
    #         raise NotImplementedError()


# class RandomForestWrapper(SklearnImageClassificationWrapper):
#     def _create_model(self, config: dict) -> None:
#         self.model = RandomForestClassifier(**config)

#     @staticmethod
#     def default_config() -> dict:
#         return {"n_estimators": 100, "criterion": "gini"}
