from sklearn import preprocessing
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
from . import data_utils


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

    def _preprocess_for_training(self, data, is_train: bool = False):
        """Preprocess a dataset with images and labels for use in training."""

        # Return for empty val/test data.
        if data is None:
            return None, None

        # Convert format.
        data = data_utils.to_numpy(data, resize=28, crop=28)
        images, labels = data

        # Flatten.
        images = images.reshape(len(images), -1)

        # Scale mean and std.
        # TODO: Maybe make mean and std as config parameters here.
        if is_train:
            self.scaler = preprocessing.StandardScaler().fit(images)
        images = self.scaler.transform(images)

        # Shuffle train set.
        if is_train:
            images, labels = shuffle(images, labels)

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

        # Preprocess all datasets.
        logger.info("Preprocessing datasets...")
        train_images, train_labels = self._preprocess_for_training(
            train_data, is_train=True
        )
        val_images, val_labels = self._preprocess_for_training(val_data)
        test_images, test_labels = self._preprocess_for_training(test_data)

        # TODO: Refactor this.
        logger.info(
            f"Train data: {len(train_images)} samples, {train_images.shape[1]} features"
        )
        if val_data is None:
            logger.info("Val data:   Not given")
        else:
            logger.info(
                f"Val data:   {len(val_images)} samples, {val_images.shape[1]} features"
            )
        if test_data is None:
            logger.info("Test data:  Not given")
        else:
            logger.info(
                f"Test data:  {len(test_images)} samples, {test_images.shape[1]} "
                "features"
            )
        logger.info("")

        # Create and fit model.
        logger.info("Creating model...")
        self._create_model()
        logger.info("Training model... (this may take a while)")
        self.model.fit(train_images, train_labels)
        logger.info("Training finished!")
        logger.info("")

        # Evaluate accuracy on all datasets and log to experiment.
        train_acc = self.model.score(train_images, train_labels)
        # TODO: Maybe refactor this so I don't have to call similar functions 3 times.
        # TODO: Remove tabs here, they interfere with loguru's file saving (and are
        #   ugly).
        logger.info(f"Train accuracy:\t {train_acc}")
        writer.add_scalar("train_accuracy", train_acc)
        experiment.log_metric("train_accuracy", train_acc)
        if val_data is not None:
            val_acc = self.model.score(val_images, val_labels)
            logger.info(f"Val accuracy:\t {val_acc}")
            writer.add_scalar("val_accuracy", val_acc)
            experiment.log_metric("val_accuracy", val_acc)
        if test_data is not None:
            test_acc = self.model.score(test_images, test_labels)
            logger.info(f"Test accuracy:\t {test_acc}")
            writer.add_scalar("test_accuracy", test_acc)
            experiment.log_metric("test_accuracy", test_acc)

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
        image_format = data_utils.recognize_image_format(image)
        if image_format == "files":
            # TODO: If the network was trained with numpy images,
            #   we need to convert to the same size and potentially convert to
            #   grayscale.
            image = data_utils.load_image(image, to_numpy=True, resize=28, crop=28)
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
