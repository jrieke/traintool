"""
Base class for all model wrappers.
"""

from pathlib import Path
from fastapi import FastAPI
import uvicorn
import numpy as np
from abc import ABC, abstractmethod
from datetime import datetime
from pydantic import BaseModel


class PredictRequest(BaseModel):
    image: list


class ModelWrapper(ABC):
    """
    A basic wrapper for machine learning models.

    This wrapper should contain the model itself and any additional configuration or
    resources required to run the model/make predictions. It offers a standard interface
    to interact with models regardless of their implementation or framework, e.g. to
    save/load a model to file or make a prediction.
    """

    def __init__(self, model_name: str, config: dict, out_dir: Path) -> None:
        self.model_name = model_name
        self.config = config
        self.out_dir = out_dir

    @abstractmethod
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
        pass

    # @abstractmethod
    # def save(self, out_dir: Path):
    #     """Saves the model to file."""
    #     pass

    @abstractmethod
    def _load(self) -> None:
        """Loads the model from the out dir."""
        pass

    # TODO: Maybe check in predict, raw, and deploy that model was trained or loaded.
    @abstractmethod
    def predict(self, image) -> dict:
        """Runs data through the model and returns output."""
        pass

    @abstractmethod
    def raw(self) -> dict:
        """Returns a dict of raw model objects."""
        pass

    def _create_fastapi(self):
        """
        Create a FastAPI app that can be used to deploy the model.
        
        This needs to be a separate method from deploy so that the API can be tested 
        properly.
        """
        # TODO: Set version here via version arg. Maybe read _version.txt file or
        #   implement version in __version__.
        app = FastAPI(title="traintool")
        deploy_time = datetime.now()

        @app.get("/")
        def index():
            # TODO: Maybe return the same information as in info.yml file,
            #   or experiment ID.
            return {
                "model_name": self.model_name,
                "config": self.config,
                "deploy_time": deploy_time,
            }

        @app.post("/predict")
        def predict(predict_request: PredictRequest):
            """Endpoint to classify an image with a deployed model"""

            start_time = datetime.now()

            # Convert image to numpy array (is list of lists, i.e. what
            # array.tolist() prints out).
            # TODO: Accept image files and paths to images online.
            # TODO: Think about which other parameters to include in predict_request.
            img_arr = np.asarray(predict_request.image)

            # Run through model.
            result = self.predict(img_arr)

            # Convert numpy arrays in result dict to lists.
            for key, value in result.items():
                try:
                    result[key] = value.tolist()
                except AttributeError:
                    pass  # not an array

            result["runtime"] = str(datetime.now() - start_time)
            return result

        return app

    def deploy(self, **kwargs) -> None:
        """Deploys the model through a REST API. kwargs are forwarded to uvicorn."""
        app = self._create_fastapi()
        uvicorn.run(app, **kwargs)

    def __repr__(self):
        # TODO: Maybe print something a bit shorter.
        return (
            f"Model '{self.model_name}' with config {self.config}, "
            f"saved in {self.out_dir}"
        )

    # @staticmethod
    # @abstractmethod
    # def default_config(model_name: str) -> dict:
    #     pass


# class DummyModelWrapper(ModelWrapper):
#     def __init__(self, model_name: str) -> None:
#         self.model_name = model_name
#         self.model = None

#     def train(
#         self,
#         train_data,
#         val_data,
#         test_data,
#         config: dict,
#         out_dir: Path,
#         writer,
#         experiment,
#         dry_run: bool = False,
#     ) -> None:
#         self.model = "a cool model"
#         print("Dummy model has accuracy 100 %")
#         with (out_dir / "model.txt").open("w") as f:
#             f.write(self.model)

#     @classmethod
#     def load(cls, out_dir: Path, model_name: str) -> DummyModelWrapper:
#         model_wrapper = cls(model_name)
#         with (out_dir / "model.txt").open() as f:
#             model_wrapper.model = f.read()
#         return model_wrapper

#     def predict(self, data):
#         return {"probabilities": 0}

#     def raw(self) -> dict:
#         return {"model": self.model}

# @staticmethod
# def default_config(model_name: str) -> dict:
#     return {"dummy_param": 1}


# class ClassificationModelWrapper(BaseModelWrapper):
#     def classify(self, data, config):
#         """
#         Runs data through the model and returns the predicted class and class
#         probabilites.
#         """
#         probabilites = self.predict(data, config)
#         predicted_class = probabilities.argmax()
#         return predicted_class, probabilities
