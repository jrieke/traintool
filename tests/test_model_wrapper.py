from fastapi.testclient import TestClient
import numpy as np

from traintool.model_wrapper import ModelWrapper


class DummyWrapper(ModelWrapper):
    def _train(
        self,
        train_data,
        val_data,
        test_data,
        writer,
        experiment,
        dry_run: bool = False,
    ) -> None:
        self.model = "a cool model"

    def _load(self) -> None:
        self.model = "a cool model"

    def predict(self, image) -> dict:
        return {"predicted_class": 0, "probabilities": np.array([0.1, 0.9])}

    def raw(self) -> dict:
        return {"model": self.model}


def test_wrapper_init(tmp_path):
    wrapper = DummyWrapper("dummy", {}, tmp_path)  # noqa: F841


def test_create_fastapi(tmp_path):
    wrapper = DummyWrapper("dummy", {}, tmp_path)
    app = wrapper._create_fastapi()
    client = TestClient(app)

    # index endpoint
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["model_name"] == "dummy"
    assert response.json()["config"] == {}

    # predict endpoint
    # TODO: Maybe test this for a properly trained model (especially the data
    #   conversion).
    response = client.post("/predict", json={"image": [[[0, 1], [1, 0]]]})
    assert response.status_code == 200
    assert response.json()["predicted_class"] == 0
    assert np.all(np.array(response.json()["probabilities"]) == np.array([0.1, 0.9]))
