# pylint: disable=wrong-import-order

import mnist
import traintool
from torchvision import datasets, transforms

# Connect traintool to comet.ml to track metrics
# api_key = ""  # ENTER HERE
# traintool.connect_comet(project_name="mnist-cnn", api_key=api_key)

# Load data (numpy).
train_data = [mnist.train_images()[:, None] / 255, mnist.train_labels()]
test_data = [mnist.test_images()[:, None] / 255, mnist.test_labels()]
sample = (test_data[0][0], test_data[1][0])

# Load data (torch).
# transform = transforms.Compose([transforms.ToTensor()])
# train_data = datasets.MNIST("data", train=True, download=True, transform=transform)
# test_data = datasets.MNIST("data", train=False, download=True, transform=transform)
# sample = test_data[0]

# Set hyperparameters.
config = {"epochs": 1, "lr": 0.1, "batch_size": 64}  # epochs: 14
# config = {"n_estimators": 10}

# Start training a simple CNN
# This spins up an AWS instance, installs dependencies, trains the model, logs metrics
# to comet.ml, and saves the model. You can shutdown your PC while this is running.
model_wrapper = traintool.train(
    "simple-cnn",
    train_data=train_data,
    test_data=test_data,
    config=config,
    dry_run=False,
)


# Make prediction on sample from test set.
# model_wrapper.predict(img_arr=sample[0])
# print(
#     f"Classified test image as class {result['predicted_class']} "
#     f"(probability: {100*result['probabilities'][result['predicted_class']]:.1f} %), "
#     f"is actually class {sample[1]}"
# )
# print()


# Fetch model as native pytorch object (e.g. for further evaluation)
# torch_model = model_wrapper.get_model(model_format="torch")
# print("Here's the model as native pytorch object:")
# print(torch_model)
# print()
