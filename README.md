<p align="center">
    <img src="docs/assets/cover.png" alt="traintool">
</p>
<p align="center">
    <em>Train off-the-shelf machine learning models with one line of code</em>
</p>
<p align="center">
    <b><a href="https://traintool.jrieke.com/">Documentation</a> • <a href="https://github.com/jrieke/traintool">Github</a> • <a href="mailto:johannes.rieke@gmail.com">Contact</a></b>
</p>
<p align="center">
    <a href="https://github.com/jrieke/traintool/actions"><img src="https://github.com/jrieke/traintool/workflows/build/badge.svg" alt="build"></a>
    <a href="https://traintool.jrieke.com"><img src="https://github.com/jrieke/traintool/workflows/docs/badge.svg" alt="docs"></a>
    <a href="https://codecov.io/gh/jrieke/traintool"><img src="https://codecov.io/gh/jrieke/traintool/branch/master/graph/badge.svg?token=NVH72ZXX8Z" alt="codecov"/></a>
    <a href="https://github.com/psf/black"><img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Code style: black"></a>
</p>

---

traintool is a Python library for **applied machine learning**. It allows you to train 
off-the-shelf models with minimum code: You just give your data and say which model you 
want to train, and traintool takes care of the rest. It has **pre-implemented models** 
for most major use cases, works with different data formats and follows best practices 
for **experiment tracking** and **deployment**. 

<sup>Alpha Release: Note that traintool is in an early alpha release. The API can and will change 
without notice. If you find a bug, please file an issue on [Github](https://github.com/jrieke/traintool) 
or [write me](mailto:johannes.rieke@gmail.com).</sup>


## Installation

```bash
pip install git+https://github.com/jrieke/traintool
```

## Is traintool for you?

**YES** if you...

- need to solve standard ML tasks with standard, off-the-shelf models
- prefer 98 % accuracy with one line of code over 98.1 % with 1000 lines
- want to compare different model types (e.g. deep network vs. SVM)
- care about experiment tracking & deployment


**NO** if you...

- need to customize every aspect of your model, e.g. in basic research
- want to chase state of the art


## Features

- **Minimum coding —** traintool is designed from the ground up to require as few lines of code as possible. It offers a sleek and intuitive interface that gets you started in seconds. Training a model just takes a single line:

    ```python
    traintool.train("resnet18", train_data, test_data)
    ```

- **Pre-implemented models —** traintool offers fully implemented and tested models – from simple classifiers to deep neural networks. The alpha version supports image classification only but we will add more models soon. Here are only a few of the models you can use:

    ```python
    "svm", "random-forest", "alexnet", "resnet50", "inception_v3", ...
    ```

- **Easy, yet fully customizable —** You can customize every aspect of the model training and hyperparameters. Simply pass along a config dictionary:

    ```python
    traintool.train(..., config={"optimizer": "adam", "lr": 0.1})
    ```

- **Automatic experiment tracking —** traintool automatically calculates metrics and stores them – without requiring you to write any code. You can visualize the results with tensorboard or stream directly to [comet.ml](https://www.comet.ml/).

- **Automatic saving and checkpoints —** traintool automatically stores model checkpoints, logs, and experiment information in an intuitive directory structure. No more worrying about where you've put that one good experiment or which configuration it had. 

- **Works with multiple data formats —** traintool understands numpy arrays, pytorch datasets, or files and automatically converts them to the correct format for the model you train. 

- **Instant deployment —** You can deploy your model with one line of code to a REST API that you can query from anywhere. Just call:

    ```python
    model.deploy()
    ```

- **Built on popular ML libraries —** Under the hood, traintool uses common open-source frameworks like pytorch, tensorflow, and scikit-learn. You can always access the raw models from these frameworks if you want to do more complex analysis:

    ```python
    torch_model = model.raw()["model"]
    ```




<!--
Features & design principles:

- **pre-implemented models** for most major use cases
- automatic experiment tracking with **tensorboard or comet.ml**
- instant **deployment** through REST API
- supports multiple data formats (numpy, pytorch/tensorflow, files, ...)
- access to raw models from sklearn/pytorch/tensorflow
-->



## Example: Image classification on MNIST

```python
import mnist
import traintool

# Load MNIST data as numpy arrays (also works with torch/tensorflow datasets, files, ...)
train_data = [mnist.train_images(), mnist.train_labels()]
test_data = [mnist.test_images(), mnist.test_labels()]

# Train SVM
svm = traintool.train("svm", train_data=train_data, test_data=test_data)

# Train ResNet with custom hyperparameters & track metrics to tensorboard
config = {"lr": 0.1, "optimizer": "adam"}
resnet = traintool.train("resnet", train_data=train_data, test_data=test_data, 
                         config=config, tensorboard=True)

# Make prediction
result = resnet.predict(test_data[0][0])
print(result["predicted_class"])

# Deploy to REST API (with fastapi)
resnet.deploy()

# Get underlying pytorch model (e.g. for custom analysis)
pytorch_model = resnet.raw()["model"]
```

Interested? Have a look at the [tutorial](https://traintool.jrieke.com/tutorial/) or check 
out available [models](https://traintool.jrieke.com/models/).


## Get in touch!

You have a question on traintool, want to use it in production, or miss a feature? I'm 
happy to hear from you! Write me at [johannes.rieke@gmail.com](mailto:johannes.rieke@gmail.com). 
