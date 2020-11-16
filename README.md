<p align="center">
    <img src="docs/assets/cover.png" alt="traintool">
</p>

<!--
<p align="center">
    <a href="example.com" style="color: white; padding: 15px; border-radius: 10px; margin-right: 10px; box-shadow: 2px 2px 5px 0px rgba(150,150,150,1); background: rgb(120,88,188); background: linear-gradient(327deg, rgba(120,88,188,1) 0%, rgba(72,146,236,1) 100%);">Try it out</a>
    <a href="example.com" style="color: white; background-color: #7858BC; padding: 15px; border-radius: 10px; box-shadow: 2px 2px 5px 0px rgba(150,150,150,1);">Documentation</a>
</p>
<br>
-->

<p align="center">
    <em>Train off-the-shelf machine learning models in one line of code</em>
</p>

<p align="center">
    <a href="https://pypi.org/project/traintool/"><img src="https://img.shields.io/badge/Python-3.6%2B-blue"></a>
    <a href="https://github.com/jrieke/traintool/actions"><img src="https://github.com/jrieke/traintool/workflows/tests/badge.svg" alt="tests"></a>
    <a href="https://codecov.io/gh/jrieke/traintool"><img src="https://codecov.io/gh/jrieke/traintool/branch/master/graph/badge.svg?token=NVH72ZXX8Z" alt="codecov"/></a>
    <a href="https://github.com/psf/black"><img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Code style: black"></a>
</p>

<p align="center">
    <b><a href="https://traintool.jrieke.com/">Try it out in Google Colab</a> • <a href="https://traintool.jrieke.com/">Documentation</a></b>
</p>

---

traintool is a Python library for **applied machine learning** (ML). It allows you to train 
off-the-shelf ML models with minimum code: Just give your data and 
the model name, and traintool takes care of the rest. It has **pre-implemented models** 
for major use cases, works with different data formats and integrates 
automatic **visualizations**, **experiment tracking**, and **deployment**. 


<sup>Alpha Release: traintool is in an early alpha release. The API can and will change 
without notice. If you find a bug, please file an issue on 
[Github](https://github.com/jrieke/traintool) or 
[write me](mailto:johannes.rieke@gmail.com).</sup>



<!-- <br>
<p align="center">
    <b><a href="https://colab.research.google.com/github/jrieke/traintool/blob/master/docs/tutorial/quickstart.ipynb" style="padding: 10px; margin-right: 10px; color: white; background-color: #4892EC; border: 2px solid #4892EC; border-radius: 10px;">Try it out in Google Colab</a></b>
    <b><a href="https://colab.research.google.com/github/jrieke/traintool/blob/master/docs/tutorial/quickstart.ipynb" style="padding: 10px; border: 2px solid #4892EC; border-radius: 10px;">View Docs</a></b>
</p> -->

<!--

## Is traintool for you?

**YES** if you...

- need to solve standard ML tasks with standard, off-the-shelf models
- prefer 98 % accuracy with one line of code over 98.1 % with 1000 lines
- want to compare different model types (e.g. deep network vs. SVM)
- care about experiment tracking & deployment


**NO** if you...

- need to customize every aspect of your model, e.g. in basic research
- want to chase state of the art

-->


## Features

- **Minimum coding —** traintool is designed to require as few lines of code as 
possible. It offers a sleek and intuitive interface that gets you started in seconds. 
Training a model just takes a single line:

        traintool.train("resnet18", train_data, test_data, config={"optimizer": "adam", "lr": 0.1})


- **Pre-implemented models —** The heart of traintool are fully implemented and tested 
models – from simple classifiers to deep neural networks; built on sklearn, pytorch, 
or tensorflow. Here are only a few of the models you can use:

        "svc", "random-forest", "alexnet", "resnet50", "inception_v3", ...

- **Automatic visualizations & experiment tracking —** traintool automatically 
calculates metrics, creates beautiful visualizations (in 
[tensorboard](https://www.tensorflow.org/tensorboard) or 
[comet.ml](https://www.comet.ml/)), and stores experiment data and 
model checkpoints – without needing a single additional line of code. 

- **Ready for your data —** traintool understands numpy arrays, pytorch datasets, 
and files. It automatically converts and preprocesses everything based on the model you 
use.

- **Instant deployment —** In one line of code, you can deploy your model to a REST 
API that you can query from anywhere. Just call:

        model.deploy()


<!--
Features & design principles:

- **pre-implemented models** for most major use cases
- automatic experiment tracking with **tensorboard or comet.ml**
- instant **deployment** through REST API
- supports multiple data formats (numpy, pytorch/tensorflow, files, ...)
- access to raw models from sklearn/pytorch/tensorflow
-->



## Installation

```bash
pip install traintool
```

## Example: Image classification on MNIST

Run this example interactively in Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jrieke/traintool/blob/master/docs/tutorial/quickstart.ipynb)

```python
import mnist
import traintool

# Load MNIST data as numpy
train_data = [mnist.train_images(), mnist.train_labels()]
test_data = [mnist.test_images(), mnist.test_labels()]

# Train SVM classifier
svc = traintool.train("svc", train_data=train_data, test_data=test_data)

# Train ResNet with custom hyperparameters
resnet = traintool.train("resnet", train_data=train_data, test_data=test_data, 
                         config={"lr": 0.1, "optimizer": "adam"})

# Make prediction
result = resnet.predict(test_data[0][0])
print(result["predicted_class"])

# Deploy to REST API
resnet.deploy()

# Get underlying pytorch model (e.g. for custom analysis)
pytorch_model = resnet.raw()["model"]
```

For more information, check out the 
[complete tutorial](https://traintool.jrieke.com/tutorial/quickstart/).


## Get in touch!

You have a question on traintool, want to use it in production, or miss a feature? I'm 
happy to hear from you! Write me at [johannes.rieke@gmail.com](mailto:johannes.rieke@gmail.com). 
