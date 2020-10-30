# Training and Prediction

## Your first model

As a first example, we'll train a very simple model: A 
[Support Vector Machine](https://en.wikipedia.org/wiki/Support_vector_machine) or SVM. 
We will use the image classification dataset MNIST throughout this tutorial, so let's 
load it now (the `mnist` package was installed along with traintool):

```python
import mnist
train_data = [mnist.train_images(), mnist.train_labels()]
test_data = [mnist.test_images(), mnist.test_labels()]
```

!!! tip
    The code above loads the data as numpy arrays but traintool can also deal with 
    files and pytorch datasets (see here). More data formats will be added soon.

Training the SVM is very simple now:

```python
import traintool
svm = traintool.train("svm", train_data=train_data, test_data=test_data)
```

That's it! traintool will take care of reading and converting the data, applying some 
light preprocessing, training and saving the model, and tracking all metrics. It will 
also print out the final loss and accuracy (the test accuracy should be around XX % 
here).


## Making predictions

Of course, you can do predictions with the trained model. Let's run it on an image of 
the test set:

```python
pred = svm.predict(test_data[0][0])
print("Predicted:", pred["predicted_class"], " - Is:", test_data[1][0])
```

This should print out the predicted class and the ground truth. Note that `pred` is a 
dictionary with the predicted class (`pred["predicted_class"]`) and the probabilities 
for each class (`pred["probabilities"]`).

!!! tip
    Again, we use a numpy array for the test image here but traintool can also handle 
    pytorch tensors and files. You can even pass in a whole batch of images 
    (e.g. `test_data[0][0:2]`). 


## Using other models

Now, let's check a more advanced model. We will train a [Residual Network](https://arxiv.org/abs/1512.03385) 
(ResNet), a modern deep neural network. Usually, training this model instead of an SVM 
would require you to use an advanced framework like pytorch or tensorflow and rewrite 
most of your codebase. With traintool, it's as simple replacing the model name in the `train` method:

```python
resnet = traintool.train("resnet", train_data=train_data, test_data=test_data)
```

And this syntax stays the same for every other model that traintool supports! This makes 
it really easy to compare a bunch of different models on your dataset and see what 
performs best. 


## Custom hyperparameters

In machine learning, most models have some hyperparameters that control the training 
process (e.g. the learning rate). traintool uses sensible defaults specific to each 
model, but gives you the flexibility to fully customize everything. 

First, let's find out which hyperparameters the model supports and what their defaults 
are:

```python
print(traintool.default_hyperparameters("resnet"))
```

This should print out a dictionary of hyperparameters and defaults. Now, we want to 
change the learning rate and use a different optimizer. To do this, simply pass a 
`config` dict to the train method:

```python
config = {"lr": 0.1, "optimizer": "adam"}
better_resnet = traintool.train("resnet", config=config, train_data=train_data, test_data=test_data)
```


## Saving and loading models

There are two options to save a model to disk. Either use the `save` method after 
training like this:

```python
model = traintool.train("...")
model.save("path/to/dir")
```

Or you can specify an output directory directly during training. This makes sense for 
long-running processes, so you don't lose the whole progress in case your machine is 
interrupted:

```python
model = traintool.train("...", save="path/to/dir")
```

In both cases, loading a model works via:

```python
model = traintool.load("path/to/dir")
```


<!--
---

This tutorial should show you everything to get started with traintool. We'll train and 
use a few different models on MNIST. 


## Installation

In the terminal, type:

```bash
pip install git+https://github.com/jrieke/traintool
```

Note that traintool requires Python 3.


## Data

We will use the image classification dataset MNIST throughout this tutorial. It 
contains images of handwritten digits, that need to be classified according to the 
digit 0-9. To load it, start a Python console and enter: 

```python
import mnist
train_data = [mnist.train_images(), mnist.train_labels()]
test_data = [mnist.test_images(), mnist.test_labels()]
```

The `mnist` package should have been installed along with traintool. It loads the 
images and labels as numpy arrays. 

!!! tip
    Besides numpy arrays, traintool can also handle pytorch datasets and files. More 
    data formats will be added soon.


## Training

Train a [Support Vector Machine](https://en.wikipedia.org/wiki/Support_vector_machine):

```python
import traintool
svm = traintool.train("svm", train_data=train_data, test_data=test_data)
```

Or train a [Residual Network](https://arxiv.org/abs/1512.03385):

```python
resnet = traintool.train("resnet", train_data=train_data, test_data=test_data)
```

Or train any other model that traintool supports! It's as simple as changing the model 
name – no need to learn a new framework or change your entire code base. traintool
makes it super easy to compare different models.


## Prediction

Run an image from the test set through the model:

```python
pred = svm.predict(test_data[0][0])
print("Predicted:", pred["predicted_class"], " - Is:", test_data[1][0])
```

`pred` is a dictionary with the predicted class (`pred["predicted_class"]`) and the 
probabilities for each class (`pred["probabilities"]`)


## Hyperparameters

Every model comes with sensible defaults for the hyperparameters. You can get these 
defaults via: 

```python
print(traintool.default_hyperparameters("resnet"))
```

To change hyperparameters, pass a `config` dict to the train method:

```python
config = {"lr": 0.1, "optimizer": "adam"}
better_resnet = traintool.train("resnet", config=config, train_data=train_data, test_data=test_data)
```
>