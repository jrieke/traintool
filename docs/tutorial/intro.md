# Intro

This tutorial shows you everything that **traintool** can do. 

We will train a few different models on MNIST, use automated experiment tracking, deploy 
the models via REST APIs, and get access to the underlying, raw models. 


## Installation

If you haven't installed traintool yet, now is a good time:

```bash
pip install git+https://github.com/jrieke/traintool
```


## Dataset

We will use the MNIST dataset throughout this tutorial. Just in case you never heard of 
it: MNIST is a popular dataset for image classification. It contains images of 
handwritten digits and the task is to predict which digit is shown on a given image. 
Below are some examples.

![](https://upload.wikimedia.org/wikipedia/commons/2/27/MnistExamples.png)
