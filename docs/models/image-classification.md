# Image classification

Models for image classification take in and image and output one label.

## Input formats

### Numpy arrays

Each data split should be a list of two elements: The first element is a numpy array 
of all images of size `(number of images, color channels, width, height)`. The second 
element is an array of labels (integers).

Example:

```python
train_images = np.zeros(32, 3, 256, 256)  # 32 images with 3 color channels and size 256x256
train_labels = np.zeros(32)
train_data = [train_images, train_labels]
```

### Pytorch datasets

Each element of the dataset should be a tuple: The first element is a torch tensor of 
the image of size `(color channels, width, height)`. The second element is the label as 
integer. All [datasets from torchvision](https://pytorch.org/docs/stable/torchvision/datasets.html) 
are compatible to this format. 


### Files

Coming soon!


## Preprocessing

TODO

## Models

TODO

<!--
### Random forest




"simple-cnn": image_classification.TorchImageClassificationWrapper,
    "resnet18": image_classification.TorchImageClassificationWrapper,
    "alexnet": image_classification.TorchImageClassificationWrapper,
    "vgg16": image_classification.TorchImageClassificationWrapper,
    "squeezenet": image_classification.TorchImageClassificationWrapper,
    "densenet": image_classification.TorchImageClassificationWrapper,
    "inception": image_classification.TorchImageClassificationWrapper,
    "googlenet": image_classification.TorchImageClassificationWrapper,
    "shufflenet": image_classification.TorchImageClassificationWrapper,
    "mobilenet": image_classification.TorchImageClassificationWrapper,
    "resnext50_32x4d": image_classification.TorchImageClassificationWrapper,
    "wide_resnet50_2": image_classification.TorchImageClassificationWrapper,
    "mnasnet": image_classification.TorchImageClassificationWrapper,


    "random-forest": image_classification.SklearnImageClassificationWrapper,
    "gradient-boosting": image_classification.SklearnImageClassificationWrapper,
    "gaussian-process": image_classification.SklearnImageClassificationWrapper,
    "logistic-regression": image_classification.SklearnImageClassificationWrapper,
    "sgd": image_classification.SklearnImageClassificationWrapper,
    "perceptron": image_classification.SklearnImageClassificationWrapper,
    "passive-aggressive": image_classification.SklearnImageClassificationWrapper,
    "gaussian-nb": image_classification.SklearnImageClassificationWrapper,
    "k-neighbors": image_classification.SklearnImageClassificationWrapper,
    "mlp": image_classification.SklearnImageClassificationWrapper,
    "svc": image_classification.SklearnImageClassificationWrapper,
    "linear-svc": image_classification.SklearnImageClassificationWrapper,
    "decision-tree": image_classification.SklearnImageClassificationWrapper,
    "extra-tree": image_classification.SklearnImageClassificationWrapper,
-->