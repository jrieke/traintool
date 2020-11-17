# Image classification

![](https://devopedia.org/images/article/172/7316.1561043304.png)

Image classification models classify an image into one out of several categories or 
classes, based on the image content (e.g. "cat" or "dog"). 

## Input formats

### Numpy arrays

Each data set should be a list of two elements: The first element is a numpy array 
of all images of shape `(number of images, color channels (1 or 3), height, width)`. The second 
element is an array of labels (as integer indices).

Example:

```python
train_images = np.zeros(32, 3, 256, 256)  # 32 images with 3 color channels and size 256x256
train_labels = np.zeros(32, dtype=int)

traintool.train(..., train_data=[train_images, train_labels])
```

<!--
### Pytorch datasets

Each element of the dataset should be a tuple: The first element is a torch tensor of 
the image of size `(color channels, width, height)`. The second element is the label as 
integer. All [datasets from torchvision](https://pytorch.org/docs/stable/torchvision/datasets.html) 
are compatible to this format. 
-->

### Files

Image files should be arranged in one folder per class, similar to this:

```
train
+-- dogs
|   +-- funny-dog.jpg
|   +-- another-dog.png
+-- cats
|   +-- brown-cat.png
|   +-- black-cat.png
...
```

Then simply pass the directory path to the `train` function:

```python
traintool.train(..., train_data="./train")
```


## Scikit-learn models

These models implement simple classification algorithms that should train in a 
reasonable amount of time. Note that they are not GPU-accelerated so they might still
take quite long with large datasets. 

**Preprocessing:** Image files are first loaded to a size of 28 x 28. All images (numpy 
or files) are then flattened and scaled to mean 0, standard deviation 1 (based on the 
train set). 

**Config parameters:**

- `num_samples`: Set the number of samples to train on. This can be used to train on a 
subset of the data. Defaults to None (i.e. train on all data).
- `num_samples_to_plot`: Set the number of samples to plot to tensorboard for each 
dataset. Defaults to 5.
- All other config parameters are forwarded to the constructor of the sklearn object

**Models:**

- `random-forest`: A random forest classifier, from [sklearn.ensemble.RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
- `gradient-boosting`: Gradient boosting for classification, from [sklearn.ensemble.GradientBoostingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html)
- `gaussian-process`: Gaussian process classification based on Laplace approximation, from [sklearn.gaussian_process.GaussianProcessClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessClassifier.html#sklearn.gaussian_process.GaussianProcessClassifier)
- `logistic-regression`: Logistic Regression (aka logit, MaxEnt) classifier, from [sklearn.linear_model.LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- `sgd`: Linear classifiers (SVM, logistic regression, etc.) with SGD training, from [sklearn.linear_model.SGDClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html)
- `perceptron`: A perceptron classifier, from [sklearn.linear_model.Perceptron](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Perceptron.html)
- `passive-aggressive`: Passive aggressive classifier, from [sklearn.linear_model.PassiveAggressiveClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PassiveAggressiveClassifier.html)
- `gaussian-nb`: Gaussian Naive Bayes, from [sklearn.naive_bayes.GaussianNB](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html)
- `k-neighbors`: Classifier implementing the k-nearest neighbors vote, from [sklearn.neighbors.KNeighborsClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)
- `mlp`: Multi-layer Perceptron classifier, from [sklearn.neural_network.MLPClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html)
- `svc`: C-Support Vector Classification, from [sklearn.svm.SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
- `linear-svc`: Linear Support Vector Classification, from [sklearn.svm.LinearSVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html)
- `decision-tree`: A decision tree classifier, from [sklearn.tree.DecisionTreeClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)
- `extra-tree`: An extra-trees classifier, from [sklearn.ensemble.ExtraTreesClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html)



## PyTorch models

These models implement deep neural networks that can give better results on complex 
datasets. They are GPU-accelerated if run on a machine with a GPU. 

**Preprocessing:** All images (numpy or files) are rescaled to 256 x 256, then 
center-cropped to 224 x 224, MEAN STD

**Config parameters:**

- `num_classes`: The number of classes/different output labels (and therefore number of 
output neurons of the network). Defaults to None, in which case it will be automatically 
inferred from the data. 
- `num_samples`: Set the number of samples to train on. This can be used to train on a 
subset of the data. Defaults to None (i.e. train on all data).
- `num_samples_to_plot`: Set the number of samples to plot to tensorboard for each 
dataset. Defaults to 5.
- `pretrained`: Whether to use pretrained weights for the models (trained on ImageNet). 
Note that this requires that there are 1000 classes (the ImageNet classes). Defaults to 
False. 

**Models:**

More information on the [torchvision docs](https://pytorch.org/docs/stable/torchvision/models.html). 

- `alexnet`: AlexNet model architecture from the [“One weird trick…”](https://arxiv.org/abs/1404.5997) paper
- `vgg11`, `vgg11_bn`, `vgg13`, `vgg13_bn`, `vgg16`, `vgg16_bn`, `vgg19`, or `vgg19_bn`: VGG model variants from [“Very Deep Convolutional Networks For Large-Scale Image Recognition”](https://arxiv.org/pdf/1409.1556.pdf)
- `resnet18`, `resnet34`, `resnet50`, `resnet101`, or `resnet152`: ResNet model variants from [“Deep Residual Learning for Image Recognition”](https://arxiv.org/pdf/1512.03385.pdf)
- `squeezenet1_0`, or `squeezenet1_1`: SqueezeNet model variants from the [“SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size”](https://arxiv.org/abs/1602.07360) paper.
- `densenet121`, `densenet169`, `densenet161`, or `densenet201`: Densenet model variants from [“Densely Connected Convolutional Networks”](https://arxiv.org/pdf/1608.06993.pdf)
- `inception_v3`: Inception v3 model architecture from [“Rethinking the Inception Architecture for Computer Vision”](http://arxiv.org/abs/1512.00567)
- `googlenet`: GoogLeNet (Inception v1) model architecture from [“Going Deeper with Convolutions”](http://arxiv.org/abs/1409.4842)
- `shufflenet_v2_x0_5`, `shufflenet_v2_x1_0`, `shufflenet_v2_x1_5`, or `shufflenet_v2_x2_0`: ShuffleNetV2 variants, as described in [“ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design”](https://arxiv.org/abs/1807.11164)
- `mobilenet_v2`: MobileNetV2 architecture from [“MobileNetV2: Inverted Residuals and Linear Bottlenecks”](https://arxiv.org/abs/1801.04381)
- `resnext50_32x4d` or `resnext101_32x8d`: ResNeXt model variants from [“Aggregated Residual Transformation for Deep Neural Networks”](https://arxiv.org/pdf/1611.05431.pdf)
- `wide_resnet50_2` or `wide_resnet101_2`: Wide ResNet-50-2 model variants from [“Wide Residual Networks”](https://arxiv.org/pdf/1605.07146.pdf)
- `mnasnet0_5`, `mnasnet0_75`, `mnasnet1_0`, or `mnasnet1_3`: MNASNet variants from [“MnasNet: Platform-Aware Neural Architecture Search for Mobile”](https://arxiv.org/pdf/1807.11626.pdf)


<!--
### Random forest




- `simple-cnn`: 
- `resnet18`: 
- `alexnet`: 
- `vgg16`: 
- `squeezenet`: 
- `densenet`: 
- `inception`: 
- `googlenet`: 
- `shufflenet`: 
- `mobilenet`: 
- `resnext50_32x4d`: 
- `wide_resnet50_2`: 
- `mnasnet`: 


    `random-forest`: 
    `gradient-boosting`: 
    `gaussian-process`: 
    `logistic-regression`: 
    `sgd`: 
    `perceptron`: 
    `passive-aggressive`: 
    `gaussian-nb`: 
    `k-neighbors`: 
    `mlp`: 
    `svc`: 
    `linear-svc`: 
    `decision-tree`: 
    `extra-tree`: 
-->