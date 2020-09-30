# Experiment tracking

traintool tracks common metrics automatically (e.g. accuracy on train and test set)
and has different options to store and visualize them. 


## Tensorboard

[Tensorboard](https://www.tensorflow.org/tensorboard) is a popular visualization toolkit from Google's tensorflow framework. By default, traintool automatically stores logs for tensorboard along with the model, so that you can visualize the metrics of your experiments.

To start tensorboard, run on your terminal (from the project dir):

```bash
tensorboard --logdir traintool-experiments
```

Navigate your browser to [http://localhost:6006/](http://localhost:6006/) and you should see the tensorboard window:

INSERT IMAGE HERE

On the bottom left, you can select all the different runs (same names as the directories in `traintool-experiments`), on the right side you can view the metrics. 

If you want to disable tensorboard logging for a run, use `traintool.train(..., tensorboard=False)`.



## Comet.ml

You can store these metrics in [comet.ml](https://www.comet.ml/), a popular platform 
for experiment tracking. They offer free accounts (you can sign up with your Github 
account), and free premium for students & academia. 

Once you have your account, log in to comet.ml, click on your profile in the upper 
right corner, go on settings and on "Generate API Key". Pass this API key along to the 
`train` function like this:

```python
traintool.train("resnet", train_data=train_data, test_data=test_data, 
                comet_config={"api_key": YOUR_API_KEY, "project_name": OPTIONAL_PROJECT_NAME})
```

Now you can head on over to [comet.ml](https://www.comet.ml/) and follow the metrics in 
real time!

