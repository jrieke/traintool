# Accessing raw models

traintool is built on top of powerful machine learning libraries like scikit-learn or p
ytorch. After training, it gives you full access to the raw models with: 

```python
model.raw()
```

This returns a dict of all underlying model objects. It usually contains the model 
itself (`model.raw()["model"]`) but might also contain some other 
objects, e.g. data scalers (`model.raw()["scaler"]`).
