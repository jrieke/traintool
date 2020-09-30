# Getting access to raw models

traintool gives you full access to the raw models it uses under the hood (e.g. from 
sklearn or pytorch). Just call:

```python
sklearn_model = model.raw()["model"]
```

In some cases, the dictionary returned by `model.raw()` might also contain some other 
objects, e.g. data scalers.
