# Deployment

traintool can automatically deploy your model through a REST API (using 
[FastAPI](https://fastapi.tiangolo.com/) under the hood). Simply run:

```python
model.deploy()
```

This will start a server on 127.0.0.1 at port 8000 (modify via `host` and `port` 
arguments). The API offers a POST endpoint `/predict` which works similar to the 
predict method described above. 
