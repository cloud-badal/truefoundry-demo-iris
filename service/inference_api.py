import os

import mlfoundry
import pandas as pd
from fastapi import FastAPI

print(os.environ)
MODEL_FQN = os.getenv("MLF_MODEL_FQN")

client = mlf.get_client()
model_version = client.get_model(MODEL_FQN)
model = model_version.load()

app = FastAPI()


@app.get("/predict")
def predict(
        sepal_length: float, sepal_width: float, petal_length: float, petal_width: float
):
    data = dict(
        sepal_length=sepal_length,
        sepal_width=sepal_width,
        petal_length=petal_length,
        petal_width=petal_width,
    )
    return {"prediction": int(model.predict(pd.DataFrame([data]))[0])}
