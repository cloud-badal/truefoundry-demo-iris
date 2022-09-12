import os

import mlfoundry
import pandas as pd
from fastapi import FastAPI

RUN_FQN = os.getenv("MLF_RUN_FQN")
run = mlfoundry.get_client().get_run(RUN_FQN)
model = run.get_model()

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