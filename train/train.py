import os
import mlfoundry
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

print(os.environ)

x, y = load_iris(as_frame=True, return_X_y=True)
x = x.rename(
    columns={
        "sepal length (cm)": "sepal_length",
        "sepal width (cm)": "sepal_width",
        "petal length (cm)": "petal_length",
        "petal width (cm)": "petal_width",
    }
)

# NOTE:- You can pass these configurations via command line
# arguments, config file, environment variables.
train_x, test_x, train_y, test_y = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)

pipe = Pipeline([("scaler", StandardScaler()), ("svc", SVC())])
pipe.fit(train_x, train_y)

run = mlfoundry.get_client().create_run(project_name="iris-classification-badal")
# NOTE:- We are saving the model using mlfoundry.
# You can use any other model registry or blob storage.
run.log_model(name="iris-model-sklearn", description="Iris Model Sklearn", model=pipe, framework="sklearn")
run.log_metrics({"accuracy": pipe.score(test_x, test_y)})
