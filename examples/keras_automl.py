# https://keras.io/examples/vision/mnist_convnet/
from tensorflow import keras
import numpy as np

from llm_optimize import optimize, eval_utils

num_classes = 10
input_shape = (28, 28, 1)

# Load the data and split it between train and test sets
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
# Make sure images have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)


# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

x0 = """
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential(
    [
        keras.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(10, activation="softmax"),
    ]
)

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(x_train, y_train, batch_size=128, epochs=3, validation_split=0.1)
"""

TASK = f"""
You will be given keras modeling code as the input to optimize over an unseen test set. 

Vary functions, imports, arguments, model type, model shape, layers, etc to perform this task to the best of your abilities.

Rules:
* The script should always create a "model" variable that is a compiled keras model
* "model" should always be set to the best estimator
* Do not use models that are not builtin to keras (no pip install)
* Be sure to include relevant keras imports
* Do not try to compute or display the test score
* Do not change the model input and output tensor shapes

Hints:
* x_train.shape == {x_train.shape}
* num_classes == {num_classes} 
"""

QUESTION = """
What is the next x to try such that the test set accuracy increases and the model better generalizes? 
"""


def train_model(script):
    try:
        result = eval_utils.exec_with_timeout_unsafe(
            script, {"x_train": x_train, "y_train": y_train}, ["model"], timeout_secs=300
        )
        model = result["model"]
        train_score = model.evaluate(x_train, y_train, verbose=0)
        score = model.evaluate(x_test, y_test, verbose=0)
        return (
            score[1],
            f"Train Loss = {train_score[0]:.3f}, Train Accuracy = {train_score[1]:.3f}, Test Loss = {score[0]:.3f}, Test Accuracy = {score[1]:.3f} (optimize)",
        )
    except Exception as e:
        return (0.0, "Exception " + str(e))


if __name__ == "__main__":
    best_code = optimize.run(TASK, QUESTION, train_model, x0=x0, stop_score=1.0, max_steps=3)
    print(best_code)
