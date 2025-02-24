# # client.py
# import flwr as fl
# import tensorflow as tf
# from tensorflow import keras
# import numpy as np

# class FlowerClient(fl.client.NumPyClient):
#     def __init__(self, model, x_train, y_train, x_val, y_val):
#         self.model = model
#         self.x_train = x_train
#         self.y_train = y_train
#         self.x_val = x_val
#         self.y_val = y_val

#     def get_parameters(self, config):
#         # Get model parameters using Keras' built-in method.
#         return self.model.get_weights()

#     def set_parameters(self, parameters):
#         # Set model parameters using Keras' built-in method.
#         self.model.set_weights(parameters)

#     def fit(self, parameters, config):
#         self.set_parameters(parameters)
        
#         # Get hyperparameters for this round
#         batch_size = config["batch_size"]
#         epochs = config["local_epochs"]

#         # Train the model
#         history = self.model.fit(
#             self.x_train,
#             self.y_train,
#             batch_size=batch_size,
#             epochs=epochs,
#             validation_data=(self.x_val, self.y_val),
#         )

#         # Return updated model parameters and training results
#         parameters_updated = self.get_parameters(config={})
#         results = {
#             "loss": history.history["loss"][-1],
#             "accuracy": history.history["accuracy"][-1],
#             "val_loss": history.history["val_loss"][-1],
#             "val_accuracy": history.history["val_accuracy"][-1],
#         }
        
#         return parameters_updated, len(self.x_train), results

#     def evaluate(self, parameters, config):
#         self.set_parameters(parameters)
#         loss, accuracy = self.model.evaluate(self.x_val, self.y_val)
#         return loss, len(self.x_val), {"accuracy": accuracy}
# client.py
import flwr as fl
import tensorflow as tf
from tensorflow import keras
import numpy as np

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, x_train, y_train, x_val, y_val):
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val

    def get_parameters(self, config):
        return self.model.get_weights()

    def set_parameters(self, parameters):
        self.model.set_weights(parameters)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        
        # Use default values if keys are missing in config
        batch_size = config.get("batch_size", 32)       # default batch size is 32
        epochs = config.get("local_epochs", 1)            # default local epochs is 1

        # Train the model
        history = self.model.fit(
            self.x_train,
            self.y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(self.x_val, self.y_val),
        )

        # Return updated model parameters and training results
        parameters_updated = self.get_parameters(config={})
        results = {
            "loss": history.history["loss"][-1],
            "accuracy": history.history["accuracy"][-1],
            "val_loss": history.history["val_loss"][-1],
            "val_accuracy": history.history["val_accuracy"][-1],
        }
        
        return parameters_updated, len(self.x_train), results

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = self.model.evaluate(self.x_val, self.y_val)
        return loss, len(self.x_val), {"accuracy": accuracy}
