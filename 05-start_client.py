# start_client.py
from client import FlowerClient
import flwr as fl
from tensorflow import keras

def start_client(client_id):
    # Load your .h5 model
    model = keras.models.load_model(f"model_{client_id}.h5")
    
    # Load your data for this client
    # Replace these with your actual data loading logic
    x_train, y_train = load_data_for_client(client_id, "train")
    x_val, y_val = load_data_for_client(client_id, "val")
    
    # Create client
    client = FlowerClient(model, x_train, y_train, x_val, y_val)
    
    # Start client
    fl.client.start_numpy_client(
        server_address="localhost:8080",
        client=client,
    )

def load_data_for_client(client_id, split):
    """
    Implement your data loading logic here.
    This should return x_data, y_data for the specific client
    """
    # Example dummy data for testing
    import numpy as np
    x_data = np.random.rand(100, 28, 28, 1)  # For example, MNIST shape
    y_data = np.random.randint(0, 10, 100)
    return x_data, y_data
