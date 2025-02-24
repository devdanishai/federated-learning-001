import tensorflow as tf
import numpy as np
import pickle
import glob

# Load client models
client_model_files = glob.glob("model_*.h5")  # Finds all client models saved as model_1.h5, model_2.h5, etc.

if not client_model_files:
    print("❌ No client models found!")
    exit()

print(f"📂 Found {len(client_model_files)} client models: {client_model_files}")

# Create an empty model with the same architecture as clients
global_model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(28, 28, 1)),
    tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10, activation="softmax")
])

# Load weights from client models
client_weights = []
for model_file in client_model_files:
    model = tf.keras.models.load_model(model_file)
    client_weights.append(model.get_weights())

# Aggregate weights (Federated Averaging)
num_clients = len(client_weights)
avg_weights = [np.mean([client[i] for client in client_weights], axis=0) for i in range(len(client_weights[0]))]

# Set aggregated weights to the global model
global_model.set_weights(avg_weights)

# Save the global model
global_model.save("global_model.h5")
print("✅ Global model created and saved as global_model.h5")
