import tensorflow as tf
import numpy as np

# Load the trained global model
global_model = tf.keras.models.load_model("global_model.h5")

# Print the model architecture
print(global_model.summary())  

# Generate a dummy input sample (random 28x28 grayscale image)
sample_input = np.random.rand(1, 28, 28, 1)  

# Get the model's prediction
prediction = global_model.predict(sample_input)

# Print the predicted class
print("Predicted class:", np.argmax(prediction))
