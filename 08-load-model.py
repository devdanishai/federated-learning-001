import pickle

with open("global_model_round_3.pkl", "rb") as f:
    global_model_weights = pickle.load(f)

print("✅ Global model loaded successfully!")
