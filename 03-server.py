import flwr as fl
import pickle
from typing import List, Tuple, Optional

# Define Flower server strategy with debug prints
class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.common.NDArrays, int]],
        failures: List[BaseException],
    ) -> Optional[fl.common.NDArrays]:
        print(f"🔍 [DEBUG] aggregate_fit called for round {server_round}")

        aggregated_weights = super().aggregate_fit(server_round, results, failures)

        if aggregated_weights is not None:
            print(f"✅ [DEBUG] Aggregated weights received for round {server_round}")

            # Save global model
            with open(f"global_model_round_{server_round}.pkl", "wb") as f:
                pickle.dump(aggregated_weights, f)
            print(f"✅ Saved global model weights for round {server_round}")
        else:
            print(f"⚠️ [WARNING] No aggregated weights received for round {server_round}")

        return aggregated_weights

# Define and start the server
def start_server():
    strategy = SaveModelStrategy(
        min_fit_clients=2,         # Minimum number of clients to train in each round
        min_available_clients=2,   # Minimum number of clients that need to be connected
    )

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=3)
    )

    print("✅ Training complete!")

# # Run the server
# if __name__ == "__main__":
#     start_server()
