# src/secure_agg_poc.py
import numpy as np

def make_params(seed, size):
    rng = np.random.RandomState(seed)
    return rng.randn(size)

def client_update(base_params, seed_update, size):
    # Simulate client computing delta, then masking with pairwise masks
    rng = np.random.RandomState(seed_update)
    delta = rng.randn(size) * 0.01  # small update
    return delta

def pairwise_mask(seed_a, seed_b, size):
    # deterministic mask between two clients
    rng = np.random.RandomState(seed_a ^ seed_b)
    return rng.randn(size)

def simulate(n_clients=4, size=1000):
    # Each client computes delta and mask sum with others
    seeds = [i+1 for i in range(n_clients)]
    client_params = [make_params(s+100, size) for s in seeds]
    client_deltas = [client_update(client_params[i], seeds[i]+200, size) for i in range(n_clients)]

    # Each client builds mask = sum_{j>i} mask(i,j) - sum_{j<i} mask(j,i)
    masked_updates = []
    for i in range(n_clients):
        mask = np.zeros(size)
        for j in range(n_clients):
            if j == i: continue
            m = pairwise_mask(seeds[i], seeds[j], size)
            if i < j:
                mask += m
            else:
                mask -= m
        masked_updates.append(client_deltas[i] + mask)
    # Server sums masked_updates
    aggregate = np.sum(masked_updates, axis=0)
    # True aggregate
    true_agg = np.sum(client_deltas, axis=0)
    diff = np.linalg.norm(aggregate - true_agg)
    print("Masked aggregate - true_aggregate norm:", diff)
    assert diff < 1e-8 or diff < 1e-6
    print("Secure additive masking PoC success for", n_clients, "clients.")

if __name__ == "__main__":
    simulate(n_clients=5, size=5000)
