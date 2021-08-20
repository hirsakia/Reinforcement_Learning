import numpy as np

def policy_evaluation(pi, MDP, gamma=1.0, epsilon=1e-10):
    prev_V = np.zeros(len(MDP), dtype=np.float64)
    while True:
        V = np.zeros(len(MDP), dtype=np.float64)
        for s in range(len(MDP)):
            for prob, next_state, reward, done in MDP[s][pi(s)]:
                V[s] += prob * (reward + gamma * prev_V[next_state] * (not done))
        if np.max(np.abs(prev_V - V)) < epsilon:
            break
        prev_V = V.copy()
    print(V)
    return V