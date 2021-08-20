from Policy_evaluation import policy_evaluation
from Generate_Trajectory import generate_trajectory
from Decay_Schedule import decay_schedule
import numpy as np

def mc_prediction(pi, env, gamma=1.0, init_alpha=0.5, min_alpha=0.01, alpha_decay_ratio=0.5,
                  n_episodes=500, max_steps=200, first_visit=True):
    nS = len(env)
    discounts = np.logspace(0, max_steps, num=max_steps, base=gamma, endpoint=False)
    
    alphas = decay_schedule(init_alpha, min_alpha, alpha_decay_ratio, n_episodes)
    V = np.zeros(nS, dtype=np.float64)
    V_track = np.zeros((n_episodes, nS), dtype=np.float64)
    targets = {state:[] for state in range(nS)}

    for e in range(n_episodes):
        trajectory = generate_trajectory(pi, env, max_steps)
        visited = np.zeros(nS, dtype=np.bool)
        for t, (state, _, reward, _, _) in enumerate(trajectory):
            if visited[state] and first_visit:
                continue
            visited[state] = True

            n_steps = len(trajectory[t:])
            G = np.sum(discounts[:n_steps] * trajectory[t:, 2])
            targets[state].append(G)
            mc_error = G - V[state]
            V[state] = V[state] + alphas[e] * mc_error
        V_track[e] = V
    return V.copy(), V_track, targets

MDP_SBW=dict()

# state=action(probability,state,reward,terminal_or_nonterminal)
# left:0 , right:1

MDP_SBW[0]={0: [(1.0, 0, 0.0, True)], 1:[(1.0, 0, 0.0, True)]}
MDP_SBW[1]={0: [(0.5, 0, 0.0, True), (0.5 ,2, 0.0, False)], 1: [(0.5, 2, 0.0, False), (0.5 ,0 ,0 ,True)]}
MDP_SBW[2]={0: [(0.5, 1, 0.0, False), (0.5 ,3, 0.0, False)], 1: [(0.5, 3, 0.0, False), (0.5 ,1 ,0 ,False)]}
MDP_SBW[3]={0: [(0.5, 2, 0.0, False), (0.5 ,4, 0.0, False)], 1: [(0.5, 4, 0.0, False), (0.5 ,2 ,0 ,False)]}
MDP_SBW[4]={0: [(0.5, 3, 0.0, False), (0.5 ,5, 0.0, False)], 1: [(0.5, 5, 0.0, False), (0.5 ,3 ,0 ,False)]}
MDP_SBW[5]={0: [(0.5, 4, 0.0, False), (0.5 ,6, 1.0, True)],  1: [(0.5, 6, 1.0, True),  (0.5 ,4 ,0 ,False)]}
MDP_SBW[6]={0: [(1.0, 6, 0.0, True)], 1:[(1.0, 6, 0.0, True)]}

pi = lambda s: {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0}[s]

V, V_track, targets=mc_prediction(pi,MDP_SBW)
print("targets are"+str(targets))
print("Estimated values are"+str(V_track))
print("Actual values are"+str(policy_evaluation(pi,MDP_SBW)))
