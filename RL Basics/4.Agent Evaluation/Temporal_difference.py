from Monte_carlo import mc_prediction
from Policy_evaluation import policy_evaluation
from Decay_Schedule import decay_schedule
import numpy as np

def td(pi, env, gamma=1.0,init_alpha=0.5,min_alpha=0.01,alpha_decay_ratio=0.5,n_episodes=500):

    nS = len(env)
    V = np.zeros(nS, dtype=np.float64)
    V_track = np.zeros((n_episodes, nS), dtype=np.float64)
    targets = {state:[] for state in range(nS)}
    alphas = decay_schedule(
        init_alpha, min_alpha,
        alpha_decay_ratio, n_episodes)
    for e in range(n_episodes):
        state, done = 3, False
        while not done:
            action = pi(state)
            choice=np.random.choice(len(env[state][action]), 1, p=[env[state][action][0][0],env[state][action][1][0]])
            next_state, reward, done = env[state][action][choice[0]][1:]
            td_target = reward + gamma * V[next_state] * (not done)
            targets[state].append(td_target)
            td_error = td_target - V[state]
            V[state] = V[state] + alphas[e] * td_error
            state = next_state
        V_track[e] = V
    return V, V_track, targets

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

V_mc, V_track_mc, targets_mc=mc_prediction(pi,MDP_SBW)
V, V_track_td, targets_td=td(pi,MDP_SBW)
#print("targets are"+str(targets_mc))

actual_value=policy_evaluation(pi,MDP_SBW)
print("Estimated values are with mc "+str(V_track_mc))
print("Actual values are "+str(actual_value))
print("Estimated values are with td "+str(V_track_td))

print("mc is "+str(V_track_mc[-1][-2])+" td is "+str(V_track_td[-1][-2])+" actual is "+str(actual_value[-2]))
