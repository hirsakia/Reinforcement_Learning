from Policy_evaluation import policy_evaluation
from Decay_Schedule import decay_schedule
import numpy as np

def ntd(pi, 
        env, 
        gamma=1.0,
        init_alpha=0.5,
        min_alpha=0.01,
        alpha_decay_ratio=0.5,
        n_step=3,
        n_episodes=500):
    nS = len(env)
    V = np.zeros(nS, dtype=np.float64)
    V_track = np.zeros((n_episodes, nS), dtype=np.float64)
    discounts = np.logspace(0, n_step+1, num=n_step+1, base=gamma, endpoint=False)
    alphas = decay_schedule(
        init_alpha, min_alpha, 
        alpha_decay_ratio, n_episodes)
    for e in range(n_episodes):
        state, done, path = 3, False, []
        while not done or path is not None:
            path = path[1:]
            while not done and len(path) < n_step:
                action = pi(state)
                choice=np.random.choice(len(env[state][action]), 1, p=[env[state][action][0][0],env[state][action][1][0]])
                next_state, reward, done = env[state][action][choice[0]][1:]
                experience = (state, reward, next_state, done)
                path.append(experience)
                state = next_state
                if done:
                    break

            n = len(path)
            est_state = path[0][0]
            rewards = np.array(path)[:,1]
            partial_return = discounts[:n] * rewards
            bs_val = discounts[-1] * V[next_state] * (not done)
            ntd_target = np.sum(np.append(partial_return, bs_val))
            ntd_error = ntd_target - V[est_state]
            V[est_state] = V[est_state] + alphas[e] * ntd_error
            if len(path) == 1 and path[0][3]:
                path = None

        V_track[e] = V
    return V, V_track

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

V, V_track=ntd(pi,MDP_SBW)
print("Estimated values are"+str(V_track))
print("Actual values are"+str(policy_evaluation(pi,MDP_SBW)))