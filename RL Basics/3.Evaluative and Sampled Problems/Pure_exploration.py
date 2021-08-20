import numpy as np
import tqdm


def pure_exploration(env, n_episodes=10000):
    Q = np.zeros((len(env[1])), dtype=np.float64)
    N = np.zeros(len(env[1]), dtype = np.int)
    Qe = np.empty((n_episodes, len(env[1])), dtype=np.float64)
    returns = np.empty(n_episodes, dtype=np.float64)
    actions = np.empty(n_episodes, dtype=np.int)
    name = 'Pure exploration'
    action_one = 0
    action_zero = 0
    for e in range(n_episodes):
        action = np.random.randint(len(Q))
        if action == 1:
            action_one += 1
        else:
            action_zero += 1
        #print("action is "+str(action))
        choice_space=len(env[1][action])
        choice = np.random.choice(choice_space, 1, p=[env[1][action][0][0],env[1][action][1][0]])
        reward = env[1][action][choice[0]][2]
        #print("reward is "+str(reward))
        N[action] += 1
        Q[action] = Q[action] + (reward - Q[action])/N[action]
        #print("N is "+str(N[action]))
        #print("Q is "+str(Q[action]))
        Qe[e] = Q
        returns[e] = reward
        actions[e] = action
    return name, returns, Qe, actions,action_one,action_zero