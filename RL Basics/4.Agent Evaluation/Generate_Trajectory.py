import numpy as np

def generate_trajectory(pi, env, max_steps=200):
    done, trajectory = False, []
    while not done:
        state = 3
        for t in range(max_steps):
            action = pi(state) 
            choice=np.random.choice(len(env[state][action]), 1, p=[env[state][action][0][0],env[state][action][1][0]])
            next_state, reward, done = env[state][action][choice[0]][1:]
            experience = (state, action, reward, next_state, done)
            trajectory.append(experience)
            if done:
                break
            if t >= max_steps - 1:
                trajectory = []
                break
            state = next_state
    return np.array(trajectory, np.object)

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

trajectory=generate_trajectory(pi,MDP_SBW,200)
print(trajectory)