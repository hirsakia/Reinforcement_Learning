import numpy as np

class MDP:
    # The init method or constructor
    
    def __init__(self):

        # Instance Variable
         
        print("0: Slippery Walk (Tiles = 7, Move forward = 0.5, No Move = 0.33, Move Backward = 0.16)",\
            "1: Slippery Walk (Tiles = 3, Move forward = 0.8, Move Backward = 0.2)",\
            "2: Frozen Lake (Tiles = 16, Move forward = 0.33, Move Orthogonally = 0.33)", sep="\n")

        self.MDP_number = int(input(("Choose MDP ")))
        
    def model(self):
        
        Model=dict()

        if self.MDP_number==1:

            Model[0]={0: [(1.0, 0, 0.0, True)], 1:[(1.0, 0, 0.0, True)]}
            Model[1]={0: [(0.5, 0, 0.0, True),  (0.33, 1, 0.0, False), (0.16 ,2, 0.0, False)], 1: [(0.5, 2, 0.0, False), (0.33, 1, 0.0, False), (0.16 ,0 ,0 ,True)]}
            Model[2]={0: [(0.5, 1, 0.0, False), (0.33, 2, 0.0, False), (0.16 ,3, 0.0, False)], 1: [(0.5, 3, 0.0, False), (0.33, 2, 0.0, False), (0.16 ,1 ,0 ,False)]}
            Model[3]={0: [(0.5, 2, 0.0, False), (0.33, 3, 0.0, False), (0.16 ,4, 0.0, False)], 1: [(0.5, 4, 0.0, False), (0.33, 3, 0.0, False), (0.16 ,2 ,0 ,False)]}
            Model[4]={0: [(0.5, 3, 0.0, False), (0.33, 4, 0.0, False), (0.16 ,5, 0.0, False)], 1: [(0.5, 5, 0.0, False), (0.33, 4, 0.0, False), (0.16 ,3 ,0 ,False)]}
            Model[5]={0: [(0.5, 4, 0.0, False), (0.33, 5, 0.0, False), (0.16 ,6, 1.0, True)],  1: [(0.5, 6, 1.0, True),  (0.33, 5, 0.0, False), (0.16 ,4 ,0 ,False)]}
            Model[6]={0: [(1.0, 6, 0.0, True)], 1:[(1.0, 6, 0.0, True)]}
            print("Model is 7-tiles slippery")
        
        elif self.MDP_number==2:

            Model[0]={0: [(1.0, 0, 0.0, True)], 1:[(1.0, 0, 0.0, True)]}
            Model[1]={0: [(0.8, 0, -1.0, True),  (0.2, 0, 100.0, True)],1: [(0.8, 2, 100.0, True), (0.2, 0, -1.0, True)]}
            Model[2]={0: [(1.0, 2, 0.0, True)], 1:[(1.0, 2, 0.0, True)]}
            print("Model is 3-tiles slippery")
        
        elif self.MDP_number==3:

            # state=action(probability,state,reward,terminal_or_nonterminal)
# left:0 , right:1, up:2, down:3

            Model[0]={0: [(0.33, 0, 0.0, False), (0.33, 0, 0.0, False), (0.33 ,4, 0.0, False)], \
    1:[(0.33, 1, 0.0, False), (0.33, 4, 0.0, False), (0.33, 0, 0.0, False)], \
    2:[(0.33, 0, 0.0, False), (0.33, 0, 0.0, False), (0.33, 1, 0.0, False)], \
    3:[(0.33, 1, 0.0, False), (0.33, 4, 0.0, False), (0.33, 0, 0.0, False)]}
  
            Model[1]={0: [(0.33, 0, 0.0, False), (0.33, 1, 0.0, False), (0.33 ,5, 0.0, True)], \
    1:[(0.33, 2, 0.0, False),(0.33, 1, 0.0, False),(0.33, 5, 0.0, True)], \
    2:[(0.33, 1, 0.0, False),(0.33, 2, 0.0, False),(0.33, 0, 0.0, False)], \
    3:[(0.33, 5, 0.0, True),(0.33, 0, 0.0, False),(0.33, 2, 0.0, False)]}
  
            Model[2]={0: [(0.33, 1, 0.0, False), (0.33, 2, 0.0, False), (0.33 ,6, 0.0, False)], \
    1:[(0.33, 3, 0.0, False),(0.33, 2, 0.0, False),(0.33, 6, 0.0, False)], \
    2:[(0.33, 2, 0.0, False),(0.33, 1, 0.0, False),(0.33, 3, 0.0, False)], \
    3:[(0.33, 6, 0.0, False),(0.33, 1, 0.0, False),(0.33, 3, 0.0, False)]}
  
            Model[3]={0: [(0.33, 2, 0.0, False), (0.33, 3, 0.0, False), (0.33 ,7, 0.0, True)], \
    1:[(0.33, 3, 0.0, False),(0.33, 3, 0.0, False),(0.33, 7, 0.0, True)], \
    2:[(0.33, 3, 0.0, False),(0.33, 3, 0.0, False),(0.33, 2, 0.0, False)], \
    3:[(0.33, 7, 0.0, True),(0.33, 3, 0.0, False),(0.33, 2, 0.0, False)]}
  
            Model[4]={0: [(0.33, 0, 0.0, False), (0.33, 4, 0.0, False), (0.33 ,8, 0.0, False)], \
    1:[(0.33, 5, 0.0, True),(0.33, 0, 0.0, False),(0.33, 8, 0.0, False)], \
    2:[(0.33, 0, 0.0, False),(0.33, 5, 0.0, True),(0.33, 4, 0.0, False)], \
    3:[(0.33, 8, 0.0, False),(0.33, 5, 0.0, True),(0.33, 4, 0.0, False)]}
  
            Model[5]={0: [(1.0, 5, 0.0, True)], \
    1:[(1.0, 5, 0.0, True)], \
    2:[(1.0, 5, 0.0, True)], \
    3:[(1.0, 5, 0.0, True)]}
  
            Model[6]={0: [(0.33, 5, 0.0, True), (0.33, 2, 0.0, False), (0.33 ,10, 0.0, False)], \
    1:[(0.33, 7, 0.0, True),(0.33, 2, 0.0, False),(0.33, 10, 0.0, False)], \
    2:[(0.33, 5, 0.0, True),(0.33, 7, 0.0, True),(0.33, 2, 0.0, False)], \
    3:[(0.33, 5, 0.0, False),(0.33, 7, 0.0, True),(0.33, 10, 0.0, False)]}
  
            Model[7]={0: [(1.0, 7, 0.0, True)], \
    1:[(1.0, 7, 0.0, True)], \
    2:[(1.0, 7, 0.0, True)], \
    3:[(1.0, 7, 0.0, True)]}

            Model[8]={0: [(0.33, 8, 0.0, False), (0.33, 4, 0.0, False), (0.33 ,12, 0.0, True)], \
    1:[(0.33, 9, 0.0, False),(0.33, 4, 0.0, False),(0.33, 12, 0.0, True)], \
    2:[(0.33, 4, 0.0, False),(0.33, 9, 0.0, False),(0.33, 8, 0.0, False)], \
    3:[(0.33, 12, 0.0, True),(0.33, 9, 0.0, False),(0.33, 8, 0.0, False)]}
  
            Model[9]={0: [(0.33, 8, 0.0, False), (0.33, 5, 0.0, True), (0.33 ,13, 0.0, False)], \
    1:[(0.33, 10, 0.0, False),(0.33, 5, 0.0, True),(0.33, 13, 0.0, False)], \
    2:[(0.33, 5, 0.0, True),(0.33, 8, 0.0, False),(0.33, 10, 0.0, False)], \
    3:[(0.33, 13, 0.0, False),(0.33, 10, 0.0, False),(0.33, 8, 0.0, False)]}
  
            Model[10]={0: [(0.33, 9, 0.0, False), (0.33, 6, 0.0, False), (0.33 ,14, 0.0, False)], \
    1:[(0.33, 11, 0.0, True),(0.33, 6, 0.0, False),(0.33, 14, 0.0, False)], \
    2:[(0.33, 6, 0.0, False),(0.33, 11, 0.0, True),(0.33, 9, 0.0, False)], \
    3:[(0.33, 14, 0.0, False),(0.33, 11, 0.0, True),(0.33, 9, 0.0, False)]}

            Model[11]={0: [(1.0, 11, 0.0, True)], \
    1:[(1.0, 11, 0.0, True)], \
    2:[(1.0, 11, 0.0, True)], \
    3:[(1.0, 11, 0.0, True)]}

            Model[12]={0: [(1.0, 12, 0.0, True)], \
    1:[(1.0, 12, 0.0, True)], \
    2:[(1.0, 12, 0.0, True)], \
    3:[(1.0, 12, 0.0, True)]}

            Model[13]={0: [(0.33, 12, 0.0, True), (0.33, 9, 0.0, False), (0.33 ,13, 0.0, False)], \
    1:[(0.33, 14, 0.0, False),(0.33, 9, 0.0, False),(0.33, 13, 0.0, False)], \
    2:[(0.33, 9, 0.0, False),(0.33, 14, 0.0, False),(0.33, 12, 0.0, True)], \
    3:[(0.33, 13, 0.0, False),(0.33, 14, 0.0, False),(0.33, 12, 0.0, True)]}

            Model[14]={0: [(0.33, 13, 0.0, False), (0.33, 10, 0.0, False), (0.33 ,14, 0.0, False)], \
    1:[(0.33, 15, 1.0, True),(0.33, 10, 0.0, False),(0.33, 14, 0.0, False)], \
    2:[(0.33, 10, 0.0, False),(0.33, 15, 1.0, True),(0.33, 13, 0.0, False)], \
    3:[(0.33, 14, 0.0, False),(0.33, 15, 1.0, True),(0.33, 13, 0.0, False)]}

            Model[15]={0: [(1.0, 15, 0.0, True)], \
    1:[(1.0, 15, 0.0, True)], \
    2:[(1.0, 15, 0.0, True)], \
    3:[(1.0, 15, 0.0, True)]}
            print("model is 16-tiles Frozen lake")
        return Model
    


    def epsilon_greedy(self,env, epsilon=0.01, n_episodes=10000):
        Q = np.zeros((len(env[1])), dtype=np.float64)
        N = np.zeros(len(env[1]), dtype = np.int)
        Qe = np.empty((n_episodes, len(env[1])), dtype=np.float64)
        returns = np.empty(n_episodes, dtype=np.float64)
        actions = np.empty(n_episodes, dtype=np.int)
        name = 'Epsilon-Greedy {}'.format(epsilon)
        action_one = 0
        action_zero = 0
        for e in range(n_episodes):
            if np.random.uniform() > epsilon:
                action = np.argmax(Q)
            else:
                action = np.random.randint(len(Q))
            if action == 1:
                action_one += 1
            else:
                action_zero += 1
            choice_space=len(env[1][action])
            choice = np.random.choice(choice_space, 1, p=[env[1][action][0][0],env[1][action][1][0]])
            reward = env[1][action][choice[0]][2]
            N[action] += 1
            Q[action] = Q[action] + (reward - Q[action])/N[action]
            Qe[e] = Q
            returns[e] = reward
            actions[e] = action
        return name, returns, Qe, actions,action_one,action_zero
    


    def pure_exploitation(self, env, n_episodes=10000):
        Q = np.zeros((len(env[1])), dtype=np.float64)
        N = np.zeros(len(env[1]), dtype = np.int)
        Qe = np.empty((n_episodes, len(env[1])), dtype=np.float64)
        returns = np.empty(n_episodes, dtype=np.float64)
        actions = np.empty(n_episodes, dtype=np.int)
        name = 'Pure exploitation'
        action_one = 0
        action_zero = 0
        for e in range(n_episodes):
            action = np.argmax(Q)
            if action == 1:
                action_one += 1
            else:
                action_zero += 1
            choice_space=len(env[1][action])
            choice = np.random.choice(choice_space, 1, p=[env[1][action][0][0],env[1][action][1][0]])
            reward = env[1][action][choice[0]][2]
            N[action] += 1
            Q[action] = Q[action] + (reward - Q[action])/N[action]
            Qe[e] = Q
            returns[e] = reward
            actions[e] = action
        return name, returns, Qe, actions,action_one,action_zero
    


    def pure_exploration(self, env, n_episodes=10000):
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
            choice_space=len(env[1][action]) 
            p=[env[1][action][0][0],env[1][action][1][0]]
            choice = np.random.choice(choice_space, 1, p)
            reward = env[1][action][choice[0]][2]
            N[action] += 1
            Q[action] = Q[action] + (reward - Q[action])/N[action]
            Qe[e] = Q
            returns[e] = reward
            actions[e] = action
        return name, returns, Qe, actions,action_one,action_zero
    


    def decay_schedule(self, init_value, min_value, decay_ratio, max_steps, log_start=-2, log_base=10):
        decay_steps = int(max_steps * decay_ratio)
        rem_steps = max_steps - decay_steps
        values = np.logspace(log_start, 0, decay_steps, base=log_base, endpoint=True)[::-1]
        values = (values - values.min()) / (values.max() - values.min())
        values = (init_value - min_value) * values + min_value
        values = np.pad(values, (0, rem_steps), 'edge')
        return values
    


    def generate_trajectory(self, pi, env, max_steps=200):
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
    


    def mc_control(self, env, gamma=1.0, init_alpha=0.5, min_alpha=0.01, alpha_decay_ratio=0.5,\
        init_epsilon=1.0, min_epsilon=0.1, epsilon_decay_ratio=0.9, n_episodes=3000, max_steps=200, first_visit=True):
        nS, nA = len(env), len(env[1])
        discounts = np.logspace(0, max_steps, num=max_steps, base=gamma, endpoint=False) 
        alphas = self.decay_schedule(init_alpha, min_alpha, alpha_decay_ratio, n_episodes)
        epsilons = self.decay_schedule(init_epsilon, min_epsilon, epsilon_decay_ratio, n_episodes)
        pi_track = []
        Q = np.zeros((nS, nA), dtype=np.float64)
        Q_track = np.zeros((n_episodes, nS, nA), dtype=np.float64)
        select_action = lambda state, Q, epsilon: np.argmax(Q[state]) \
            if np.random.random() > epsilon \
            else np.random.randint(len(Q[state]))
        for e in range(n_episodes):
            trajectory = self.generate_trajectory(select_action, Q, epsilons[e], env, max_steps)
            visited = np.zeros((nS, nA), dtype=np.bool)
            for t, (state, action, reward, _, _) in enumerate(trajectory):
                if visited[state][action] and first_visit:
                    continue
                visited[state][action] = True
                n_steps = len(trajectory[t:])
                G = np.sum(discounts[:n_steps] * trajectory[t:, 2])
                Q[state][action] = Q[state][action] + alphas[e] * (G - Q[state][action])
            Q_track[e] = Q
            pi_track.append(np.argmax(Q, axis=1))
        V = np.max(Q, axis=1)
        pi = lambda s: {s:a for s, a in enumerate(np.argmax(Q, axis=1))}[s]
        return Q, V, pi, Q_track, pi_track
    


    def mc_prediction(self, pi, env, gamma=1.0, init_alpha=0.5, min_alpha=0.01,\
         alpha_decay_ratio=0.5, n_episodes=500, max_steps=200, first_visit=True):
        nS = len(env)
        discounts = np.logspace(0, max_steps, num=max_steps, base=gamma, endpoint=False)
        alphas = self.decay_schedule(init_alpha, min_alpha, alpha_decay_ratio, n_episodes)
        V = np.zeros(nS, dtype=np.float64)
        V_track = np.zeros((n_episodes, nS), dtype=np.float64)
        targets = {state:[] for state in range(nS)}
        for e in range(n_episodes):
            trajectory = self.generate_trajectory(pi, env, max_steps)
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



    def policy_improvement(V, MDP, gamma=1.0):
        Q = np.zeros((len(MDP), len(MDP[0])), dtype=np.float64)
        for s in range(len(MDP)):
            for a in range(len(MDP[s])):
                for prob, next_state, reward, done in MDP[s][a]:
                    Q[s][a] += prob * (reward + gamma * V[next_state] * (not done))
        new_pi = lambda s: {s:a for s, a in enumerate(np.argmax(Q, axis=1))}[s]
        return new_pi



    def td_lambda(self, pi, env, gamma=1.0, init_alpha=0.5, min_alpha=0.01, alpha_decay_ratio=0.5, lambda_=0.3, n_episodes=500):
        nS = len(env)
        V = np.zeros(nS, dtype=np.float64)
        E = np.zeros(nS, dtype=np.float64)
        V_track = np.zeros((n_episodes, nS), dtype=np.float64)
        alphas = self.decay_schedule(
            init_alpha, min_alpha, 
            alpha_decay_ratio, n_episodes)
        for e in range(n_episodes):
            E.fill(0)
            state, done = 3, False
            while not done:
                action = pi(state)
                choice=np.random.choice(len(env[state][action]), 1, p=[env[state][action][0][0],env[state][action][1][0]])
                next_state, reward, done = env[state][action][choice[0]][1:]
                td_target = reward + gamma * V[next_state] * (not done)
                td_error = td_target - V[state]
                E[state] = E[state] + 1
                V = V + alphas[e] * td_error * E
                E = gamma * lambda_ * E
                state = next_state
            V_track[e] = V
        return V, V_track



    def ntd(self,pi, env, gamma=1.0, init_alpha=0.5, min_alpha=0.01, alpha_decay_ratio=0.5, n_step=3, n_episodes=500):
        nS = len(env)
        V = np.zeros(nS, dtype=np.float64)
        V_track = np.zeros((n_episodes, nS), dtype=np.float64)
        discounts = np.logspace(0, n_step+1, num=n_step+1, base=gamma, endpoint=False)
        alphas = self.decay_schedule(
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



    def td(self, pi, env, gamma=1.0,init_alpha=0.5,min_alpha=0.01,alpha_decay_ratio=0.5,n_episodes=500):
        nS = len(env)
        V = np.zeros(nS, dtype=np.float64)
        V_track = np.zeros((n_episodes, nS), dtype=np.float64)
        targets = {state:[] for state in range(nS)}
        alphas = self.decay_schedule(
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

mdp=MDP()
model=mdp.model()
print(model[0])
import gym
