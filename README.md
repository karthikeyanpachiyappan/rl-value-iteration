# VALUE ITERATION ALGORITHM

## AIM
To develop a Python program to find the optimal policy for the given MDP using the value iteration algorithm.

## PROBLEM STATEMENT
The FrozenLake environment in OpenAI Gym is a gridworld problem that challenges reinforcement learning agents to navigate a slippery terrain to reach a goal state while avoiding hazards. Note that the environment is closed with a fence, so the agent cannot leave the gridworld.

## States

* *5 Terminal States:*
    * G - (Goal): The state the agent aims to reach.
    * H - (Hole): A hazardous state that the agent must avoid at all costs.

* *11 Non-Terminal States:*
    * S - (Starting state): The initial position of the agent.
    * Intermediate states: Grid cells forming a layout that the agent must traverse.

## Actions
The agent can take 4 actions in each state:

* LEFT
* RIGHT
* UP
* DOWN

## VALUE ITERATION ALGORITHM

* Value iteration is a method of computing an optimal MDP policy and its value.
* It begins with an initial guess for the value function, and iteratively updates it towards the optimal value function, according to the Bellman optimality equation.
* The algorithm is guaranteed to converge to the optimal value function, and in the process of doing so, also converges to the optimal policy.

The algorithm is as follows:

1. Initialize the value function V(s) arbitrarily for all states s.
2. Repeat until convergence:
    * Initialize aaction-value function Q(s, a) arbitrarily for all states s and actions a.
    * For all the states s and all the action a of every state:
        * Update the action-value function Q(s, a) using the Bellman equation.
        * Take the value function V(s) to be the maximum of Q(s, a) over all actions a.
        * Check if the maximum difference between Old V and new V is less than theta.
        * Where theta is a small positive number that determines the accuracy of estimation.
3. If the maximum difference between Old V and new V is greater than theta, then
    * Update the value function V with the maximum action-value from Q.
    * Go to step 2.
4. The optimal policy can be constructed by taking the argmax of the action-value function Q(s, a) over all actions a.
5. Return the optimal policy and the optimal value function.

## VALUE ITERATION FUNCTION

### Name: KARTHIKEYAN P
### Register Number: 212223230102
### Include the value iteration function
```
envdesc  = ['FSFH','HFFH','HFGF', 'FFFH']
env = gym.make('FrozenLake-v1',desc=envdesc)
init_state = env.reset()
goal_state = 10
P = env.env.P
```
```
def value_iteration(P, gamma=1.0, theta=1e-10):
    V = np.zeros(len(P), dtype=np.float64)
    while True:
      Q=np.zeros((len(P),len(P[0])),dtype=np.float64)
      for s in range(len(P)):
        for a in range(len(P[s])):
          for prob,next_state,reward,done in P[s][a]:
            Q[s][a]+=prob*(reward+gamma*V[next_state]*(not done))
      if np.max(np.abs(V-np.max(Q,axis=1)))<theta:
        break
      V=np.max(Q,axis=1)
    pi=lambda s:{s:a for s,a in enumerate(np.argmax(Q,axis=1))}[s]
    return V, pi
```
## OUTPUT:

### Mention the optimal policy:
![Screenshot 2025-03-28 140952](https://github.com/user-attachments/assets/75aa7ba2-1f33-4fd6-86f4-28cac816b071)

### Optimal value function:
![Screenshot 2025-03-28 141005](https://github.com/user-attachments/assets/ac7a000e-1519-4e05-b222-8e1fdbb72fc6)

### Success Rate of optimal policy: 
![Screenshot 2025-03-28 141001](https://github.com/user-attachments/assets/2b7224d2-df6f-416e-a76d-40eee43b20d8)


## RESULT:
Thus, a Python program is developed to find the optimal policy for the given MDP using the value iteration algorithm.

