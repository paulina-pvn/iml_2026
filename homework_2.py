# %%
# %load_ext autoreload
# %autoreload 2

# %%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from iml.dungeon import Dungeon
from iml.agent import DungeonAgent
from iml.visualization import DungeonVisualization
from iml.dungeon_maker import make_treasure_room, make_underground_maze, make_underground_maze_with_meat


# %% [markdown]
# **Group members:** Emin Cesme, Valentine Scillia, Paulina Pervukhina

# %% [markdown]
# ## Part 1: Gridworld Environment

# %% [markdown]
# ### 1.1 A drunken knight in a treasure room
# ### (a)
# > Implement a stochastic policy where you select actions uniformly at random.

# %%
def my_reward_11(state, action, next_state, dungeon):
    if next_state == dungeon.end:
        return 1.0
    return 0.0

def my_action_11(state, q_value):
	# <implement your action here>
	# return ... (int)
    return int(np.random.choice(4))

def my_Q_update_11(agent, state, action, reward, next_state, dungeon):
    return None # use None for now

treasure_room = make_treasure_room()
vis_11 = DungeonVisualization(treasure_room)

drunken_knight = DungeonAgent(n_states=treasure_room.n_states, choose_action=my_action_11,reward_func=my_reward_11,update_q_values=my_Q_update_11)

# Run the agent for 1 episode & collect states, actions, rewards
states, actions, rewards = drunken_knight.run_episode(treasure_room, nstep=10, gamma=1.0)

# Show the map, Q-values, and path.
vis_11.show(states, actions, drunken_knight.q_values)

# %% [markdown]
# ### (b)
# > Observe how you move and draw a Markov decision process (MDP) diagram for all four non-wall states.
#
# *(Insert your MDP diagram photo/figure here)*

# %% [markdown]
# ### (c)
# > Compute the theoretical probability distribution over a 2-step horizon analytically (without writing code) when you use the stochastic policy you implemented.
# For step 1 : 
# P(s) = 1/2, P(A) = 1/4, P(B)= 1/4
# For step 2 :
# P(s)= 1/2*1/2+1/4*1/4+1/4 = 9/16
# P(A)= 1/2*1/4+1/4*1/2 = 1/4
# P(B)= 1/2*1/4 = 1/8
# P(e) = 1/4*1/4 = 1/16
# All probabilities sum to 1

# %% [markdown]
# ### (d)
# > If you implement Q-learning with an episode length of 1 (`nstep = 1`) under current setup, will you manage to perform better? Why or why not? Explain your reasoning.
# nstep=1 technically means that the agent will only take 1 step per episode. But it actally makes zero steps because we do `range(nstep-1)` in `run_episode` function. So the agent receives no reward and is not able to learn and improve its performance for the next episodes.

# %% [markdown]
# ### (e)
# > For a dungeon of width $N$ and height $M$, what is the minimum number of steps required to reach the goal from the starting position? Why? 
# If $N$ and $M$ are non-wall cells and the start and treasure are in opposite corners, the minimum is $(N-1)+(M-1)$ steps.
# This is because the agent needs to move $N-1$ (whole width-1) steps in one direction and $M-1$(whole height-1) steps in the other direction to reach the goal.
# If $N$ and $M$ include a outer wall border, the interior width and height are $N-2$ and $M-2$, so the opposite interior corners are $(N-3)+(M-3)$ steps apart. 
# The general formula is the manhattan distance between the start and treasure positions, which is $|x_1 - x_2| + |y_1 - y_2|$ where $(x_1, y_1)$ and $(x_2, y_2)$ are the coordinates of the start and goal positions.
# The very minimum number of steps is 1 if start ans trasure are adjacent.
# %% [markdown]
# ---
# ### 1.2 A tricky knight in an underground maze
#
# ### (a)
# > (a) Using your stochastic policy $\pi$, implement the Q-learning update with a discount factor ($\gamma = 1$) and a maximum episode length of 30 steps ($n_{\text{step}} = 30$). Run consecutive episodes without resetting the Q-table. Explain your observations on how the Q-values change relative to the goal and starting locations.

# %%
def my_reward_12(state, action, next_state, dungeon):
    if next_state == dungeon.end:
        return 1.0
    return 0.0

def my_action_12(state, q_values):
    return np.random.choice(4) 

alpha = 0.1
gamma = 1.0

def my_Q_update_12(agent, state, action, reward, next_state, dungeon):
    future_value = (
        0.0 if next_state == dungeon.end
        else np.max(agent.q_values[:, next_state])
    )

    old_value = agent.q_values[action, state]
    target = reward + gamma * future_value
    agent.q_values[action, state] = old_value + alpha * (target - old_value)


underground_maze = make_underground_maze(p_slippery=0.0)
vis_12 = DungeonVisualization(underground_maze)

tricky_knight = DungeonAgent(n_states=underground_maze.n_states, choose_action=my_action_12, reward_func=my_reward_12, update_q_values=my_Q_update_12)

training_reward = []
# program your knight here
# compute episodic reward
# append "training_reward"
q_snapshots = {}
np.random.seed(12)  # For reproducibility

for episode in range(200):
    states, actions, rewards = tricky_knight.run_episode(underground_maze, nstep=30, gamma=1.0)
    training_reward.append(sum(rewards))
    if episode in (0,49,199):
        q_snapshots[episode+1] = tricky_knight.q_values.copy()

# plot "training_reward"
vis_12.plot_training_reward(training_reward)


# %% [markdown]
# ### (b)
# > By varying the discount factor $\gamma$, do you observe any differences? Summarize how the discount rate affects the results and explain why it is necessary.
# Gamma controls how much the knight cares about future rewards. (1 = cares abt future, 0 = cares abt immediate).
# 
gammas  = [1.0, 0.5, 0.1, 0.0]

for gamma in gammas: 
    np.random.seed(12)
    knight = DungeonAgent(n_states=underground_maze.n_states, choose_action=my_action_12, reward_func=my_reward_12, update_q_values=my_Q_update_12)
    training_reward = []
    for episode in range(200):
        states, actions, rewards = knight.run_episode(underground_maze, nstep=30, gamma=gamma)
        training_reward.append(sum(rewards))
    vis_12.show(states, actions, knight.q_values)

# $\gamma$ = 1.0 : future rewards are not discounted.Q-values can propagate from treasure back to start.
# $\gamma$ = 0.5 : future rewards are discounted by half each step. Q-values are lower and propagate less far back to start.
# $\gamma$ = 0.1 : future rewards are heavily discounted. Q-values are very low and only propagate a few steps back.
# $\gamma$ = 0.0 : future rewards are ignored. Q-values are zero except for immediate rewards at the treasure. The knight only learns about immediate rewards and does not plan ahead.
# The discount factor is important, because it controles how much future rewards contribue to Q-values.

# %% [markdown]
# ### (c)
# > Implement an $\epsilon$-greedy policy for your agent by varying $\epsilon$. Plot the sum of discounted rewards per episode over 100 training runs and explain your observations.
# The parameter $\epsilon$ controls the balance between exploration and exploitation. 
# A low $\epsilon$ means the agent chooses action with higuest Q-value. It exploits current knowledge. 
# A high $\epsilon$ means the agent explores more frequently but also makes random moves. 
# An intermediate $\epsilon$ balances exploration and what he has learned.
# A $\epsilon$ -greedy policy helps agent discover useful actions while benefiting from learned Q-values.
def make_epsilon_policy(epsilon):
    def choose_action(state, q_values):
        if np.random.random() < epsilon:
            return int(np.random.choice(4))

        values = q_values[:, state]
        best_actions = np.flatnonzero(values == np.max(values))
        return int(np.random.choice(best_actions))

    return choose_action


def make_q_update(gamma):
    def update(agent, state, action, reward, next_state, dungeon):
        old_value = agent.q_values[action, state]

        if next_state == dungeon.end:
            future_value = 0.0
        else:
            future_value = np.max(agent.q_values[:, next_state])

        target = reward + gamma * future_value

        agent.q_values[action, state] = (
            old_value + alpha * (target - old_value)
        )

    return update


epsilons = [0.0, 0.2, 0.5, 0.8, 0.99, 1.0]

results_c = {}
snapshots_c = {}

for epsilon in epsilons:
    np.random.seed(12)

    knight = DungeonAgent(
        n_states=underground_maze.n_states,
        choose_action=make_epsilon_policy(epsilon),
        reward_func=my_reward_12,
        update_q_values=make_q_update(gamma)
    )

    training_reward = []

    for episode in range(100):
        states, actions, rewards = knight.run_episode(
            underground_maze, nstep=30, gamma=gamma
        )
        training_reward.append(sum(rewards))

    results_c[epsilon] = training_reward
    snapshots_c[epsilon] = (
        states.copy(), actions.copy(), knight.q_values.copy()
    )

# Plot discounted reward per episode
plt.figure(figsize=(10, 5))

for epsilon, rewards in results_c.items():
    plt.plot(rewards, label=f"epsilon={epsilon}", alpha=0.8)

plt.xlabel("Episode")
plt.ylabel("Discounted reward")
plt.title("Effect of epsilon on Q-learning")
plt.legend()
plt.show()

# Show 3 selected snapshots
for epsilon in [0.0, 0.5, 1.0]:
    states, actions, q_values = snapshots_c[epsilon]
    print(f"Epsilon = {epsilon}")
    vis_12.show(states, actions, q_values)

# Which epsilon performed best : 

# %% [markdown]
# ### (d)
# > Is a single training run sufficient to evaluate and compare an agent’s performance across different hyperparameter settings? Specifically: (i) Will an agent reach the goal on every evaluation episode? Explain why or why not. (ii) Under what specific conditions would RL training and evaluation results be strictly deterministic and reproducible? And (iii) Given these factors, how should an RL agent's performance be properly evaluated and compared, both when those deterministic conditions are met and when they are not?
# (i) No, with a stochastic policy, the agent can choose random actions and may fail to reach the treasure within the episode.
# (ii) Results are deteministic when startiting conditions, environement, transitions... are deterministic. Using fixed random seeds also makes stochastic experiments reproducible, assumning same environment. 
# (iii) For stochastic experiments, mulitple independent trials using different seeds and compare the average reward, success rate and variability. For a determinsitic experiment, repeated runs under identical conditions should produce identical outputs. 

# %% [markdown]
# ### (e)
# > Recreate the maze using `make_underground_maze_with_meat()`, update the reward function, set $\epsilon = 0.9$, $n_{\text{step}} = 50$, and $\gamma = 0.95$, and run the agent for 100 to 200 training runs. Explain your observations regarding your strategy (Q-values) and behavior.
# The agent is trained for 200 episodes using $\epsilon = 0.9$, episode length of 50, and $\gamma = 0.95$. 
# The treasure gives a reward of +1 while meat gives +0.5 when agent enters or stays on the cell. Because it can be collected repeatedly, the agent can obtain a total reward by staying on the meat instead of looking for trasure.
# This is called reward hacking : the agent maximises the reward fincion without completing the intiial objective.
# Here, $\epsilon = 0.9$, so the agent explores randomly most of the time, so the path may vary between episodes.
# %%
def my_reward_12(state, action, next_state, dungeon):
    if next_state == dungeon.end:
        return 1.0
    if dungeon.tiles[next_state] == "+":
        return 0.5
    return 0.0

epsilon = 0.9
def my_action_12(state, q_values):
	# <implement your action here>
	# return ... (int)
    if np.random.random() < epsilon:
        return int(np.random.choice(4))
    else : 
        return (int(np.argmax(q_values[:, state])))

alpha = 0.1
gamma = 0.95

def my_Q_update_12(agent, state, action, reward, next_state, dungeon):
	# <implement your Q update here>
	# Hint: lecture 9, page 18 <20/29>
	# return ... (float)
    old_value = agent.q_values[action, state]
    if next_state == dungeon.end:
        future_value = 0.0
    else:
        future_value = np.max(agent.q_values[:, next_state])
    target = reward + gamma * future_value
    agent.q_values[action, state] = old_value + alpha * (target - old_value)


underground_maze = make_underground_maze_with_meat(p_slippery=0.0)
vis_12 = DungeonVisualization(underground_maze)

tricky_knight = DungeonAgent(n_states=underground_maze.n_states, choose_action=my_action_12, reward_func=my_reward_12, update_q_values=my_Q_update_12)

training_reward = []
# program your knight here
# compute episodic reward
# append "training_reward"

# plot "training_reward"
vis_12.plot_training_reward(training_reward)

# %% [markdown]
# > (f) Compare the state values $V(s)$ when you go directly to the treasure (e), collects free meat as you observed (+), or do both. Explain your results mathematically.
# We compare the 3 strategies using this return formula : 
# [V(s) = \sum_{t=0}^{T-1}\gamma^t r_t,\qquad \gamma=0.95]
# 1. Directly to treasure : the knight reaches the treasure after 5 moves.
# [ V_1=0.95^4\approx0.8145 ]
# 2. Repeatedly collecting meat : the knight reaches the meat after 3 moves and can collect +0.5 repeatedly by staying on that cell. Over 49 moves : 
# [V_2=0.5\sum_{t=2}^{48}0.95^t\approx8.2151 ]
# 3. Meat and then treasure : the knight collects meat on move 3 and reaches the treasure on move 11.
# [V_3=0.5(0.95^2)+0.95^{10}\approx1.0500]
# Therefore = 
# [\boxed{V_2>V_3>V_1}]
# Repeated meat collection gives the highest discounted return. This shows that the agent can maximize its reward without reaching the treasure, which is an example of reward hacking.

# %% [markdown]
# > (g) How might you modify the reward function or environment design to prevent this *reward hacking*? Propose and explain a solution. (Bonus)
# To prevent reward hacking, we could make the meat reward available only once per episode.
# After the knight collects the meat, it disappears or stops giving rewards. This prevents the agent from repeatedly earning +0.5 by staying on the same square.
# We could also add a small penalty for each move to discourage unnecessarily long paths.
# These changes would encourage the agent to reach the treasure instead of exploiting the meat reward.


# %% [markdown]
# ## Part 2: Bayesian Decision-Making in Multi-Armed Bandits

# %% [markdown]
# ### 2.1 Bayesian Estimation of Arm Rewards

# %% [markdown]
# ### (a)
# > Starting from Bayes' theorem and the Bernoulli likelihood, derive the posterior distribution $p(\theta \mid D)$ and identify its updated parameters. Hence, show that the Beta distribution is a conjugate prior to the Bernoulli likelihood. 

# %% [markdown]
# ### (b)
# > Suppose the prior is $(\beta(2,2))$. After ten pulls, the arm produces seven successes and three failures. Compute the posterior distribution and posterior mean. Show that, in general, the posterior mean can be expressed as a weighted average of the prior mean and the empirical success rate. Interpret the weights and explain how they change as the number of observations increases

# %% [markdown]
# ### 2.2 Learning with Thompson Samplin

# %% [markdown]
# ### (a)
# > Implement a five-armed Bernoulli bandit environment that generates binary rewards using the given success probabilities. Then implement a Thompson sampling agent that maintains the posterior parameters for each arm, selects actions using posterior sampling, and updates the corresponding posterior after receiving each reward. 

# %% [markdown]
# ### (b)
# > Using the results from part (a), plot the evolution of the posterior mean estimates for all five arms as a function of time; visualize the posterior distributions; discuss how the posterior means and distributions evolve; and explain why some arms become more certain about their estimated success probabilities than others.

# %% [markdown]
# ### 2.3 Experimental Evaluation and Exploration

# %% [markdown]
# ### (a)
# > Perform 100 independent trials of Thompson sampling, each consisting of 2,000 decisions. Initialize a new agent with the same priors at the beginning of every trial.

# %% [markdown]
# ### (b)
# > Derive the agent's expected reward per decision and its expected instantaneous pseudo-regret. Hence, derive the expected cumulative pseudo-regret $\mathbb{E}[\mathcal{R}_T]$. How does it grow with $T$, assuming $\epsilon$ remains fixed and the greedy choice remains optimal? (Bonus)

# %% [markdown]
# ### (c)
# > Compare $\epsilon$-greedy exploration with Thompson sampling. Explain why Thompson sampling may select an arm with a lower posterior mean and how its exploration behavior changes as more observations accumulate.

# %%
