
#----------------------------------
# Evaluation function to evaluate the performance of an agent on the environment (without exploration)
#----------------------------------
def evaluate(env, agent, n_episodes=10):
    """
    Evaluates the agent's performance on the environment by running a few episodes without exploration and averaging the rewards.
    Args:
        - env: The environment to evaluate on.
        - agent: The agent to evaluate.
        - n_episodes: The number of episodes to run for evaluation.
    Returns:
        - avg_reward: The average reward obtained across the evaluation episodes.
    """
    agent.set_eval_mode(True)

    rewards = []
    for _ in range(n_episodes):
        state, _ = env.reset()
        done = False
        total = 0

        while not done:
            action = agent.act(state)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total += reward

        rewards.append(total)

    agent.set_eval_mode(False)
    return sum(rewards) / len(rewards)