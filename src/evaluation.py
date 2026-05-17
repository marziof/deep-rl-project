import imageio

#----------------------------------
# Evaluation function to evaluate the performance of an agent on the environment (without exploration)
#----------------------------------

def evaluate(env, agent, n_episodes=10, visualize=False, video_title = None):
    """
    Evaluates the agent's performance on the environment by running a few episodes without exploration and averaging the rewards.
    Args:
        - env: The environment to evaluate on.
        - agent: The agent to evaluate.
        - n_episodes: The number of episodes to run for evaluation.
        - visualize: Whether to render the environment, and create a video during evaluation.
    Returns:
        - avg_reward: The average reward obtained across the evaluation episodes.
    """
    agent.set_eval_mode(True)

    if visualize:
        frames = []

    rewards = []
    for _ in range(n_episodes):
        state, _ = env.reset(seed=seed)
        done = False
        total = 0

        while not done:
            if visualize:
                frame = env.render()
                frames.append(frame)

            action = agent.act(state)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total += reward

        rewards.append(total)

    if visualize:
        if video_title is not None:
            imageio.mimsave(f"media/{video_title}.mp4", frames, fps=30)
        else:
            imageio.mimsave("media/all_episodes.mp4", frames, fps=30)

    agent.set_eval_mode(False)
    return sum(rewards) / len(rewards)

def evaluate_PPO(env, agent, n_episodes=10, visualize=False, video_title = None):
    """
    Evaluates the agent's performance on the environment by running a few episodes without exploration and averaging the rewards.
    Args:
        - env: The environment to evaluate on.
        - agent: The agent to evaluate.
        - n_episodes: The number of episodes to run for evaluation.
        - visualize: Whether to render the environment, and create a video during evaluation.
    Returns:
        - avg_reward: The average reward obtained across the evaluation episodes.
    """

    if visualize:
        frames = []

    rewards = []
    for _ in range(n_episodes):
        state, _ = env.reset()
        done = False
        total = 0

        while not done:
            if visualize:
                frame = env.render()
                frames.append(frame)

            action, _ = agent.act(state)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total += reward

        rewards.append(total)

    if visualize:
        if video_title is not None:
            imageio.mimsave(f"media/{video_title}.mp4", frames, fps=30)
        else:
            imageio.mimsave("media/all_episodes.mp4", frames, fps=30)

    return sum(rewards) / len(rewards)