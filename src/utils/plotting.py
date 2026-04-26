import matplotlib.pyplot as plt

# -----------------------
# Starter plotting function for the learning curve
# For now basic - plots mean + std of rewards per episode across seeds 
# Future: will need to handle more complex cases (multiple algorithms), maybe df instead of arrays, etc.
# -----------------------
def plot_learning_curve(episodes, mean_rewards, std_rewards):
    """
    Plot the learning curve with mean rewards and standard deviation.
    Args:
    - episodes (array): Array of episode numbers.
    - mean_rewards (array): Array of mean rewards.
    - std_rewards (array): Array of standard deviations.
    """
    plt.plot(episodes, mean_rewards, label="Mean reward")

    plt.fill_between(
        episodes,
        mean_rewards - std_rewards,
        mean_rewards + std_rewards,
        alpha=0.3,
        label="Std deviation"
    )

    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("CartPole Learning Curve (3 seeds)")
    plt.legend()
    plt.show()