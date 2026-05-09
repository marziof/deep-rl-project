import matplotlib.pyplot as plt
from datetime import datetime

# -----------------------
# Starter plotting function for the learning curve
# For now basic - plots mean + std of rewards per episode across seeds 
# Future: will need to handle more complex cases (multiple algorithms), maybe df instead of arrays, etc.
# -----------------------
import matplotlib.pyplot as plt
import os

def plot_learning_curve(episodes, mean_rewards, std_rewards, title="Learning Curve", save_path=None, exp_name=None):
    """
    Plot the learning curve with mean rewards and standard deviation and save to disk.
    Args:
    - episodes (array): Array of episode numbers.
    - mean_rewards (array): Array of mean rewards.
    - std_rewards (array): Array of standard deviations.
    - title (str): The title for the plot and the basis for the filename.
    """
    if exp_name is None:
        exp_name = datetime.now().strftime("%Y-%m-%d_%H-%M")
        
    plt.figure(figsize=(10, 6))
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
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    # 1. Define the save path
    save_dir = os.path.join("results", "plots", exp_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 2. Create a filename from the title (e.g., "SAC Pendulum" -> "sac_pendulum.png")
    filename = title.lower().replace(" ", "_") + ".png"
    save_path = os.path.join(save_dir, filename)

    # 3. Save and show
    plt.savefig(save_path)
    print(f"Plot saved to: {save_path}")
    plt.show()
    plt.close() # Close plot to free up memory