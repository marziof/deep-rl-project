import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns

# -----------------------
# Starter plotting function for the learning curve
# For now basic - plots mean + std of rewards per episode across seeds 
# Future: will need to handle more complex cases (multiple algorithms), maybe df instead of arrays, etc.
# -----------------------
import os

FONT_SIZE = 20

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




def plot_env_curves(df, env_name, metric="eval_reward", save_path=None, bin_size=100):
    """
    Plot learning curves for multiple algorithms and environments from a DataFrame with step binning.
    Args:
    - df (DataFrame): DataFrame containing columns ['algo', 'env', 'seed', 'step', 'metric', 'value'].
    - env_name (str): Name of the environment for which to plot curves.
    - save_path (str): Optional path to save the plot. If None, saves to default location.
    - bin_size (int): Size of step bins for aggregation (default 1000).
    """
    plt.figure(figsize=(12, 8))

    df = df[(df['metric'] == metric) & (df['env'] == env_name)].copy()
    # Bin steps for fair comparison across seeds
    df['step_bin'] = (df['step'] // bin_size) * bin_size
    df_binned = df.groupby(['algo', 'seed', 'step_bin'])['value'].mean().reset_index()
    df['eval_idx'] = df.groupby(['algo', 'seed']).cumcount()
    
    # replace eval_idx with average step at that eval
    df['aligned_step'] = (
        df.groupby(['algo', 'eval_idx'])['step']
        .transform('mean')
    )
    print(df_binned.groupby(['algo', 'step_bin']).size())
    print("Seeds: ", df_binned['seed'].unique())
    sns.lineplot(data=df, x='aligned_step', y='value', hue='algo', errorbar='sd')
    plt.title(f"{env_name}", fontsize=FONT_SIZE)
    plt.xlabel("Environment Steps", fontsize=FONT_SIZE)
    plt.ylabel("Mean Reward", fontsize=FONT_SIZE)
    plt.legend(title="Algorithm", fontsize=FONT_SIZE, title_fontsize=FONT_SIZE-1)
    plt.xticks(fontsize=FONT_SIZE-6)
    plt.yticks(fontsize=FONT_SIZE-6)
    plt.grid(True, linestyle='--', alpha=0.6)

    if save_path is not None:
        plt.savefig(save_path)
        print(f"Plot saved to: {save_path}")
    #plt.show()
    plt.close()



    