import os
import pandas as pd
from src.utils.plotting import plot_learning_curve, plot_env_curves
import argparse

#We parse the environment name to get the corresponding logs for each algorithm, then we plot the learning curves for all algorithms on the same plot for comparison. We will do this for both environments.
parser = argparse.ArgumentParser(description="Run RL Experiments")
parser.add_argument(
    "--env", 
    type=str, 
    default="Pendulum-v1", # Choose from [CartPole-v1, Pendulum-v1, InvertedDoublePendulum-v5, LunarLander-v3]
    help="Name of the environment to plot (e.g., 'Pendulum-v1' or 'CartPole-v1')"
)
args = parser.parse_args()
if args.env not in ["Pendulum-v1", "CartPole-v1", "LunarLander-v3", "InvertedDoublePendulum-v5"]:
    raise ValueError("Unsupported environment. Please choose 'Pendulum-v1', 'CartPole-v1', 'LunarLander-v3', or 'InvertedDoublePendulum-v5'.")
env_name = args.env
LOAD_DIR = os.path.join("results", "data")
LOAD_DIR2 = os.path.join("final_results", "data")

results_df = None
SAVE_NAME = ""

if env_name=="Pendulum-v1":
    SAVE_NAME = "Pendulum_comparison.png"
    FILE_NAME = "Pendulum_SAC_2/sac_Pendulum-v1_logs.csv"
    FILE_PATH = os.path.join(LOAD_DIR, FILE_NAME)
    results_df1 = pd.read_csv(FILE_PATH)

    FILE_NAME2 = "Pendulum_PPO/ppo_Pendulum-v1_logs.csv"
    FILE_PATH2 = os.path.join(LOAD_DIR, FILE_NAME2)
    results_df2 = pd.read_csv(FILE_PATH2)

    FILE_NAME3 = "Pendulum_TD3/td3_Pendulum-v1_logs.csv"
    FILE_PATH3 = os.path.join(LOAD_DIR, FILE_NAME3)
    results_df3 = pd.read_csv(FILE_PATH3)

    results_df = pd.concat([results_df1, results_df2, results_df3], ignore_index=True)


elif env_name=="CartPole-v1":
    SAVE_NAME = "CartPole_comparison.png"
    FILE_NAME = "CartPole_DDQN/ddqn_CartPole-v1_logs.csv"
    FILE_PATH = os.path.join(LOAD_DIR, FILE_NAME)
    results_df1 = pd.read_csv(FILE_PATH)

    FILE_NAME2 = "CartPole_PPO/ppo_CartPole-v1_logs.csv"
    FILE_PATH2 = os.path.join(LOAD_DIR, FILE_NAME2)
    results_df2 = pd.read_csv(FILE_PATH2)

    FILE_NAME3 = "CartPole_DQN/dqn_CartPole-v1_logs.csv"
    FILE_PATH3 = os.path.join(LOAD_DIR, FILE_NAME3)
    results_df3 = pd.read_csv(FILE_PATH3)

    results_df = pd.concat([results_df1, results_df2, results_df3], ignore_index=True)

elif env_name=="LunarLander-v3":
    SAVE_NAME = "LunarLander_comparison.png"
    FILE_NAME = "LunarLander_DQN/dqn_LunarLander-v3_logs.csv"
    FILE_PATH = os.path.join(LOAD_DIR, FILE_NAME)
    results_df1 = pd.read_csv(FILE_PATH)

    FILE_NAME2 = "LunarLander_PPO/ppo_LunarLander-v3_logs.csv"
    FILE_PATH2 = os.path.join(LOAD_DIR, FILE_NAME2)
    results_df2 = pd.read_csv(FILE_PATH2)

    FILE_NAME3 = "LunarLander_DDQN/ddqn_LunarLander-v3_logs.csv"
    FILE_PATH3 = os.path.join(LOAD_DIR, FILE_NAME3)
    results_df3 = pd.read_csv(FILE_PATH3)

    results_df = pd.concat([results_df1, results_df2, results_df3], ignore_index=True)

elif env_name=="InvertedDoublePendulum-v5":
    SAVE_NAME = "InvertedDoublePendulum_comparison.png"
    FILE_NAME = "InvertedDoublePendulum_SAC_2500/sac_InvertedDoublePendulum-v5_logs.csv"
    FILE_PATH = os.path.join(LOAD_DIR, FILE_NAME)
    results_df1 = pd.read_csv(FILE_PATH)

    FILE_NAME2 = "InvertedDoublePendulum_PPO_250ksteps_500T2N/ppo_InvertedDoublePendulum-v5_logs.csv"
    FILE_PATH2 = os.path.join(LOAD_DIR2, FILE_NAME2)
    results_df2 = pd.read_csv(FILE_PATH2)
    # FILE_NAME2 = "InvertedDoublePendulum_PPO/ppo_InvertedDoublePendulum-v5_logs.csv"
    # FILE_PATH2 = os.path.join(LOAD_DIR, FILE_NAME2)
    # results_df2 = pd.read_csv(FILE_PATH2)

    FILE_NAME3 = "InvertedDoublePendulum_TD3_2500/td3_InvertedDoublePendulum-v5_logs_2500.csv"
    FILE_PATH3 = os.path.join(LOAD_DIR, FILE_NAME3)
    results_df3 = pd.read_csv(FILE_PATH3)

    results_df = pd.concat([results_df1, results_df2, results_df3], ignore_index=True)


LOAD_DIR = os.path.join("results", "data")
SAVE_DIR = os.path.join("results", "plots")
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# env_name = "CartPole-v1"


# FILE_NAME = "CartPole_DQN/dqn_CartPole-v1_logs.csv"
# FILE_PATH = os.path.join(LOAD_DIR, FILE_NAME)
# results_df = pd.read_csv(FILE_PATH)

# SAVE_NAME = "CartPole_DQN_learning_curve.png"


plot_env_curves(results_df, env_name=env_name, metric="eval_reward", save_path=os.path.join(SAVE_DIR, SAVE_NAME), bin_size=5)