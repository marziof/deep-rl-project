import os
import pandas as pd
from src.utils.plotting import plot_learning_curve, plot_env_curves

LOAD_DIR = os.path.join("results", "data")

FILE_NAME = "Pendulum_DQN_test/Pendulum-v1_logs.csv"
FILE_PATH = os.path.join(LOAD_DIR, FILE_NAME)

SAVE_DIR = os.path.join("results", "plots")
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)


env_name = "Pendulum"
SAVE_NAME = f"{env_name}_DQN_test.png"


results_df = pd.read_csv(FILE_PATH)

plot_env_curves(results_df, env_name=env_name, save_path=os.path.join(SAVE_DIR, SAVE_NAME))