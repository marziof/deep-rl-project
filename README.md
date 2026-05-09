# deep-rl-project
Implementation and testing of famous RL algorithms (DQN, PPO, SAC, TD3) on some OpenAI-Gym environments

# Prepare, run and evaluate an experiment 
To prepare an experiment, you must modify or create your own config file at ```configs/```. You must select the algorithm, the environment and the hyperparameters. Customize the name of the folder ```exp_name``` in which your results will be stored alongside a copy of your configuration file. If you do not specify the name of the folder, it will be saved in a folder named with the following structure ```YEAR-MONTH-DAY-HOUR-MINUTE```
To run your experiment, use the following command:
```
python experiment_runner.py --config configs/conf.yaml
```
Structure of the training pipeling:
```experiment_runner.py```runs the main logic, it reads your config file and run the experiment, plots the results and save them. The ```run_experiments```function is called from ```src/train.py```. This file contains ```run_experiment```and ```run_experiments```. These functions create the environment you specified in the config file and runs the required ```run_episode```function from the environment-specifc scripts. In the environment script, the ```run_episode```handles the loops differently depending on the algorithm you have chosen. 

The framework supports the following environments and algorithms (so far):
- CartPole-v1 with DQN
- Pendulum-v1 with SAC and TD3
