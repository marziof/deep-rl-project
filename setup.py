from setuptools import setup, find_packages

setup(
    name="deep-rl-project",
    version="0.1.0",
    description="Implementation of DQN, PPO, SAC and TD3 on OpenAI Gym environments",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "gymnasium>=0.29.0",
        "pyyaml>=6.0",
        "matplotlib>=3.7.0",
        "pandas>=2.0.0",
    ],
)
