# Exploring Reinforcement Learning

The aim of this project is simply exploring different techniques for reinforcement learning.
To this end, I adapted an implementation of a maze from [Nima Siboni](https://github.com/nima-siboni), who has given an [amazing lecture](https://github.com/nima-siboni/RL-course-batch-25-DSR) at Data Science Retreat.

At the current state the environment is either composed of a Maze generated by the Prim algorithm, or set manually at start time using a simple GUI.
The agent then starts to learn to navigate the specific maze using q-learning either in a greedy or softmax-fashion.

![Animation for a trained agent.](./data/run_eposide_350.mp4)

## Setup

To run this code, you should consider creating a new python (3.8) environment (venv, conda, ...) and then execute:

```bash
pip install -r requirements
```

This will install all the models that you should need to run the tested parts of this repository.

## Run Training

To run the training simply call the Makefile:

```bash
make train-instruction INSTRUCTION=<path to instruction, e.g. "instructions/prim_maze.json">
```

By default this will run the training of an agent, displaying the value maps and simulations in regular intervals.