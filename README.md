# dqn
Implementing Deep Q Networks (DQN) from scratch, using pytorch. I wrote a Medium blog post describing my process,
learnings, and results [link soon to be added].

## Installation
1. I use the Poetry package manager. If you don't already have Poetry installed, see their docs for instructions
(https://python-poetry.org/docs/master/). E.g. for a macOS, it just amounts to running
`curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/install-poetry.py | python -` in terminal.
2. To install this `dqn` repository: git clone it, then navigate to the root directory and run `poetry install`.
3. To get the Atari envs working, you'll also need to follow these short instructions to download and import the Atari
ROMs: https://github.com/openai/atari-py#roms
4. Test by running unit tests! Run `pytest` in the root directory.

## Results
I tested this DQN implementation on some classic benchmarks (CartPole and FrozenLake) and some Atari games as well (Pong, Freeway). Here is a summary of the results (see my Medium blog post for full details).

<img src="img/cartpole_training_mean.png" width="200"/> <img src="img/cartpole_training_1.png" width="200"/> <img src="img/cartpole_gameplay.gif" width="200"/>
(Left) Mean of 10 training runs on CartPole. Error ribbons, indicating 1 standard error, are in red. (Middle) A representative training run, where x-axis is number of env steps, y-axis is mean episode return over 100 evaluation
episodes. (Right) Gameplay of a fully trained agent, whose goal is to move the cart so the pole stays balanced without
toppling. (Image and gif source: author)

test2: ![alt text](img/cartpole_gameplay.gif | height=200 | width=100)

test3: <img src="img/cartpole_gameplay.gif" width="200"/>
