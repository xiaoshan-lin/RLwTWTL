# RLwTWTL (Reinforcement Learning with Time-Window-Temporal-Logic)

**RLwTWTL** is developed for learning a policy subjecting to (non-relaxed) time-window temporal logic specifications via reinforcement learning.

**Authors:** Xiaoshan Lin, Abbasali Koochakzadeh, Levi Vande Kamp

**News:** 
  - **Jun. 3, 2023**: Our paper **Reinforcement Learning Under Probabilistic Spatio-Temporal Constraints with Time Windows** is under review for IROS 2023.

## Table of contents

- [Quick start](#quick-start)
- [Usage](#usage)
- [Acknowledgments](#acknowledgments)

## Quick start

1. Some prerequisites: python3.10, git, graphviz

2. Clone this repository 
   ```bash
   git@github.com:xiaoshan-lin/RLwTWTL.git
   cd RLwTWTL
   ```
3. Setup virtual environment 
   ```bash
   python3.10 -m venv venv
   source venv/bin/activate
   pip install --upgrade pip
   # Install the dependencies
   pip install -r requirements.txt
   ```
4. Run the codes
   ```bash
   python src/main.py # Run Q-learning to learn a policy 
   python plot_route.py # Run an animation
   ```
5. To deactivate the virtual environment
   ```bash
   deactivate
   ```
## Usage
Running `main.py` in the `src` directory will use Q-learning to find the optimal policy for a simple pick-up and delivery problem with static rewards. The code will print out some time performance information and, at the end, the results of testing the found optimal policy. Running this code will also make some files in the `output` directory. You can look at `mdp_trajectory_log.txt` with an ANSI escape code compatible text viewer (I use `less -R`) for some colors that represent what is happening during learning. You can find at the top of `main.py` what each color represents. 

Next you can run `plot_route.py` which will show a nice animation of a route generated during policy test. The animation is not connected to this config file yet, so you will have to edit that file if you want to animate a different problem.

You can edit the environment, TWTL, STL, rewards, and learning parameters in `config/default.yaml`. All options are explained in that file.

## Acknowledgments

- PyTWTL is used to generate a DFA from a TWTL specification. You can learn more about PyTWTL at the following web page: https://sites.bu.edu/hyness/twtl/. Please cite their paper if you use this package.
- Special thanks to the following individuals for their valuable contributions to this project: Levi Vande Kamp, Abbasali Koochakzadeh



