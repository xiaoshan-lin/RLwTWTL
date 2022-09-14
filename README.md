<font size="6"> ConstrainedRL </font>  

<br />

# Purpose
```
TODO: This section is taken from the previous readme. Update for current work.
```
We consider a delivery drone that is supposed to achieve pick-up and delivery tasks that arrive stochastically during a mission. Since a delivery drone is often equipped with a camera, it can also gather useful information by monitoring the environment during the pick-up and delivery task. Motivated by the multi-use of drones, we address a persistent monitoring problem where a drone’s high-level decision making is modeled as a Markov decision process (MDP) with unknown transition probabilities. The reward function is designed based on the valuable information over the environment, and the pick-up and delivery tasks are defined by bounded temporal logic specifications. We use a reinforcement learning (RL) algorithm that maximizes the expected sum of rewards while various dynamically arriving temporal logic specifications are satisfied with a desired probability in every episode during learning.

<br />

# Setup


## Ubuntu setup (Tested on 20.04 but should work on future versions)

Add the repository to apt for pre-built Python
```bash
sudo apt install build-essential -y
sudo add-apt-repository ppa:deadsnakes/ppa
# Press ENTER when prompted
sudo apt update && sudo apt upgrade -y
```

Install the latest Python 3.10
```bash
sudo apt install python3.10 libpython3.10-dev python3.10-venv python3.10-tk curl -y
curl -sSL https://bootstrap.pypa.io/get-pip.py | python3.10
```

Install git if you don't have it
```bash
sudo apt install git -y
```

Install graphviz to draw visual graph representations
```bash
sudo apt install graphviz libgraphviz-dev -y
```

<br />

## Mac OS setup (Tested on Monterey but should work on other versions)

Install homebrew https://brew.sh/
```zsh
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

Install the latest Python 3.10
```zsh
brew install python@3.10
[INSERT BREW COMMAND TO INSTALL PYTHON HEADER FILES]
# Create a symlink to a folder on your PATH
ln -s /usr/local/opt/python@3.10/bin/python3 /usr/local/bin/python3.10
python3.10 -m pip install --upgrade pip
python3.10 -m pip install virtualenv
```

See if you have git. If you don't have it, you will get a pop-up prompting you to install it.
```zsh
git --version
```

Install graphviz to draw visual graph representations
```zsh
[INSERT BREW COMMAND]
```

<br />

## GitHub SSH key configuration

If you haven't previously, you will need to create an ssh key pair and add the public key to your GitHub account. This is required for cloneing a private repository and also for authentication when pushing. Alternatively, if you have created a GitHub Personal Access Token, you may be able to use that instead. Follow the instructions on the following pages.

1. Check for existing SSH keys: https://docs.github.com/en/authentication/connecting-to-github-with-ssh/checking-for-existing-ssh-keys
2. Generating a new SSH key: https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent
3. Add the key to your GitHub account: https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account

Clone this repository. Type `yes` if asked.
```bash
git clone git@github.com:assured-autonomy/ConstrainedRL.git
cd ConstrainedRL
# Checkout the py3 branch
git checkout py3
```

<br />

## Virtual environment setup (OS independent)

Install virtualenv and create a Python virtual environment called `venv` in the `ConstrainedRL` directory
```bash
python3.10 -m venv venv
source venv/bin/activate
pip install --upgrade pip
# Install the dependencies
pip install -r requirements.txt
```

You are now in a self contained Python environment. Python, pip, and all installed packages will go into the `venv` directory.
```bash
# Run the program
cd src
python main.py
# Run an animation
python plot_route.py
# Deactivate the environment
deactivate
```

<br />

## Usage
Running `main.py` in the `src` directory will use Q-learning to find the optimal policy for a simple pick-up and delivery problem with static rewards. The code will print out some time performance information and, at the end, the results of testing the found optimal policy. Running this code will also make some files in the `output` directory. You can look at `mdp_trajectory_log.txt` with an ANSI escape code compatible text viewer (I use `less -R`) for some colors that represent what is happening during learning. You can find at the top of `main.py` what each color represents. 

Next you can run `plot_route.py` which will show a nice animation of a route generated during policy test. The animation is not connected to this config file yet, so you will have to edit that file if you want to animate a different problem.

You can edit the environment, TWTL, STL, rewards, and learning parameters in `config/default.yaml`. All options are explained in that file.

<br />

## PyTWTL
PyTWTL is the included program used to generate a DFA from a TWTL specification. You can learn more about PyTWTL at the following web page: https://sites.bu.edu/hyness/twtl/. Please cite their paper if you use this package.