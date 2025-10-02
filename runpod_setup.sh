# RUNPOD SETUP FILE
# ASSUMES YOU HAVE A GIT KEY SAVED IN workspace/.ssh/id_ed25519

export GIT_REPO_NAME=permutation-tuning
export NFS_DIR=/workspace
export LOCAL_SSD_DIR=/tmp

# Set env variables
export HF_HUB_CACHE=$LOCAL_SSD_DIR/_model_cache
export HF_HOME=$LOCAL_SSD_DIR/_hf

export VLLM_CONFIG_ROOT=$LOCAL_SSD_DIR/_vllm_config
export VLLM_CACHE_ROOT=$LOCAL_SSD_DIR/_vllm

export PIP_CACHE_DIR=$LOCAL_SSD_DIR/_pip_cache
export TORCH_EXTENSIONS_DIR=/tmp/_torch_extensions
export CUDA_CACHE_PATH=/tmp/_cuda_cache

export WANDB_DIR=$NFS_DIR/$GIT_REPO_NAME/wandb

export UV_CACHE_DIR=/tmp/_uv_cache
export VENV_DIR=/tmp/_uv_venv/$GIT_REPO_NAME


# Run installation of basic packages
apt-get update
apt-get install -y vim
apt-get install -y git
apt-get install -y tmux

# Ensure ssh dir and permissions
mkdir -p ~/.ssh
cp $NFS_DIR/.ssh/id_ed25519 ~/.ssh/id_ed25519
chmod 600 ~/.ssh/id_ed25519

# Optional: avoid strict host key checking
echo -e "Host *\n\tStrictHostKeyChecking no\n" > ~/.ssh/config

# Add GitHub as known host
ssh-keyscan github.com >> ~/.ssh/known_hosts

# Check if directory already exists before cloning
if [ ! -d "$NFS_DIR/$GIT_REPO_NAME" ]; then
    git clone git@github.com:ariahw/$GIT_REPO_NAME.git $NFS_DIR/$GIT_REPO_NAME
else
    echo "Directory $NFS_DIR/$GIT_REPO_NAME already exists, skipping git clone"
fi

# Load other environment variables
cd $NFS_DIR/$GIT_REPO_NAME
source setup.sh

# Unsloth will use a local file cache which is slower for runpod, change to tmp dir
if [ ! -e "$NFS_DIR/$GIT_REPO_NAME/unsloth_compiled_cache" ]; then
    ln -s $LOCAL_SSD_DIR/unsloth_compiled_cache $NFS_DIR/$GIT_REPO_NAME/unsloth_compiled_cache
fi

# Install uv
pip install uv
uv venv $VENV_DIR
source $VENV_DIR/bin/activate
uv sync --active

# Create jupyter kernel
uv run --active python -m ipykernel install --user --name sl-venv --display-name "Python (sl-venv)"

export TMUX_SESSION_NAME=workworkwork

# Create a new tmux session
tmux new -s $TMUX_SESSION_NAME
tmux attach -d -t "$TMUX_SESSION_NAME"

# NOTE: User needs to manually activate the venv in order to run uv commands using --active flag

