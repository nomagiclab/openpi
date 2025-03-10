# Setup

- SSH into the Entropy login node.
- `git clone --recurse-submodules git@github.com:nomagiclab/openpi.git` (you will need your SSH key on Entropy added on Github).
- `cd openpi`
- Install uv: `curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.bashrc`.
- Setup the venv: `uv python install 3.11 && uv venv .venv --python=3.11 && GIT_LFS_SKIP_SMUDGE=1 uv sync`.
- `cp .env.template .env` and configure the secrets in `.env`.


# Running jobs

Example:
- `cd openpi` (the repository in your entropy home directory).
- `sbatch entropy/compute_stats.sbatch` and wait for stats to be ready.
- `sbatch entropy/finetune.sbatch`.
