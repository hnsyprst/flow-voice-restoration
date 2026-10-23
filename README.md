# flow-voice-restoration

Research repository for voice restoration via flow matching.

# Quick Start
## Prerequisites
 - Python 3.12+
 - gcloud CLI (installation guide here)
 - Access to the BandLab PoC SG project on gcloud

## Setup and Launch Training

```bash
# 1. Clone repository
git clone https://github.com/hnsyprst/flow-voice-restoration
cd flow-voice-restoration

# 2. Install uv for package and project management
curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. Setup environment
uv sync

# 4. Authenticate with the gcloud CLI
gcloud auth login

# 5. Set your gcloud project to the BandLab PoC SG project
gcloud config set project bandlab-poc-sg

# 6. Launch training with default config
uv run src/train.py experiment=first
```

## Linting and Formatting

This repository uses [Ruff](https://docs.astral.sh/ruff/) for formatting and linting. This will automatically be installed for you when running `uv sync`. Passing the formatting and linting checks is required before a PR can be merged.

To make life easier, it is strongly recommended to install [the Ruff VS Code extension](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff), which can automatically fix linting and formatting errors for you on file save. You can set this up by adding the following lines to your [`settings.json` file](https://code.visualstudio.com/docs/configure/settings#_settings-json-file):
```json
{
    "notebook.formatOnSave.enabled": true,
    "notebook.codeActionsOnSave": {
        "notebook.source.fixAll": "explicit",
        "notebook.source.organizeImports": "explicit"
    },
    "[python]": {
        "editor.formatOnSave": true,
        "editor.codeActionsOnSave": {
            "source.fixAll": "explicit",
            "source.organizeImports": "explicit"
        },
        "editor.defaultFormatter": "charliermarsh.ruff",
    }
}
```