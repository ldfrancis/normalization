


# Setting up

## Virtual Environment
```bash
python -m venv venv
source venv/bin/activate
```

## Install dependencies
install pytorch (see https://pytorch.org/get-started/locally/)
```bash
pip install datasets evaluate transformers wandb hydra-core sacrebleu
```

export PYTHONPATH=$(pwd)

## Configure wandb
```bash
wandb login
```

## HF login
```bash
hf auth login
```

## Download ImageNet initially
```bash
python -c "from nstudy.core.data import imagenet_loaders; imagenet_loaders(root='<dir>/imagenet', num_workers=0); print('ImageNet ready')"