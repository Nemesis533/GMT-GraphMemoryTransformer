# Setup & Virtual Environment

The GMT project uses a virtual environment located at:

```
/home/llhama-usr/Documents/Memory_Experiments/venv
```

## Quick Start

```bash
# Activate the virtual environment
source /home/llhama-usr/Documents/Memory_Experiments/venv/bin/activate

# Install the project (editable) with test dependencies
pip install -e "/home/llhama-usr/Documents/GMT-GraphMemoryTransformer[test]"

# Verify everything works
pytest
```

## Or via requirements.txt

```bash
source /home/llhama-usr/Documents/Memory_Experiments/venv/bin/activate
pip install -r /home/llhama-usr/Documents/GMT-GraphMemoryTransformer/requirements.txt
pip install -e /home/llhama-usr/Documents/GMT-GraphMemoryTransformer
```

## Running Tests

```bash
/home/llhama-usr/Documents/Memory_Experiments/venv/bin/python -m pytest tests/test_smoke.py -v
```

## Running Training

```bash
/home/llhama-usr/Documents/Memory_Experiments/venv/bin/python scripts/train_gmt_v7.py
```

## Creating a Fresh venv (if needed)

```bash
python3.12 -m venv /home/llhama-usr/Documents/Memory_Experiments/venv
source /home/llhama-usr/Documents/Memory_Experiments/venv/bin/activate
pip install --upgrade pip
pip install -e "/home/llhama-usr/Documents/GMT-GraphMemoryTransformer[test]"
```
