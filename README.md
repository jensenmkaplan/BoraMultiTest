# BoraMultiTest

Simple setup for graphing distributions with NumPy and Matplotlib.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python plot_distributions.py
```

`plot_distributions.py` initializes an array in `[0, 1]` with 1000 evenly spaced steps using:

```python
x = np.linspace(0.0, 1.0, 1000)
```
