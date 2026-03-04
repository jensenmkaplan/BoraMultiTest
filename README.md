# BoraMultiTest

Simple setup for graphing probability distributions with NumPy and Matplotlib.

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

For a 3D comparison view:

```bash
python plot_distributions.py --plot-3d
```

You can combine with `--save`, for example:

```bash
python plot_distributions.py --plot-3d --save distributions_3d.png
```

`plot_distributions.py` initializes an array in `[0.01, 1]` with 1000 evenly spaced steps using:

```python
x = np.linspace(0.01, 1.0, 1000)
```

It currently plots Normal, Exponential, Uniform, Beta, Gamma, and a Lorentzian family (narrow, centered, broad).
