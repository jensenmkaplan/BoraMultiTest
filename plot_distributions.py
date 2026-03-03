import numpy as np
import matplotlib.pyplot as plt


def normal(x, mu=0.5, sigma=0.1):
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def exponential(x, lam=2):
    return lam * np.exp(-lam * x)


def uniform(x, a=0.2, b=0.8):
    return np.where((x >= a) & (x <= b), 1 / (b - a), 0)


def main() -> None:
    x = np.linspace(0.0, 1.0, 1000)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].plot(x, normal(x), label="Normal (μ=0.5, σ=0.1)")
    axes[0].set_title("Normal Distribution")
    axes[0].legend()

    axes[1].plot(x, exponential(x, lam=2), label="Exponential (λ=2)")
    axes[1].set_title("Exponential Distribution")
    axes[1].legend()

    axes[2].plot(x, uniform(x), label="Uniform (a=0.2, b=0.8)")
    axes[2].set_title("Uniform Distribution")
    axes[2].legend()

    for ax in axes:
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
