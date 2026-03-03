import numpy as np
import math
import matplotlib.pyplot as plt
import argparse


def normal(x, mu=0.5, sigma=0.1):
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def exponential(x, lam=2):
    return lam * np.exp(-lam * x)


def uniform(x, a=0.2, b=0.8):
    return np.where((x >= a) & (x <= b), 1 / (b - a), 0)


def beta(x, alpha=2, beta_param=5):
    return (x ** (alpha - 1) * (1 - x) ** (beta_param - 1)) / (math.gamma(alpha) * math.gamma(beta_param) / math.gamma(alpha + beta_param))


def gamma(x, k=2, theta=0.2):
    return (x ** (k - 1) * np.exp(-x / theta)) / (math.gamma(k) * theta ** k)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", type=str, help="Save plot to file")
    args = parser.parse_args()

    x = np.linspace(0.01, 1.0, 1000)

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    axes[0, 0].plot(x, normal(x, mu=0.5, sigma=0.1), "b-", label="Normal")
    axes[0, 0].fill_between(x, normal(x, mu=0.5, sigma=0.1), alpha=0.3)
    axes[0, 0].set_title("Normal (μ=0.5, σ=0.1)")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(x, exponential(x, lam=2), "g-", label="Exponential")
    axes[0, 1].fill_between(x, exponential(x, lam=2), alpha=0.3)
    axes[0, 1].set_title("Exponential (λ=2)")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[0, 2].plot(x, uniform(x), "r-", label="Uniform")
    axes[0, 2].fill_between(x, uniform(x), alpha=0.3)
    axes[0, 2].set_title("Uniform (a=0.2, b=0.8)")
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    axes[1, 0].plot(x, beta(x, alpha=2, beta_param=5), "m-", label="Beta")
    axes[1, 0].fill_between(x, beta(x, alpha=2, beta_param=5), alpha=0.3)
    axes[1, 0].set_title("Beta (α=2, β=5)")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(x, gamma(x, k=2, theta=0.2), "c-", label="Gamma")
    axes[1, 1].fill_between(x, gamma(x, k=2, theta=0.2), alpha=0.3)
    axes[1, 1].set_title("Gamma (k=2, θ=0.2)")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    axes[1, 2].axis("off")

    for row in axes:
        for ax in row:
            ax.set_xlabel("x")
            ax.set_ylabel("y")

    plt.suptitle("Probability Distributions", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if args.save:
        plt.savefig(args.save, dpi=150)
        print(f"Saved to {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
