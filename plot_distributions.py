import numpy as np
import math
import matplotlib.pyplot as plt
import argparse
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


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


def lorentzian(x, x0=0.5, gamma_param=0.1):
    return (1 / np.pi) * (gamma_param / ((x - x0) ** 2 + gamma_param ** 2))


def plot_distributions_3d(x):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    distributions = [
        ("Normal", normal(x, mu=0.5, sigma=0.1), "b"),
        ("Exponential", exponential(x, lam=2), "g"),
        ("Uniform", uniform(x), "r"),
        ("Beta", beta(x, alpha=2, beta_param=5), "m"),
        ("Gamma", gamma(x, k=2, theta=0.2), "c"),
        ("Lorentzian", lorentzian(x, x0=0.5, gamma_param=0.1), "k"),
    ]

    y_positions = np.arange(len(distributions))

    for idx, (name, y_values, color) in enumerate(distributions):
        y = np.full_like(x, y_positions[idx], dtype=float)
        ax.plot(x, y, y_values, color=color, linewidth=2, label=name)

    ax.set_xlabel("x")
    ax.set_ylabel("Distribution")
    ax.set_zlabel("Density")
    ax.set_yticks(y_positions)
    ax.set_yticklabels([name for name, _, _ in distributions])
    ax.set_title("3D Probability Distributions")
    ax.view_init(elev=28, azim=-60)
    ax.legend(loc="upper left")
    fig.subplots_adjust(left=0.03, right=0.95, bottom=0.08, top=0.92)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", type=str, help="Save plot to file")
    parser.add_argument("--plot-3d", action="store_true", help="Render 3D line plot")
    args = parser.parse_args()

    x = np.linspace(0.01, 1.0, 1000)

    if args.plot_3d:
        plot_distributions_3d(x)
        if args.save:
            plt.savefig(args.save, dpi=150)
            print(f"Saved to {args.save}")
        else:
            plt.show()
        return

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

    axes[1, 2].plot(x, lorentzian(x, x0=0.3, gamma_param=0.05), "y-", label="Lorentzian (narrow)")
    axes[1, 2].plot(x, lorentzian(x, x0=0.5, gamma_param=0.1), "k-", label="Lorentzian (centered)")
    axes[1, 2].plot(x, lorentzian(x, x0=0.7, gamma_param=0.2), "orange", label="Lorentzian (broad)")
    axes[1, 2].set_title("Lorentzian Family")
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

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
