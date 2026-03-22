import numpy as np
import math
import matplotlib
matplotlib.use("macosx")
import matplotlib.pyplot as plt
import argparse
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


DISTRIBUTION_CHOICES = ["normal", "exponential", "uniform", "beta", "gamma", "lorentzian", "weighted_lorentzian"]


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


def weighted_lorentzian(x, weights=(0.5, 1.0, 0.3), x0s=(0.3, 0.5, 0.75), gammas=(0.04, 0.08, 0.06)):
    result = sum(w * lorentzian(x, x0, g) for w, x0, g in zip(weights, x0s, gammas))
    area = float(np.trapezoid(result, x))
    return result / area if area > 0 else result


def distribution_values(name, x):
    if name == "normal":
        return normal(x, mu=0.5, sigma=0.1)
    if name == "exponential":
        return exponential(x, lam=2)
    if name == "uniform":
        return uniform(x)
    if name == "beta":
        return beta(x, alpha=2, beta_param=5)
    if name == "gamma":
        return gamma(x, k=2, theta=0.2)
    if name == "lorentzian":
        return lorentzian(x, x0=0.5, gamma_param=0.1)
    if name == "weighted_lorentzian":
        return weighted_lorentzian(x)
    raise ValueError(f"Unknown distribution: {name}")


def mixed_distribution(x, base_name, added_name, added_amplitude=1.0, renormalize=True):
    non_negative_amplitude = max(0.0, added_amplitude)
    base_weight = 1.0 / (1.0 + non_negative_amplitude)
    added_weight = non_negative_amplitude / (1.0 + non_negative_amplitude)

    y_base = distribution_values(base_name, x)
    y_added = distribution_values(added_name, x)
    y_mix = base_weight * y_base + added_weight * y_added

    if renormalize:
        area = float(np.trapezoid(y_mix, x))
        if area > 0:
            y_mix = y_mix / area

    return y_mix, base_weight, added_weight


def probability_between(
    name,
    lower,
    upper,
    domain_min=0.01,
    domain_max=1.0,
    mix_with=None,
    mix_amplitude=1.0,
    renormalize=True,
):
    lo = max(min(lower, upper), domain_min)
    hi = min(max(lower, upper), domain_max)
    if lo >= hi:
        return 0.0

    x = np.linspace(domain_min, domain_max, 5000)
    if mix_with is None:
        y = distribution_values(name, x)
    else:
        y, _, _ = mixed_distribution(x, name, mix_with, added_amplitude=mix_amplitude, renormalize=renormalize)

    mask = (x >= lo) & (x <= hi)
    return float(np.trapezoid(y[mask], x[mask]))


def plot_all_3d_subplots(x, mix_with=None, mix_base="lorentzian", mix_amplitude=1.0, renormalize=True):
    fig = plt.figure(figsize=(18, 11))
    n_curves = 8

    # Normal — vary sigma
    ax = fig.add_subplot(2, 3, 1, projection="3d")
    for i, sigma in enumerate(np.linspace(0.05, 0.25, n_curves)):
        y_pos = np.full_like(x, i, dtype=float)
        ax.plot(x, y_pos, normal(x, mu=0.5, sigma=sigma), color=plt.cm.Blues(0.3 + 0.7 * i / n_curves), linewidth=1.5)
    ax.set_title("Normal (vary σ)")
    ax.set_xlabel("x"); ax.set_ylabel("curve"); ax.set_zlabel("density")
    ax.set_yticks(range(n_curves))
    ax.set_yticklabels([f"{s:.2f}" for s in np.linspace(0.05, 0.25, n_curves)], fontsize=6)
    ax.view_init(elev=25, azim=-55)

    # Exponential — vary lambda
    ax = fig.add_subplot(2, 3, 2, projection="3d")
    for i, lam in enumerate(np.linspace(0.5, 5.0, n_curves)):
        y_pos = np.full_like(x, i, dtype=float)
        ax.plot(x, y_pos, exponential(x, lam=lam), color=plt.cm.Greens(0.3 + 0.7 * i / n_curves), linewidth=1.5)
    ax.set_title("Exponential (vary λ)")
    ax.set_xlabel("x"); ax.set_ylabel("curve"); ax.set_zlabel("density")
    ax.set_yticks(range(n_curves))
    ax.set_yticklabels([f"{l:.1f}" for l in np.linspace(0.5, 5.0, n_curves)], fontsize=6)
    ax.view_init(elev=25, azim=-55)

    # Uniform — vary width
    ax = fig.add_subplot(2, 3, 3, projection="3d")
    for i, half_w in enumerate(np.linspace(0.05, 0.45, n_curves)):
        y_pos = np.full_like(x, i, dtype=float)
        ax.plot(x, y_pos, uniform(x, a=0.5 - half_w, b=0.5 + half_w), color=plt.cm.Reds(0.3 + 0.7 * i / n_curves), linewidth=1.5)
    ax.set_title("Uniform (vary width)")
    ax.set_xlabel("x"); ax.set_ylabel("curve"); ax.set_zlabel("density")
    ax.set_yticks(range(n_curves))
    ax.set_yticklabels([f"{2*w:.2f}" for w in np.linspace(0.05, 0.45, n_curves)], fontsize=6)
    ax.view_init(elev=25, azim=-55)

    # Beta — vary alpha
    ax = fig.add_subplot(2, 3, 4, projection="3d")
    for i, alpha in enumerate(np.linspace(1.0, 8.0, n_curves)):
        y_pos = np.full_like(x, i, dtype=float)
        ax.plot(x, y_pos, beta(x, alpha=alpha, beta_param=5), color=plt.cm.Purples(0.3 + 0.7 * i / n_curves), linewidth=1.5)
    ax.set_title("Beta (vary α, β=5)")
    ax.set_xlabel("x"); ax.set_ylabel("curve"); ax.set_zlabel("density")
    ax.set_yticks(range(n_curves))
    ax.set_yticklabels([f"{a:.1f}" for a in np.linspace(1.0, 8.0, n_curves)], fontsize=6)
    ax.view_init(elev=25, azim=-55)

    # Gamma — vary k
    ax = fig.add_subplot(2, 3, 5, projection="3d")
    for i, k in enumerate(np.linspace(1.0, 5.0, n_curves)):
        y_pos = np.full_like(x, i, dtype=float)
        ax.plot(x, y_pos, gamma(x, k=k, theta=0.2), color=plt.cm.cool(0.1 + 0.8 * i / n_curves), linewidth=1.5)
    ax.set_title("Gamma (vary k, θ=0.2)")
    ax.set_xlabel("x"); ax.set_ylabel("curve"); ax.set_zlabel("density")
    ax.set_yticks(range(n_curves))
    ax.set_yticklabels([f"{k:.1f}" for k in np.linspace(1.0, 5.0, n_curves)], fontsize=6)
    ax.view_init(elev=25, azim=-55)

    # Weighted Lorentzian — vary middle-peak weight
    ax = fig.add_subplot(2, 3, 6, projection="3d")
    mid_weights = np.linspace(0.1, 2.0, n_curves)
    for i, mw in enumerate(mid_weights):
        y_pos = np.full_like(x, i, dtype=float)
        wl = weighted_lorentzian(x, weights=(0.5, mw, 0.3), x0s=(0.3, 0.5, 0.75), gammas=(0.04, 0.08, 0.06))
        ax.plot(x, y_pos, wl, color=plt.cm.autumn(0.1 + 0.8 * i / n_curves), linewidth=1.5)
    if mix_with is not None:
        y_mix, base_w, added_w = mixed_distribution(x, mix_base, mix_with, added_amplitude=mix_amplitude, renormalize=renormalize)
        y_pos = np.full_like(x, n_curves, dtype=float)
        ax.plot(x, y_pos, y_mix, "--", color="tab:purple", linewidth=2)
    ax.set_title("Weighted Lorentzian (vary w₂)")
    ax.set_xlabel("x"); ax.set_ylabel("curve"); ax.set_zlabel("density")
    ax.set_yticks(range(n_curves))
    ax.set_yticklabels([f"{mw:.1f}" for mw in mid_weights], fontsize=6)
    ax.view_init(elev=25, azim=-55)

    plt.suptitle("Probability Distributions (3D Parameter Families)", fontsize=13, fontweight="bold")
    plt.subplots_adjust(left=0.02, right=0.98, top=0.93, bottom=0.05, wspace=0.35, hspace=0.3)


def plot_distributions_3d(x, mix_base=None, mix_with=None, mix_amplitude=1.0, renormalize=True):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    distributions = [
        ("Normal", normal(x, mu=0.5, sigma=0.1), "b"),
        ("Exponential", exponential(x, lam=2), "g"),
        ("Uniform", uniform(x), "r"),
        ("Beta", beta(x, alpha=2, beta_param=5), "m"),
        ("Gamma", gamma(x, k=2, theta=0.2), "c"),
        ("Lorentzian", lorentzian(x, x0=0.5, gamma_param=0.1), "k"),
        ("Weighted Lorentzian", weighted_lorentzian(x), "darkorange"),
    ]

    if mix_with is not None and mix_base is not None:
        y_mix, _, _ = mixed_distribution(
            x,
            mix_base,
            mix_with,
            added_amplitude=mix_amplitude,
            renormalize=renormalize,
        )
        mix_label = f"Mix: {mix_base}+{mix_with}"
        distributions.append((mix_label, y_mix, "purple"))

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
    parser.add_argument("--plot-3d", action="store_true", help="Render combined 3D line plot (all distributions on one axis)")
    parser.add_argument(
        "--probability",
        choices=DISTRIBUTION_CHOICES,
        help="Compute P(lower <= X <= upper) for a distribution",
    )
    parser.add_argument("--lower", type=float, default=0.2, help="Lower bound for probability calculation")
    parser.add_argument("--upper", type=float, default=0.8, help="Upper bound for probability calculation")
    parser.add_argument(
        "--mix-base",
        choices=DISTRIBUTION_CHOICES,
        default="lorentzian",
        help="Base distribution for mixed-curve plotting",
    )
    parser.add_argument(
        "--mix-with",
        choices=DISTRIBUTION_CHOICES,
        help="Additional distribution to add to the curve",
    )
    parser.add_argument(
        "--mix-amplitude",
        type=float,
        default=1.0,
        help="Relative amplitude for the added distribution before renormalization",
    )
    parser.add_argument(
        "--no-renormalize",
        action="store_true",
        help="Disable post-mix renormalization to unit area",
    )
    args = parser.parse_args()

    x = np.linspace(0.01, 1.0, 1000)
    renormalize = not args.no_renormalize

    if args.probability:
        prob = probability_between(
            args.probability,
            args.lower,
            args.upper,
            domain_min=0.01,
            domain_max=1.0,
            mix_with=args.mix_with,
            mix_amplitude=args.mix_amplitude,
            renormalize=renormalize,
        )
        lo = min(args.lower, args.upper)
        hi = max(args.lower, args.upper)
        if args.mix_with is None:
            print(f"P({lo:.4f} <= X <= {hi:.4f}) for {args.probability}: {prob:.6f}")
        else:
            _, base_w, added_w = mixed_distribution(
                np.linspace(0.01, 1.0, 1000),
                args.probability,
                args.mix_with,
                added_amplitude=args.mix_amplitude,
                renormalize=renormalize,
            )
            print(
                f"P({lo:.4f} <= X <= {hi:.4f}) for mix[{args.probability}+{args.mix_with}] "
                f"(weights {base_w:.3f}/{added_w:.3f}, renormalized={renormalize}): {prob:.6f}"
            )
        return

    if args.plot_3d:
        plot_distributions_3d(
            x,
            mix_base=args.mix_base,
            mix_with=args.mix_with,
            mix_amplitude=args.mix_amplitude,
            renormalize=renormalize,
        )
    else:
        plot_all_3d_subplots(
            x,
            mix_with=args.mix_with,
            mix_base=args.mix_base,
            mix_amplitude=args.mix_amplitude,
            renormalize=renormalize,
        )

    if args.save:
        plt.savefig(args.save, dpi=150)
        print(f"Saved to {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
