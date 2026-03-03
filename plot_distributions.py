import numpy as np
import matplotlib.pyplot as plt


def main() -> None:
    x = np.linspace(0.0, 1.0, 1000)

    plt.figure(figsize=(8, 4))
    plt.plot(x, x, label="x")
    plt.title("Distribution Plot Framework")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
