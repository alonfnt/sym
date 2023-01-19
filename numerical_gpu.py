from concurrent.futures import ProcessPoolExecutor, as_completed
import itertools
import pathlib

from absl import app, flags
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import coo_array
from tqdm.auto import tqdm

flags.DEFINE_string("output", short_name="o", default="out_numerical/", help="Path of output files")
flags.DEFINE_integer("size", short_name="n", default=10_000, help="Network size")
flags.DEFINE_integer("repetitions", default=10, help="Number of independent networks")
flags.DEFINE_list(
    "t", default=[2, 4, 8], help="Distribution parameters for triangular edge generation"
)
flags.DEFINE_list("s", default=[2], help="Distribution parameters for edge generation")
FLAGS = flags.FLAGS


def random_clustered_graph_matrix(edge_degrees, triangular_degrees):
    num_nodes = len(edge_degrees)

    nodes = np.arange(num_nodes, dtype=int)
    edges = np.repeat(nodes, edge_degrees)
    triangular_edges = np.repeat(nodes, triangular_degrees)

    np.random.shuffle(edges)
    np.random.shuffle(triangular_edges)

    edges = edges.reshape((2, -1))
    triangular = triangular_edges.reshape((3, -1))

    make_pairs = lambda t: np.concatenate([t[:2], t[1:], [t[2], t[0]]], axis=-1)
    triangular_edges = make_pairs(triangular)
    edges = np.concatenate([edges, triangular_edges], axis=-1)
    rows, cols = np.sort(edges, axis=0)

    A = np.zeros(shape=(num_nodes, num_nodes))
    A[rows, cols] = 1
    A[cols, rows] = 1
    return A


find_eigenvalues = jax.jit(jnp.linalg.eigvalsh)


def main(args):
    del args

    output_dir = pathlib.Path(FLAGS.output)
    output_dir.mkdir(exist_ok=True, parents=True)

    for (kt, ks) in tqdm(itertools.product(FLAGS.t, FLAGS.s), total=len(FLAGS.t) * len(FLAGS.s)):
        while True:
            first_degrees = np.random.poisson(ks, (FLAGS.repetitions * 10, FLAGS.size))
            second_degrees = np.random.poisson(kt, (FLAGS.repetitions * 10, FLAGS.size))

            mask = (first_degrees.sum(axis=1) % 2 == 0) & (second_degrees.sum(axis=1) % 3 == 0)
            if mask.sum() >= FLAGS.repetitions:
                break
        first_degrees = first_degrees[mask][: FLAGS.repetitions]
        second_degrees = second_degrees[mask][: FLAGS.repetitions]

        eigenvalues = []
        with ProcessPoolExecutor() as executor:
            results = [
                executor.submit(random_clustered_graph_matrix, s, t)
                for (s, t) in zip(first_degrees, second_degrees)
            ]
            for res in as_completed(results):
                A = jnp.array(res.result()) / np.sqrt(2*kt+ks)
                w = find_eigenvalues(A)
                eigenvalues.append(np.array(w))

        eigenvalues = np.array(eigenvalues).flatten()
        np.save(output_dir.joinpath(f"eigenvalues_{ks}_{kt}_{FLAGS.size}.npy"), eigenvalues)

        fig, ax = plt.subplots(1, 1, figsize=(5, 4))
        _ = plt.hist(eigenvalues, 1000, density=True, facecolor="g", alpha=0.75)
        ax.set(
            xlabel=r"$\lambda$",
            ylabel=r"$\rho(\lambda)$",
            title=rf"Histogram of spectrum ({ks=}, {kt=} N={FLAGS.size})",
        )
        plt.tight_layout()
        fig.savefig(output_dir.joinpath(f"spectral_density_{ks}_{kt}_{FLAGS.size}.pdf"), dpi=300)


if __name__ == "__main__":
    app.run(main)
