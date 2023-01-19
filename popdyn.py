import concurrent.futures
from functools import partial
import pathlib

from absl import app, flags
import matplotlib.pyplot as plt
import numba
import numpy as np
from scipy import stats
from tqdm import tqdm

flags.DEFINE_integer("M", 10_000, "Network size")
flags.DEFINE_integer("kmin", 0, "Minimum value of K")
flags.DEFINE_integer("kmax", 30, "Maximum value of K")
flags.DEFINE_integer("eval_points", 100, "Number of eigenvalues to evaluate")
flags.DEFINE_integer('cs', 2, '')
flags.DEFINE_integer('ct', 4, '')
flags.DEFINE_integer("passes", 1, "Number of passes through the network")
flags.DEFINE_float("eps", 1e-4, "")
flags.DEFINE_float("J", 1, "spin-glass coupling?")
flags.DEFINE_string("output", "out_popdyn", "Output directory")
FLAGS = flags.FLAGS


def excess_deg_dist(k, c):
    pk = stats.poisson.pmf
    while True:
        qk = k * pk(k, c) / c
        if np.isclose(qk.sum(), 1):
            break
    return stats.rv_discrete(values=(k, qk))


@numba.njit
def relax_network_messages(w, R, Rm, ds, dt, eps, J):
    N = len(R)
    for node in range(N):
        # Get the indexes of those edges.
        e1_ind = np.random.randint(0, N, size=ds[node, 0])
        e2_ind = np.random.randint(0, N, size=ds[node, 1])
        e3_ind = np.random.randint(0, N, size=ds[node, 2] - 1)

        t1_ind = np.random.randint(0, N, size=dt[node, 0] - 1)
        t2_ind = np.random.randint(0, N, size=dt[node, 1] - 1)
        t3_ind = np.random.randint(0, N, size=dt[node, 2])

        # Compute the components of the new message matrix.
        a11 = w + 1j * eps - (J**2) * (np.sum(R[e1_ind]) + np.sum(Rm[t1_ind]))
        a22 = w + 1j * eps - (J**2) * (np.sum(R[e2_ind]) + np.sum(Rm[t2_ind]))

        Rm[node] = np.linalg.inv(np.array([[a11, -J], [-J, a22]]))

        # Do something else to update R
        R[node] = 1 / (w + 1j * eps - (J**2) * (np.sum(R[e3_ind]) + np.sum(Rm[t3_ind])))
    return R, Rm


@numba.njit
def random_complex(shape):
    real = np.random.random(shape)
    imaginary = np.random.random(shape)
    return real + 1j * imaginary


@numba.njit
def init_resolvents_matrices(n):
    R = random_complex(n)
    Rm = np.empty((n, 2, 2), dtype=np.complex128)
    for node in range(n):
        c = random_complex((2, 2))
        c = (c + (c.T).conj()) / 2
        Rm[node] = c

    return R, Rm


def population_dynamics(w, M, kmin, kmax, cs, ct, passes=10, J=1, eps=1e-4):
    # Generate the excess degree distributions
    k = np.arange(kmin, kmax + 1)
    qs = excess_deg_dist(k, cs)
    qt = excess_deg_dist(k, ct)

    # Initialise
    R, Rm = init_resolvents_matrices(M)

    # Run the simulation
    for _ in range(passes):
        # Generate the edges distributions beforehand.
        ds = qs.rvs(size=(M, 3))
        dt = qt.rvs(size=(M, 3))

        # Pass through the network sending messages and doing whatever Tuan says.
        R, Rm = relax_network_messages(w, R, Rm, ds, dt, eps, J)

    # Finally, compute the delta by using the previous network.
    Rf = np.empty(shape=M, dtype=np.complex128)

    s = stats.poisson.rvs(cs, size=M)
    t = stats.poisson.rvs(ct, size=M)
    for node in range(M):
        s_ind = np.random.randint(0, M, size=s[node])
        t_ind = np.random.randint(0, M, size=t[node])
        Rf[node] = 1 / (w + 1j * eps - ((J)**2) * (np.sum(R[s_ind]) + np.sum(Rm[t_ind])))

    rho = (1.0 / (M * np.pi)) * np.imag(np.sum(Rf))
    return w, rho


def main(_):

    results = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Function to run on independent threads for each eigenvalue f(w)
        f = partial(
            population_dynamics,
            M=FLAGS.M,
            kmin=FLAGS.kmin,
            kmax=FLAGS.kmax,
            cs=FLAGS.cs,
            ct=FLAGS.ct,
            passes=FLAGS.passes,
            eps=FLAGS.eps,
            J=FLAGS.J/np.sqrt(2*FLAGS.ct+FLAGS.cs),
        )
        jobs = [executor.submit(f, w) for w in np.linspace(-3, 3, FLAGS.eval_points)]

        results = [job.result() for job in tqdm(concurrent.futures.as_completed(jobs), total=FLAGS.eval_points)]

    # Sort by eigenvalue
    results = np.array(results)
    results = results[results[:, 0].argsort()]

    # Save the results
    output_dir = pathlib.Path(FLAGS.output)
    output_dir.mkdir(exist_ok=True, parents=True)

    filename = f"{FLAGS.cs}_{FLAGS.ct}_{FLAGS.M}"
    np.save(output_dir.joinpath(f"{filename}.npy"), results)

    lambdas, rhos = results.T
    fig = plt.figure()
    plt.xlabel(r'$\lambda$')
    plt.ylabel(r'$\rho(\lambda)$')
    plt.title(rf'Histogram of spectrum (ks={FLAGS.cs}, kt={FLAGS.ct} N={FLAGS.M})')
    plt.plot(lambdas, rhos, "-")
    plt.tight_layout()
    fig.savefig(output_dir.joinpath(f"{filename}.pdf"), dpi=300)


if __name__ == "__main__":
    app.run(main)
