import concurrent.futures
import argparse
from functools import partial
import pathlib

from absl import app, flags
import matplotlib.pyplot as plt
import numba
import numpy as np
from scipy import stats
from tqdm import tqdm
#python3 popdyn.py --M=500000 --ct=2 --cs=2 --eval_points=100 --passes=20 --kmax=1000

parser = argparse.ArgumentParser()
parser.add_argument("--M", default=1000_00, help="Network size", type=int)
parser.add_argument("--kmin", default=0, help="Minimum value of K", type=int)
parser.add_argument("--kmax", default=30, help="Maximum value of K", type=int)
parser.add_argument("--eval_points", default=100, help="Number of eigenvalues to evaluate", type=int)
parser.add_argument("--cs", default=2, help="", type=int)
parser.add_argument("--ct", default=2, help="", type=int)
parser.add_argument("--passes", default=10000000, help="Number of passes through the network", type=int)
parser.add_argument("--eps", default=1e-4, help="", type=float)
parser.add_argument("--J", default=1, help="spin-glass coupling?", type=float)
parser.add_argument("--output", default="out_popdyn", help="Output directory")
args = parser.parse_args()

def excess_deg_dist(k, c):
    pk = stats.poisson.pmf
    qk = k * pk(k, c) / c 
    return stats.rv_discrete(values=(k, qk))


@numba.njit
def relax_network_messages(w, R, Rm, ds, dt, s1, t1,  eps, J):
    N = len(R)
    node = np.random.choice(N)
    #for node in range(N):
    # Get the indexes of those edges.
    e1_ind = np.random.randint(0, N, size=s1[0])
    e2_ind = np.random.randint(0, N, size=s1[1])
    e3_ind = np.random.randint(0, N, size=ds[2] - 1)

    t1_ind = np.random.randint(0, N, size=dt[0] - 1)
    t2_ind = np.random.randint(0, N, size=dt[1] - 1)
    t3_ind = np.random.randint(0, N, size=t1)

    # Compute the components of the new message matrix.
    a11 = w + 1j * eps - (J**2) * (np.sum(R[e1_ind]) + np.sum(Rm[t1_ind]))
    a22 = w + 1j * eps - (J**2) * (np.sum(R[e2_ind]) + np.sum(Rm[t2_ind]))

    Rm[node] = np.linalg.inv(np.array([[a11, -J], [-J, a22]]))

    # Do something else to update R
    node2 = np.random.choice(N)
    R[node2] = 1 / (w + 1j * eps - (J**2) * (np.sum(R[e3_ind]) + np.sum(Rm[t3_ind])))
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
    s = stats.poisson.rvs(cs, size=M)
    t = stats.poisson.rvs(ct, size=M)
    s1 = stats.poisson.rvs(cs, size=[passes,2])
    t1 = stats.poisson.rvs(ct, size=passes)
    qs = excess_deg_dist(k, cs)
    qt = excess_deg_dist(k, ct)

    # Initialise
    R, Rm = init_resolvents_matrices(M)

    # Run the simulation
    # Generate the edges distributions beforehand.
    ds = qs.rvs(size=(passes, 3))
    dt = qt.rvs(size=(passes, 3))
    for i in range(passes):
        # Pass through the network sending messages and doing whatever Tuan says.
        R, Rm = relax_network_messages(w, R, Rm, ds[i], dt[i], s1[i,:], t1[i], eps, J)

    # Finally, compute the delta by using the previous network.
    Rf = np.empty(shape=M, dtype=np.complex128)


    for node in range(M):
        s_ind = np.random.randint(0, M, size=s[node])
        t_ind = np.random.randint(0, M, size=t[node])
        Rf[node] = 1 / (w + 1j * eps - ((J)**2) * (np.sum(R[s_ind]) + np.sum(Rm[t_ind])))

    rho = (1.0 / (M * np.pi)) * np.imag(np.sum(Rf))
    return w, rho


def main(args):

    results = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Function to run on independent threads for each eigenvalue f(w)
        f = partial(
            population_dynamics,
            M=args.M,
            kmin=args.kmin,
            kmax=args.kmax,
            cs=args.cs,
            ct=args.ct,
            passes=args.passes,
            eps=args.eps,
            J=args.J/np.sqrt(2*args.ct+args.cs),
        )
        jobs = [executor.submit(f, w) for w in np.linspace(-3, 3, args.eval_points)]

        results = [job.result() for job in tqdm(concurrent.futures.as_completed(jobs), total=args.eval_points)]

    # Sort by eigenvalue
    results = np.array(results)
    results = results[results[:, 0].argsort()]

    # Save the results
    output_dir = pathlib.Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)

    filename = f"{args.cs}_{args.ct}_{args.M}"
    np.save(output_dir.joinpath(f"{filename}.npy"), results)

    lambdas, rhos = results.T
    fig = plt.figure()
    plt.xlabel(r'$\lambda$')
    plt.ylabel(r'$\rho(\lambda)$')
    plt.title(rf'Histogram of spectrum (ks={args.cs}, kt={args.ct} N={args.M})')
    plt.plot(lambdas, rhos, "-")
    plt.tight_layout()
    fig.savefig(output_dir.joinpath(f"{filename}.pdf"), dpi=300)


if __name__ == "__main__":
    main(args)
