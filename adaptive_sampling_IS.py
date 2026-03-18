
"""
Adaptive Importance Sampling for Epidemic Threshold Learning

Goal

Generate parameter samples concentrated near the epidemic threshold R0 ≈ 1.

Pipeline

1. Sobol sampling for initial global exploration
2. Estimate proposal density q(theta) using Kernel Density Estimation 
3. Compute target density p(R0) peaked near R0 = 1
4. Compute importance weights w = p / q
5. Systematic resampling
6. Resample-move step (Gaussian jitter)
7. Iterate


Parameters
tau-per-contact transmission rate
gamma-recovery rate
rho-initial infected seed fraction
Ro= tau/gamma* <k²>/<k> for BA network

"""

import numpy as np
import networkx as nx
from scipy.stats import qmc
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler
import EoN
from pathlib import Path
import pickle
from scipy.stats import qmc
from pathlib import Path
import argparse
from tqdm import tqdm
from pathlib import Path
import csv

# PARAMETERS
N = 10000                  # network size
m = 5                      # Barabasi–Albert attachment parameter
seed = 4849
tmax=200 
n_timepoints=200
initial_samples =500       # initial Sobol samples
iterations = 10            # AIS iterations
sigma = 0.15               # width of R0 target distribution
kernel_bandwidth = 0.15    # KDE bandwidth
jitter_scale = 0.02        # perturbation scale after resampling
iterations=3       #iterations fro resampling
n_replicates=3  # replicates of parameter sets 
PARAM_RANGES = {
    'tau'  : (0.0024, 0.017), transmission rate
    'gamma': (0.07,   0.50),   # for 2-14 infectious period
    'rho'  : (0.001,  0.010)   # proportion of individuals infected at time 0
}

PARAM_NAMES = ['tau', 'gamma', 'rho']
jitter_fraction=0.05    # pertubation fraction, using gausisan
output_path = Path('epidemic_data_age_adaptive_sobol.pkl')

# NETWORK STATISTICS
def BA_network_stats(N, m, seed=4849):
    """
    Compute statistics of the Barabasi Albert network.

    Returns
    ratio  # <k²>/<k> for BA network
    """
 #remember Ro= tau/gamma* <k²>/<k> for BA network

    G = nx.barabasi_albert_graph(N, m, seed=seed)

    degrees = np.array([d for _, d in G.degree()])
    stats = {
            'k_avg'  : float(degrees.mean()), # first moment
            'k2_avg' : float((degrees ** 2).mean()), # second , moment
            'ratio'  : float((degrees ** 2).mean() / degrees.mean()), #  <k²>/<k> for BA network
            'k_std'  : float(degrees.std()),  
            'k_max'  : int(degrees.max()),
        }
    

    print(f"  <k>      = {stats['k_avg']:.2f}")
    print(f"  <k²>     = {stats['k2_avg']:.2f}")
    print(f"  <k²>/<k> = {stats['ratio']:.2f}")
    print(f"  k_std    = {stats['k_std']:.2f}")
    print(f"  k_max    = {stats['k_max']}")
 

    return G, stats


# R0 COMPUTATION
def compute_R0(samples,ratio):
    """
    Compute epidemic reproduction number.

    R0 = (tau/gamma) * <k²>/<k>
    """

    tau = samples[:, 0]
    gamma = samples[:, 1]

    R0 = (tau / gamma) * ratio  

    return R0


# TARGET DISTRIBUTION

def target_density(R0, sigma=sigma):
    """
    Target distribution emphasizing R0 ≈ 1.

    Gaussian centered at epidemic threshold.
    """

    return np.exp(-0.5 * ((R0 - 1.0) / sigma) ** 2)

# SOBOL INITIAL SAMPLING
def generate_sobol_samples(n_samples=initial_samples, seed=seed):
    """
    Generate Sobol low-discrepancy samples in parameter space. This is my proporsal distribution
    """

    sampler = qmc.Sobol(d=3, scramble=True, seed=seed)

    n_pow2 = 2 ** int(np.ceil(np.log2(max(n_samples, 2))))

    samples_unit = sampler.random(n=n_pow2)[:n_samples]

    samples = np.zeros_like(samples_unit)

    for i, name in enumerate(PARAM_NAMES):
        lo, hi = PARAM_RANGES[name]
        samples[:, i] = samples_unit[:, i] * (hi - lo) + lo

    disc = qmc.discrepancy(samples_unit)
    print(f"Sobol discrepancy: {disc:.6f}")

    return samples


# PROPOSAL ESTIMATION 

def estimate_proposal(samples, bandwith=kernel_bandwidth):
    """
    Estimate proposal density q(theta) using Kernel Density Estimation.
    Estimate the proposal density q(theta) at each sample using Kernel Density Estimation (KDE).
    We standardise the parameter space first because tau, gamma, rho have very different scales. StandardScaler (zero mean, unit variance) prevents KDE from treating the small-range rho dimension as irrelevant.

     samples : np.array (N, 3) — physical parameter values
     Returns : q (N,), fitted kde, fitted scaler
    """

    scaler = StandardScaler()
    X = scaler.fit_transform(samples)

    kde = KernelDensity(kernel='gaussian', bandwidth=bandwith)
    kde.fit(X)

    log_q = kde.score_samples(X)
    q = np.exp(log_q)

    return q, kde, scaler

# IMPORTANCE WEIGHTS


def compute_is_weights(samples, q, ratio):
    """
    Importance weights:

    w = p(R0) / q(theta)
    """

    R0 = compute_R0(samples,ratio)

    p = target_density(R0)

    raw_w = p / (q + 1e-12) # safegurding, 1e-12 guard prevents division by zero in very sparse regions.

    w = raw_w / raw_w.sum()

    return w, R0


# SYSTEMATIC RESAMPLING
def systematic_resample(samples, weights):
    """
    Systematic resampling — draw N new samples from the IS-weighted set.

    WHY SYSTEMATIC over other resampling methods like multinomial:  (I need your advice here ************)
        - Reduces resampling variance
        - Preserves diversity (less likely to collapse to a few points)
        - O(N) cost vs O(N log N) for some other methods
        - Single random number u0 drives the whole procedure

    samples : np.array (N, 3)
    weights : np.array (N,) normalised weights summing to 1
    Returns : np.array (N, 3) resampled parameter sets
    """
    N = len(samples)

    positions = (np.random.uniform() + np.arange(N)) / N
    cumulative = np.cumsum(weights)

    idx = np.searchsorted(cumulative, positions)
    idx       = np.clip(idx, 0, N - 1) #clamp — np.searchsorted can return N(one past the end) due to floating point in cumsum.
        
    return samples[idx]


# RESAMPLE-MOVE STEP

def jitter_samples(samples, fraction=jitter_fraction):
    """
    Adding small Gaussian perturbations to prevent particle collapse.
    After systematic resampling, many particles are duplicates.
    A small jitter diversifies them while keeping them close to
    their original positions.

    """

    perturbed = samples.copy()
 
    for i, name in enumerate(PARAM_NAMES):
        lo, hi        = PARAM_RANGES[name]
        scale         = fraction * (hi - lo)                  
        noise         = np.random.normal(0, scale, len(samples))
        perturbed[:, i] = np.clip(samples[:, i] + noise, lo, hi)  
 
    return perturbed


# EFFECTIVE SAMPLE SIZE
def compute_ess(weights):
    """
   Effective Sample Size — how much information the IS weights retain.
   Like even though I have N samples , how many independent samples are really contributing
    If all weights are equal then all samples contribute equally, ESS ~1 ( perfect balance )
    if one weight dominates , only one sample effectively contributes , ESS~1
    For normalised weights: ESS = 1 / sum(w_i^2)
     Range: [1, N]

    ESS/N close to 1 then proposal matches target, no wasted samples.
    ESS/N close to 0 then proposal very different from target, most weight
                       on a few samples, high variance estimates.
    """
    total = weights.sum()
    if total <= 0:           # guard against weights.sum() = 0 > NaN.
        return 0.0
    w = weights / total
    return float(1.0 / np.sum(w ** 2))


# ADAPTIVE IMPORTANCE SAMPLING

def adaptive_IS_sampler(ratio,initial_samples=initial_samples,iterations=iterations,seed=seed,verbose=True):
    
    np.random.seed(seed)
 
    # Sobol initialisation 
    sobol_initial = generate_sobol_samples(initial_samples, seed=seed)
    samples       = sobol_initial.copy()
    ess_history   = []
 
    if verbose:
        print(f"\n{'─'*55}")
        print(f"  Adaptive IS  |  iterations={iterations}, sigma={sigma}")
        print(f"{'─'*55}")
 
    # Adaptive loop 
    for i in range(iterations):
 
        # (a) Estimate proposal via KDE
        q, kde, scaler = estimate_proposal(samples)
 
        # (b) IS weights
        weights, R0 = compute_is_weights(samples, q, ratio)
 
        # (c) ESS before resampling
        ess = compute_ess(weights)
        ess_history.append(ess)
 
        # (d) Systematic resample
        samples = systematic_resample(samples, weights)
 
        # (e) Jitter to diversify duplicates
        samples = jitter_samples(samples)
 
        if verbose:
            R0_after = compute_R0(samples, ratio)
            near = 100 * np.mean(np.abs(R0_after - 1) < 0.2)
            print(f"  Iter {i+1}/{iterations} | "
                  f"ESS={ess:.1f} ({100*ess/len(samples):.1f}%)  |  "
                  f"R0 mean={R0_after.mean():.3f}  |  "
                  f"near thresh={near:.1f}%")
 
    # initial Sobol + resampled 
    # deduplicated by rounding to 6 decimal places to avoid the same parameter set appearing in both train and test splits (data leakage).
    combined = np.vstack([sobol_initial, samples])
    _, unique_idx = np.unique(
        np.round(combined, decimals=6), axis=0, return_index=True
    )
    final_samples = combined[np.sort(unique_idx)]
 
    return final_samples, ess_history   

# RUN MULTIPLE SIR REPLICATES

def run_sir_replicates(G, tau, gamma, rho,n_replicates=n_replicates,tmax=tmax,n_timepoints=n_timepoints):
    #Replicates reduce stochastic noise — we return the mean and std

    t_fixed = np.linspace(0, tmax, n_timepoints)
    #N_nodes = G.number_of_nodes() # normilization to make training faster

    S_runs, I_runs, R_runs = [], [], []

    for _ in range(n_replicates):
        t, S, I, R = EoN.fast_SIR(G, tau, gamma, rho=rho, tmax=tmax)

        S_runs.append(np.interp(t_fixed, t, S) )
        I_runs.append(np.interp(t_fixed, t, I) )
        R_runs.append(np.interp(t_fixed, t, R) )

    S_arr = np.array(S_runs)
    I_arr = np.array(I_runs)
    R_arr = np.array(R_runs)

    return {
        't'    : t_fixed,
        'S'    : S_arr.mean(axis=0),
        'I'    : I_arr.mean(axis=0),
        'R'    : R_arr.mean(axis=0),
        'S_std': S_arr.std(axis=0),
        'I_std': I_arr.std(axis=0),
        'R_std': R_arr.std(axis=0),
        'n_replicates': n_replicates,
    }

# RUN A BATCH OF PARAMETER SETS

def run_batch(G,params_array,n_replicates=n_replicates,tmax=tmax,n_timepoints=n_timepoints):

    results = []

    for row in tqdm(params_array,
                    desc=f"Running {len(params_array)} parameter sets"):

        tau, gamma, rho = row[0],row[1],row[2], #  # explicit, never crashes

        output = run_sir_replicates( G, float(tau), float(gamma),float(rho),n_replicates,tmax,n_timepoints)

        results.append({

            "params": {
                "tau": float(tau),
                "gamma": float(gamma),
                "rho": float(rho)
            },

            "output": output
        })

    return results




def build_dataset(all_sims, G, net, ess_history):

    dataset = {
        'simulations': all_sims,
        'network': {
            'type'   : 'barabasi_albert',
            'N'      : G.number_of_nodes(),
            'm'      : m,
            'K_avg'  : net['k_avg'],
            'K2_avg' : net['k2_avg'],
            'ratio'  : net['ratio'],
            'k_std'  : net['k_std'],
            'k_max'  : net['k_max'],
            'graph'  : G, 
        },
        'metadata': {
            'n_samples'         : len(all_sims),
            'n_replicates'      : n_replicates,
            'has_std'           : True,
            'noise_reduction'   : 'averaged_replicates',
            'model_type'        : 'network_SIR',
            'dimensionality'    : 3,
            'param_names'       : PARAM_NAMES,
            'param_ranges'      : PARAM_RANGES,
            'total_population'  : N,
            'tmax'              : tmax,
            'n_timepoints'      : n_timepoints,
            'network_type'      : 'barabasi_albert',
            'BA_attachment_m'   : m,
            'sampling_strategy' : 'adaptive_importance_sampling',
            'R0_formula'        : 'R0 = (tau/gamma) * <k^2>/<k>',  
            'IS_target'         : 'Gaussian around R0 = 1',
            'IS_sigma'          : sigma,
            'IS_proposal'       : 'Sobol_initial + KDE_adaptation',
            'IS_kernel_bandwidth': kernel_bandwidth,
            'IS_ess_history'    : ess_history,
            'IS_iterations'     : iterations,
            'initial_samples'   : initial_samples,
        },
    }
    return dataset

def save_dataset(dataset, filepath=output_path):
    """
    Save dataset to disk using pickle.
    """

    filepath = Path(filepath)

    with open(filepath, "wb") as f:
        pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)

    size_mb = filepath.stat().st_size / (1024**2)

    meta = dataset["metadata"]

    print("\nDATASET SAVED")
  

    print(f"File             : {filepath}")
    print(f"Size             : {size_mb:.2f} MB")

    print(f"Samples          : {meta['n_samples']}")
    print(f"Replicates       : {meta['n_replicates']}")
    print(f"Parameters       : {meta['param_names']}")
    print(f"Parameter space  : {meta['param_ranges']}")
    print(f"R0 formula       : {meta['R0_formula']}")
    print(f"Sampling method  : {meta['sampling_strategy']}")
    print(f"IS target        : {meta['IS_target']}")
    print(f"Kernel bandwidth : {meta['IS_kernel_bandwidth']}")
    print(f"ESS history      : {meta['IS_ess_history']}")

if __name__ == '__main__':
 
    # 1. Buildin the network once 
    G, net_stats = BA_network_stats(N, m, seed=seed)
    ratio        = net_stats['ratio']
 
    # 2. Adaptive IS sampling 
    #  unpacking the tuple properly
    samples, ess_history = adaptive_IS_sampler(
        ratio= ratio,
        initial_samples= initial_samples,
        iterations = iterations,
        seed= seed,
    )
 
 
    #  3. Run SIR simulations 
    all_sims = run_batch(
        G,
        samples,                   # (M, 3) — tau, gamma, rho only
        n_replicates = n_replicates,
        tmax         = tmax,
        n_timepoints = n_timepoints,
    )
 
    #  4. Build structured dataset 
    output_path = Path('epidemic_data_age_adaptive_sobol.pkl')
    dataset = build_dataset(all_sims, G, net_stats, ess_history)
 
    #5. Save
   
    save_dataset(dataset, filepath=output_path)
   






def save_csv(dataset, filepath=None):

    if filepath is None:
        base = Path("output").with_suffix('')
    else:
        base = Path(filepath).with_suffix('')

    params_path = base.with_name(base.name + '_parameters.csv')

    ratio = dataset['network']['ratio']
    sims  = dataset['simulations']

    param_fields = [
        'sim_id','tau','gamma','rho','R0',
        'peak_I','peak_time','final_R','attack_rate','near_threshold'
    ]

    with open(params_path, 'w', newline='') as f:

        writer = csv.DictWriter(f, fieldnames=param_fields)
        writer.writeheader()

        for sim_id, sim in enumerate(sims):

            tau   = sim['params']['tau']
            gamma = sim['params']['gamma']
            rho   = sim['params']['rho']

            R0 = (tau/gamma) * ratio

            I_traj = sim['output']['I']
            t_grid = sim['output']['t']
            R_traj = sim['output']['R']

            writer.writerow({
                'sim_id': sim_id,
                'tau': round(tau,6),
                'gamma': round(gamma,6),
                'rho': round(rho,6),
                'R0': round(R0,4),
                'peak_I': float(I_traj.max()),
                'peak_time': float(t_grid[I_traj.argmax()]),
                'final_R': float(R_traj[-1]),
                'attack_rate': float(R_traj[-1]),
                'near_threshold': int(abs(R0-1) < 0.2)
            })




output_path = Path('epidemic_data_age_adaptive_sobol.pkl')

dataset = build_dataset(all_sims, G, net_stats, ess_history)

save_dataset(dataset, filepath=output_path)

csv_base = Path(output_path).with_suffix('')

save_csv(dataset, filepath=csv_base)


############## OUTPUT WHEN I RUN THIS SCRIPT
""""
DATASET SAVED
File             : epidemic_data_age_adaptive_sobol.pkl
Size             : 11.63 MB
Samples          : 1000   
Replicates       : 3
Parameters       : ['tau', 'gamma', 'rho']
Parameter space  : {'tau': (0.0024, 0.017), 'gamma': (0.07, 0.5), 'rho': (0.001, 0.01)}
R0 formula       : R0 = (tau/gamma) * <k^2>/<k>
Sampling method  : adaptive_importance_sampling
IS target        : Gaussian around R0 = 1
Kernel bandwidth : 0.15
ESS history      : [154.09137699941363, 392.46567056668596, 412.7833579666411]


Now we have double the initial sobol samples due to resampling.
ESS history shows that 154 effective samples contributed information in iteration 1 , which means that weights were very uneven, only points near R₀ ≈ 1 had large weight
Iteration 2: ESS=392- more points have similar weights, because of resampling + jitter already moved samples near RO=1, only points near R₀ ≈ 1 had large weight
Iteration 3 ESS=413, now weights are almost uniform only points near R₀ ≈ 1 had large weight


To ask Alex
Data augmentation of Compartments only

"""
