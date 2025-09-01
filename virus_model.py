import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


# Differential system for multi-strain SEIRD
def seird_model(y, t, params):
    S = y[0]
    m = params['m']
    dSdt = 0
    E, I, R, D = [], [], [], []
    idx = 1

    # Separate compartments for each strain
    for i in range(m):
        E.append(y[idx])
        I.append(y[idx+1])
        R.append(y[idx+2])
        D.append(y[idx+3])
        idx += 4

    # Susceptible change
    for i in range(m):
        dSdt -= params['beta'][i] * S * I[i] / params['N']

    dEdt, dIdt, dRdt, dDdt = [], [], [], []
    for i in range(m):
        dE = params['beta'][i] * S * I[i] / params['N'] - params['sigma'] * E[i]
        dI = params['sigma'] * E[i] - (params['gamma'][i] + params['mu'][i]) * I[i]
        dR = params['gamma'][i] * I[i]
        dD = params['mu'][i] * I[i]
        dEdt.append(dE)
        dIdt.append(dI)
        dRdt.append(dR)
        dDdt.append(dD)

    return [dSdt] + [val for tup in zip(dEdt, dIdt, dRdt, dDdt) for val in tup]


# Simulation function
def run_simulation(N, m, beta, gamma, mu, E0, I0, R0, D0, t_inc, t_max):
    sigma = 1 / t_inc

    # Initial susceptible
    S0 = N - sum(E0) - sum(I0) - sum(R0) - sum(D0)

    # Initial condition vector
    y0 = [S0]
    for i in range(m):
        y0 += [E0[i], I0[i], R0[i], D0[i]]

    params = {
        'N': N,
        'm': m,
        'beta': beta,
        'gamma': gamma,
        'mu': mu,
        'sigma': sigma
    }

    # Time horizon
    t = np.linspace(0, t_max, t_max)

    # Solve ODE
    sol = odeint(seird_model, y0, t, args=(params,))

    # Extract results
    S = sol[:, 0]
    results = []
    idx = 1
    for i in range(m):
        Ei = sol[:, idx]
        Ii = sol[:, idx+1]
        Ri = sol[:, idx+2]
        Di = sol[:, idx+3]
        results.append((Ei, Ii, Ri, Di))
        idx += 4

    return t, S, results


# Plotting function
def plot_results(t, S, results):
    plt.figure(figsize=(12, 6))
    plt.plot(t, S, color="black", linewidth=2, label="Susceptible")

    colors = plt.cm.tab10.colors
    for i, (Ei, Ii, Ri, Di) in enumerate(results):
        plt.plot(t, Ii, color=colors[i % 10], linestyle="-", linewidth=2,
                 label=f"Infectious (Strain {i+1})")
        plt.plot(t, Di, color=colors[i % 10], linestyle="--", linewidth=2,
                 label=f"Deaths (Strain {i+1})")

    plt.xlabel("Days")
    plt.ylabel("Population count")
    plt.title("Multi-strain Virus Spread (SEIRD)")
    plt.legend(loc="upper right", bbox_to_anchor=(1.25, 1))
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# Classification function
def classify_outbreak(N, S, results):
    final_infected = N - S[-1]
    final_deaths = sum(res[3][-1] for res in results)

    infected_fraction = final_infected / N

    if infected_fraction < 0.001:
        classification = "No outbreak"
    elif infected_fraction < 0.01:
        classification = "Localized outbreak"
    elif infected_fraction < 0.2:
        classification = "Epidemic"
    else:
        classification = "Pandemic risk"

    return int(final_infected), int(final_deaths), classification


# === Example Run (replace with user input / web form later) ===
if __name__ == "__main__":
    # Example: 2 strains
    N = 1000000
    m = 2
    t_inc = 5
    t_max = 160

    beta = [0.3, 0.4]
    gamma = [1/10, 1/8]
    mu = [0.01, 0.015]

    E0 = [10, 5]
    I0 = [5, 2]
    R0 = [0, 0]
    D0 = [0, 0]

    # Run model
    t, S, results = run_simulation(N, m, beta, gamma, mu, E0, I0, R0, D0, t_inc, t_max)

    # Plot
    plot_results(t, S, results)

    # Summary
    infected, deaths, classification = classify_outbreak(N, S, results)
    print("\n===== Summary =====")
    print(f"Total infected (approx): {infected:,}")
    print(f"Total deaths: {deaths:,}")
    print(f"Classification: {classification}")
