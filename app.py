from flask import Flask, render_template, request
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import io
import base64

app = Flask(__name__)

# ====== MODEL ======
def model(y, t, params):
    S = y[0]
    m = params['m']
    dSdt = 0
    E, I, R, D = [], [], [], []
    idx = 1

    # Separate compartments for each strain
    for i in range(m):
        E.append(y[idx]); I.append(y[idx+1]); R.append(y[idx+2]); D.append(y[idx+3])
        idx += 4

    # Susceptible dynamics
    for i in range(m):
        dSdt -= params['beta'][i] * S * I[i] / params['N']

    dEdt, dIdt, dRdt, dDdt = [], [], [], []
    for i in range(m):
        dE = params['beta'][i] * S * I[i] / params['N'] - params['sigma'] * E[i]
        dI = params['sigma'] * E[i] - (params['gamma'][i] + params['mu'][i]) * I[i]
        dR = params['gamma'][i] * I[i]
        dD = params['mu'][i] * I[i]
        dEdt.append(dE); dIdt.append(dI); dRdt.append(dR); dDdt.append(dD)

    return [dSdt] + [val for tup in zip(dEdt, dIdt, dRdt, dDdt) for val in tup]


# ====== ROUTES ======
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            # Base parameters
            N = int(request.form["population"])
            m = int(request.form["strain_count"])
            incubation = float(request.form["incubation"])
            sigma = 1 / incubation
            t_max = int(request.form["days"])
            t = np.linspace(0, t_max, t_max)

            beta, gamma, mu = [], [], []
            E0, I0, R0, D0 = [], [], [], []

            # Collect strain-specific inputs
            for i in range(m):
                beta_val = float(request.form[f"beta_{i}"])
                infectious_period = float(request.form[f"infectious_period_{i}"])
                gamma_val = 1 / infectious_period
                mu_val = float(request.form[f"mu_{i}"])

                E0_val = int(request.form[f"E0_{i}"])
                I0_val = int(request.form[f"I0_{i}"])
                R0_val = int(request.form[f"R0_{i}"])
                D0_val = int(request.form[f"D0_{i}"])

                beta.append(beta_val)
                gamma.append(gamma_val)
                mu.append(mu_val)
                E0.append(E0_val)
                I0.append(I0_val)
                R0.append(R0_val)
                D0.append(D0_val)

            # Initial susceptible population
            S0 = N - sum(E0) - sum(I0) - sum(R0) - sum(D0)

            # Initial condition vector
            y0 = [S0]
            for i in range(m):
                y0 += [E0[i], I0[i], R0[i], D0[i]]

            params = {
                "N": N,
                "m": m,
                "beta": beta,
                "gamma": gamma,
                "mu": mu,
                "sigma": sigma
            }

            # Solve ODE
            sol = odeint(model, y0, t, args=(params,))

            # Extract results
            S = sol[:, 0]
            results = []
            idx = 1
            for i in range(m):
                Ei = sol[:, idx]
                Ii = sol[:, idx + 1]
                Ri = sol[:, idx + 2]
                Di = sol[:, idx + 3]
                results.append((Ei, Ii, Ri, Di))
                idx += 4

            # Plot
            plt.figure(figsize=(10, 6))
            plt.plot(t, S, label="Susceptible", linewidth=2)
            for i in range(m):
                plt.plot(t, results[i][1], label=f"Infectious Strain {i+1}")
                plt.plot(t, results[i][3], label=f"Deaths Strain {i+1}")
            plt.xlabel("Days")
            plt.ylabel("Population")
            plt.title("Multi-Strain Virus Spread Simulation")
            plt.legend()
            plt.grid()

            buf = io.BytesIO()
            plt.savefig(buf, format="png")
            buf.seek(0)
            graph_url = base64.b64encode(buf.getvalue()).decode()
            plt.close()

            # Summary stats
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

            summary = {
                "infected": int(final_infected),
                "deaths": int(final_deaths),
                "classification": classification
            }

            return render_template("index.html", graph_url=graph_url, summary=summary)

        except Exception as e:
            return f"Error: {str(e)}"

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
