import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Probability Insight Lab", layout="wide")

st.title(" Probability Insight Lab")
st.markdown("### Binomial vs Poisson — Structural, Uncertainty & Stability Analysis")

# ================= INPUT PANEL =================

st.sidebar.header(" Problem Parameters")

n = st.sidebar.slider("Binomial Trials (n)", 10, 500, 50)
p = st.sidebar.slider("Binomial Success Probability (p)", 0.01, 0.99, 0.1)
lam = st.sidebar.slider("Poisson Rate (λ)", 0.1, 100.0, float(n*p))

st.sidebar.markdown("---")
st.sidebar.markdown("### Enable Analysis Modules")

modules = st.sidebar.multiselect(
    "Select Modules",
    [
        "Distribution Shape",
        "Mean-Variance Geometry",
        "Uncertainty Landscape",
        "Stability vs Dispersion",
        "Sensitivity Heatmap",
        "Insight Engine"
    ],
    default=["Distribution Shape", "Mean-Variance Geometry"]
)

# ================= CORE STATISTICS ENGINE =================

bin_mean = n * p
bin_var = n * p * (1 - p)

pois_mean = lam
pois_var = lam

def metrics(mean, var):
    cv = np.sqrt(var) / mean
    stability = mean / var
    return cv, stability

bin_cv, bin_stability = metrics(bin_mean, bin_var)
pois_cv, pois_stability = metrics(pois_mean, pois_var)

# ================= DISPLAY CORE STATS =================

st.markdown("## 📌 Core Statistical Summary")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Binomial")
    st.write(f"Mean = {bin_mean:.3f}")
    st.write(f"Variance = {bin_var:.3f}")
    st.write(f"Coefficient of Variation = {bin_cv:.3f}")
    st.write(f"Stability Score = {bin_stability:.3f}")

with col2:
    st.markdown("### Poisson")
    st.write(f"Mean = {pois_mean:.3f}")
    st.write(f"Variance = {pois_var:.3f}")
    st.write(f"Coefficient of Variation = {pois_cv:.3f}")
    st.write(f"Stability Score = {pois_stability:.3f}")

# ================= GRAPH 1: DISTRIBUTION SHAPE =================

if "Distribution Shape" in modules:
    st.markdown("##  Distribution Shape (Reality View)")

    bin_samples = np.random.binomial(n, p, 6000)
    pois_samples = np.random.poisson(lam, 6000)

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    ax[0].hist(bin_samples, bins=30, alpha=0.7, color="royalblue")
    ax[0].axvline(bin_mean, color="red", linestyle="--")
    ax[0].set_title("Binomial Outcome Distribution")

    ax[1].hist(pois_samples, bins=30, alpha=0.7, color="orange")
    ax[1].axvline(pois_mean, color="red", linestyle="--")
    ax[1].set_title("Poisson Outcome Distribution")

    st.pyplot(fig)

# ================= GRAPH 2: MEAN–VARIANCE GEOMETRY =================

if "Mean-Variance Geometry" in modules:
    st.markdown("##  Mean–Variance Geometry (Structural View)")

    p_vals = np.linspace(0.01, 0.99, 200)
    bin_means = n * p_vals
    bin_vars = n * p_vals * (1 - p_vals)

    lam_vals = np.linspace(0.1, 100, 200)

    fig, ax = plt.subplots()
    ax.plot(bin_means, bin_vars, label="Binomial Curve")
    ax.plot(lam_vals, lam_vals, linestyle="--", label="Poisson Line (Var = Mean)")
    ax.scatter(bin_mean, bin_var, color="blue", s=60)
    ax.scatter(pois_mean, pois_var, color="red", s=60)

    ax.set_xlabel("Mean")
    ax.set_ylabel("Variance")
    ax.set_title("Geometry of Mean vs Variance")
    ax.legend()
    ax.grid(True)

    st.pyplot(fig)

# ================= GRAPH 3: UNCERTAINTY LANDSCAPE =================

if "Uncertainty Landscape" in modules:
    st.markdown("##  Uncertainty Landscape (Risk View)")

    mean_range = np.linspace(1, 100, 200)
    bin_cv_curve = np.sqrt(mean_range*(1-p)) / mean_range
    pois_cv_curve = 1 / np.sqrt(mean_range)

    fig, ax = plt.subplots()
    ax.plot(mean_range, bin_cv_curve, label="Binomial CV Curve")
    ax.plot(mean_range, pois_cv_curve, linestyle="--", label="Poisson CV Curve")
    ax.scatter(bin_mean, bin_cv, color="blue")
    ax.scatter(pois_mean, pois_cv, color="red")

    ax.set_xlabel("Mean")
    ax.set_ylabel("Coefficient of Variation (CV)")
    ax.set_title("Uncertainty vs Scale")
    ax.legend()
    ax.grid(True)

    st.pyplot(fig)

# ================= GRAPH 4: STABILITY vs DISPERSION =================

if "Stability vs Dispersion" in modules:
    st.markdown("##  Stability vs Dispersion (Control View)")

    fig, ax = plt.subplots()
    ax.scatter(bin_var, bin_stability, color="blue", label="Binomial", s=80)
    ax.scatter(pois_var, pois_stability, color="red", label="Poisson", s=80)

    ax.set_xlabel("Variance (Dispersion)")
    ax.set_ylabel("Stability Score")
    ax.set_title("Predictability of Systems")
    ax.legend()
    ax.grid(True)

    st.pyplot(fig)

# ================= GRAPH 5: SENSITIVITY HEATMAP =================

if "Sensitivity Heatmap" in modules:
    st.markdown("##  Parameter Sensitivity Heatmap (Behavior View)")

    p_grid = np.linspace(0.01, 0.99, 60)
    n_grid = np.linspace(10, 400, 60)

    heat = np.zeros((len(n_grid), len(p_grid)))

    for i, ni in enumerate(n_grid):
        for j, pj in enumerate(p_grid):
            mean = ni * pj
            var = ni * pj * (1 - pj)
            heat[i, j] = var / mean if mean != 0 else 0

    fig, ax = plt.subplots()
    im = ax.imshow(heat, aspect="auto", cmap="plasma")
    ax.set_title("Binomial Uncertainty Sensitivity")
    fig.colorbar(im, label="Relative Uncertainty")

    st.pyplot(fig)

# ================= INSIGHT ENGINE =================

if "Insight Engine" in modules:
    st.markdown("##  Insight Engine (Mathematical Interpretation)")

    insights = []

    if bin_var < bin_mean:
        insights.append("Binomial shows lower dispersion than Poisson for given parameters.")

    if bin_cv < pois_cv:
        insights.append("Binomial process is more stable than Poisson.")

    if n > 50 and p < 0.1:
        insights.append("System lies in rare-event regime where Binomial approximates Poisson.")

    if p > 0.5:
        insights.append("System operates in high-success probability regime.")

    if bin_stability > pois_stability:
        insights.append("Binomial outcomes are more predictable than Poisson outcomes.")

    if insights:
        for i in insights:
            st.write("•", i)
    else:
        st.write("No special probabilistic regime detected.")

st.markdown("---")
st.markdown("⚙️ Probability Insight Lab — exploring uncertainty, stability, and structure of distributions.")




