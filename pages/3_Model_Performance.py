import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from dashboard_core import inject_global_styles, load_model_results

st.set_page_config(
    page_title="Model Performance",
    page_icon="🧠",
    layout="wide",
)

sns.set_theme(style="whitegrid")

# Apply styles
inject_global_styles()

st.title("🧠 Model Performance")

# Load results
results_df = load_model_results()
best_row = results_df.iloc[0]

# 🔹 SUMMARY METRICS
col1, col2, col3 = st.columns(3)

col1.metric("Best Model", best_row["Model"])
col2.metric("Best F1 Score", f"{best_row['F1 Score']:.3f}")
col3.metric("Best Accuracy", f"{best_row['Accuracy']:.3f}")

# 🔹 CHART + TABLE
left, right = st.columns([1.1, 0.9])

# Chart
with left:
    st.subheader("Model Comparison")

    plot_df = results_df.melt(
        id_vars="Model",
        var_name="Metric",
        value_name="Score"
    )

    fig, ax = plt.subplots()
    sns.barplot(
        data=plot_df,
        x="Metric",
        y="Score",
        hue="Model",
        palette="crest",
        ax=ax
    )

    ax.set_ylim(0, 1)
    st.pyplot(fig)

# Table
with right:
    st.subheader("Performance Table")

    display_df = results_df.copy()
    for col in ["Accuracy", "Precision", "Recall", "F1 Score"]:
        display_df[col] = display_df[col].map(lambda x: f"{x:.3f}")

    st.dataframe(display_df)