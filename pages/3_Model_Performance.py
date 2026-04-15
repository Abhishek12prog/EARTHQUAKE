import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from dashboard_core import inject_global_styles, load_model_results, render_app_shell


st.set_page_config(
    page_title="Model Performance",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

sns.set_theme(style="whitegrid")


def main():
    inject_global_styles()
    render_app_shell()

    st.markdown('<div class="page-title">Model Performance</div>', unsafe_allow_html=True)

    results_df = load_model_results()
    best_row = results_df.iloc[0]

    a, b, c = st.columns(3)
    summary_cards = [
        ("Best Model", best_row["Model"]),
        ("Best F1 Score", f"{best_row['F1 Score']:.3f}"),
        ("Best Accuracy", f"{best_row['Accuracy']:.3f}"),
    ]
    for col, (label, value) in zip([a, b, c], summary_cards):
        col.markdown(
            f"""
            <div class="glass-card">
                <div class="card-label">{label}</div>
                <div class="metric-value">{value}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    left, right = st.columns([1.1, 0.9])

    with left:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        plot_df = results_df.melt(id_vars="Model", var_name="Metric", value_name="Score")
        fig, ax = plt.subplots(figsize=(7.2, 4.4))
        sns.barplot(data=plot_df, x="Metric", y="Score", hue="Model", palette="crest", ax=ax)
        ax.set_ylim(0, 1)
        ax.set_xlabel("")
        ax.set_title("Classifier Benchmark", fontsize=13, weight="bold")
        ax.legend(title="", frameon=False)
        sns.despine(left=True, bottom=True)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        ranking = results_df[["Model", "Accuracy", "Precision", "Recall", "F1 Score"]].copy()
        ranking[["Accuracy", "Precision", "Recall", "F1 Score"]] = ranking[
            ["Accuracy", "Precision", "Recall", "F1 Score"]
        ].applymap(lambda value: f"{value:.3f}")
        st.dataframe(ranking, use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
