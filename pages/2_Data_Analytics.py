import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from dashboard_core import inject_global_styles, load_plot_path, load_processed_data, render_app_shell


st.set_page_config(
    page_title="Data Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

sns.set_theme(style="whitegrid")


def main():
    inject_global_styles()
    render_app_shell()

    st.markdown('<div class="page-title">Data Analytics</div>', unsafe_allow_html=True)

    df = load_processed_data()

    top1, top2, top3, top4 = st.columns(4)
    metrics = [
        ("Records", f"{len(df):,}"),
        ("Avg Magnitude", f"{df['magnitude'].mean():.2f}"),
        ("Avg Depth", f"{df['depth'].mean():.1f} km"),
        ("Latest Event", df["time"].max().strftime("%Y-%m-%d")),
    ]
    for col, (label, value) in zip([top1, top2, top3, top4], metrics):
        col.markdown(
            f"""
            <div class="glass-card">
                <div class="card-label">{label}</div>
                <div class="metric-value">{value}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        risk_counts = pd.DataFrame(
            {
                "Risk Level": ["Low", "Medium", "High"],
                "Count": [
                    int(df["risk_level"].eq("Low").sum()),
                    int(df["risk_level"].eq("Medium").sum()),
                    int(df["risk_level"].eq("High").sum()),
                ],
            }
        )
        fig, ax = plt.subplots(figsize=(6.0, 4.2))
        sns.barplot(
            data=risk_counts,
            x="Risk Level",
            y="Count",
            hue="Risk Level",
            dodge=False,
            palette={"Low": "#167c59", "Medium": "#ca6702", "High": "#a62c2b"},
            ax=ax,
            legend=False,
        )
        ax.set_title("Prediction Band Distribution", fontsize=13, weight="bold")
        ax.set_xlabel("")
        sns.despine(left=True, bottom=True)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        sample_df = df.sample(min(3000, len(df)), random_state=42)
        fig, ax = plt.subplots(figsize=(6.0, 4.2))
        sns.scatterplot(
            data=sample_df,
            x="longitude",
            y="latitude",
            hue="risk_level",
            palette={"Low": "#167c59", "Medium": "#ca6702", "High": "#a62c2b"},
            s=32,
            alpha=0.68,
            ax=ax,
        )
        ax.set_title("Geographic Event Map", fontsize=13, weight="bold")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.legend(title="", frameon=False)
        sns.despine()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    preview = df.sort_values("time", ascending=False)[
        ["time", "latitude", "longitude", "depth", "magnitude", "risk_level"]
    ].head(15).copy()
    preview["time"] = preview["time"].dt.strftime("%Y-%m-%d %H:%M")
    st.dataframe(preview, use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

    gallery1, gallery2 = st.columns(2)
    plots = [
        ("Magnitude Distribution", load_plot_path("magnitude_distribution.png")),
        ("Earthquake Frequency Over Time", load_plot_path("earthquake_frequency_over_time.png")),
        ("Depth vs Magnitude", load_plot_path("depth_vs_magnitude.png")),
        ("Correlation Heatmap", load_plot_path("correlation_heatmap.png")),
    ]
    for idx, (title, path) in enumerate(plots):
        with [gallery1, gallery2][idx % 2]:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            if path.exists():
                st.image(str(path), caption=title, use_container_width=True)
            else:
                st.info(f"{title} not available yet.")
            st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
