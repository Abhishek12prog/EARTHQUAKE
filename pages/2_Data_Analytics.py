import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from dashboard_core import inject_global_styles, load_plot_path, load_processed_data

st.set_page_config(
    page_title="Data Analytics",
    page_icon="📊",
    layout="wide",
)

sns.set_theme(style="whitegrid")

# Apply styles
inject_global_styles()

st.title("📊 Data Analytics")

# Load data
df = load_processed_data()

# 🔹 TOP METRICS
col1, col2, col3, col4 = st.columns(4)

col1.metric("Records", f"{len(df):,}")
col2.metric("Avg Magnitude", f"{df['magnitude'].mean():.2f}")
col3.metric("Avg Depth", f"{df['depth'].mean():.1f} km")
col4.metric("Latest Event", df["time"].max().strftime("%Y-%m-%d"))

# 🔹 CHARTS
c1, c2 = st.columns(2)

# Risk Distribution
with c1:
    st.subheader("Risk Distribution")

    risk_counts = df["risk_level"].value_counts().reset_index()
    risk_counts.columns = ["Risk Level", "Count"]

    fig, ax = plt.subplots()
    sns.barplot(
        data=risk_counts,
        x="Risk Level",
        y="Count",
        palette={"Low": "green", "Medium": "orange", "High": "red"},
        ax=ax
    )
    st.pyplot(fig)

# Geo Scatter
with c2:
    st.subheader("Geographic Map")

    sample_df = df.sample(min(3000, len(df)))

    fig, ax = plt.subplots()
    sns.scatterplot(
        data=sample_df,
        x="longitude",
        y="latitude",
        hue="risk_level",
        palette={"Low": "green", "Medium": "orange", "High": "red"},
        ax=ax
    )
    st.pyplot(fig)

# 🔹 DATA TABLE
st.subheader("Recent Records")

preview = df.sort_values("time", ascending=False).head(15)
st.dataframe(preview)

# 🔹 IMAGE GALLERY
st.subheader("EDA Visualizations")

plots = [
    ("Magnitude Distribution", load_plot_path("magnitude_distribution.png")),
    ("Frequency Over Time", load_plot_path("earthquake_frequency_over_time.png")),
    ("Depth vs Magnitude", load_plot_path("depth_vs_magnitude.png")),
    ("Correlation Heatmap", load_plot_path("correlation_heatmap.png")),
]

cols = st.columns(2)

for i, (title, path) in enumerate(plots):
    with cols[i % 2]:
        if path.exists():
            st.image(str(path), caption=title)
        else:
            st.info(f"{title} not available")