import streamlit as st

st.set_page_config(
    page_title="Earthquake Prediction",
    page_icon="🌍",
    layout="wide"
)

# Sidebar title
st.sidebar.title("🌍 Earthquake Prediction")

# Home Page
st.markdown("""
<style>
.hero {
    padding: 2rem;
    border-radius: 15px;
    background: linear-gradient(135deg, #1f4e5f, #2e8b8b);
    color: white;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
<h1>Seismic Analysis and Prediction Dashboard</h1>
<p>Analyze earthquake data and predict risks using machine learning.</p>
</div>
""", unsafe_allow_html=True)

st.write("👉 Use the sidebar to navigate between modules.")