import streamlit as st
from dashboard_core import inject_global_styles

# Page config
st.set_page_config(
    page_title="Earthquake Prediction",
    page_icon="🌍",
    layout="wide"
)

# Apply your global styles (if needed)
inject_global_styles()

# ✅ FIX SIDEBAR VISIBILITY (IMPORTANT)
st.markdown("""
<style>

/* Sidebar background */
section[data-testid="stSidebar"] {
    background-color: #0e1117;
}

/* Sidebar text */
section[data-testid="stSidebar"] * {
    color: white !important;
}

/* Labels and headings */
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] span {
    color: white !important;
}

/* Fix radio & select text */
section[data-testid="stSidebar"] .stRadio div,
section[data-testid="stSidebar"] .stSelectbox div {
    color: white !important;
}

</style>
""", unsafe_allow_html=True)

# Sidebar title
st.sidebar.title("🌍 Earthquake Prediction")

# -------------------------------
# 🌟 HOME PAGE UI
# -------------------------------

st.markdown("""
<style>
.hero {
    padding: 2rem;
    border-radius: 15px;
    background: linear-gradient(135deg, #1f4e5f, #2e8b8b);
    color: white;
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <h1>Seismic Analysis and Prediction Dashboard</h1>
    <p>Analyze earthquake data and predict risks using machine learning models.</p>
</div>
""", unsafe_allow_html=True)

st.write("### Welcome 👋")

st.markdown("""
- ⚙️ Training Studio  
- 🔮 Prediction Studio  
- 📊 Data Analytics  
- 🧠 Model Performance  
""")