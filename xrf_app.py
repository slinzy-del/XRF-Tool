import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go

# --- PAGE CONFIG ---
st.set_page_config(layout="wide", page_title="XRF Stratigraphy & Exploration")

# --- COLOR MAP ---
COLOR_DISCRETE_MAP = {
    "Beaver Dam": "#636EFA",
    "Choptank": "#EF553B",
    "Calvert": "#00CC96",
    "Unassigned": "#7F7F7F",
    "Cluster 0": "#636EFA", "0": "#636EFA",
    "Cluster 1": "#EF553B", "1": "#EF553B",
    "Cluster 2": "#00CC96", "2": "#00CC96",
    "Cluster 3": "#AB63FA", "3": "#AB63FA"
}

# --- GEOCHEMICAL HELPER FUNCTIONS ---
def apply_clr(df_numeric, eps=1e-5):
    """Centered Log-Ratio (CLR) transformation for compositional XRF data."""
    X = df_numeric.copy().apply(pd.to_numeric, errors='coerce')
    X = X.replace(0, np.nan)
    min_pos = X.min().min()
    fill_val = eps if pd.isna(min_pos) or min_pos <= 0 else min_pos / 2.0
    X = X.fillna(fill_val)

    log_X = np.log(X)
    geom_mean = log_X.mean(axis=1)
    return log_X.sub(geom_mean, axis=0)

def create_pca_biplot(df, pca_obj, features, color_col):
    """Generates a 2D PCA Biplot with Loadings Arrows."""
    loadings = pca_obj.components_.T * np.sqrt(pca_obj.explained_variance_)

    fig = px.scatter(
        df, x='PC1', y='PC2', color=color_col,
        color_discrete_map=COLOR_DISCRETE_MAP,
        hover_data=['Sample_ID', 'Depth_Value'],
        title="PCA 2D Biplot (Scores + Element Vectors)",
        template="plotly_dark", height=650
    )

    for i, feature in enumerate(features):
        fig.add_shape(
            type='line', x0=0, y0=0,
            x1=loadings[i, 0] * 3, y1=loadings[i, 1] * 3,
            line=dict(color='yellow', width=2)
        )
        fig.add_annotation(
            x=loadings[i, 0] * 3.3, y=loadings[i, 1] * 3.3,
            text=feature, showarrow=False,
            font=dict(color='yellow', size=12)
        )

    fig.update_xaxes(title=f"PC1 ({pca_obj.explained_variance_ratio_[0]*100:.1f}% var)")
    fig.update_yaxes(title=f"PC2 ({pca_obj.explained_variance_ratio_[1]*100:.1f}% var)")
    return fig

# --- DATA LOADING ---
st.sidebar.title("🛠️ Project Controls")

col1, col2 = st.columns(2)
with col1:
    uploaded_files = st.file_uploader("Upload XRF CSV Files", type="csv", accept_multiple_files=True)
with col2:
    uploaded_txt_gamma = st.file_uploader("Upload Gamma TXT Files", type="txt", accept_multiple_files=True)

gamma_data_map = {}
if uploaded_txt_gamma:
    st.toast(f"📚 Syncing {len(uploaded_txt_gamma)} Gamma TXT log files...")
    for txt_file in uploaded_txt_gamma:
        try:
            g_df = pd.read_csv(txt_file, sep=r'\s+', engine='python', skiprows=1, header=None, usecols=[0, 1], names=['Depth', 'Gamma'])
            g_df = g_df[g_df['Gamma'] > -500].dropna().copy()
            prefix = txt_file.name.split('.')[0].upper()
            gamma_data_map[prefix] = g_df
        except Exception as e:
            st.error(f"Error parsing TXT Gamma file {txt_file.name}: {e}")

if uploaded_files:
    all_data = []
    for file in uploaded_files:
        temp_df = pd.read_csv(file)
        temp_df.columns = [c.strip() for c in temp_df.columns]
        
        sample_col = next((c for c in temp_df.columns if c.upper() in ['SAMPLE', 'DEPTH (IN)', 'DEPTH', 'DEPTH_FT']), 'Sample')
        
        temp_df = temp_df.assign(
            Source_File=str(file.name),
            Sample_ID=temp_df[sample_col].astype(str),
            Depth_Value=pd.to_numeric(temp_df[sample_col], errors='coerce')
        )
        
        # Safe helper function to get numeric series across alternative header names
        def get_elem_series(possible_names):
            for name in possible_names:
                found = next((c for c in temp_df.columns if c.upper() == name.upper()), None)
                if found is not None:
                    return pd.to_numeric(temp_df[found], errors='coerce')
            return None

        # Cross-Formation Ratios
        K_val = get_elem_series(['K', 'Potassium'])
        Ca_val = get_elem_series(['Ca', 'Calcium'])
        Fe_val = get_elem_series(['Fe', 'Iron'])
        Cr_val = get_elem_series(['Cr', 'Chromium'])
        Zr_val = get_elem_series(['Zr', 'Zirconium'])
        Sr_val = get_elem_series(['Sr', 'Strontium'])

        if K_val is not None and Zr_val is not None:
            temp_df['Ratio_K_Zr'] = (K_val / Zr_val.replace(0, np.nan)).fillna(0)
        if Ca_val is not None and Fe_val is not None:
            temp_df['Ratio_Ca_Fe'] = (Ca_val / Fe_val.replace(0, np.nan)).fillna(0)
        if Sr_val is not None and Cr_val is not None:
            temp_df['Ratio_Sr_Cr'] = (Sr_val / Cr_val.replace(0, np.nan)).fillna(0)

        # Gamma Interpolation
        file_prefix = file.name.split('.')[0].upper()
        matched_key = next((k for k in gamma_data_map if k in file_prefix or file_prefix in k), None)
        
        if matched_key:
            g_log = gamma_data_map[matched_key]
            temp_df['Gamma_API'] = np.interp(temp_df['Depth_Value'], g_log['Depth'], g_log['Gamma'])
        else:
            temp_df['Gamma_API'] = 0.0

        all_data.append(temp_df)
    
    df_raw = pd.concat(all_data, ignore_index=True).dropna(subset=['Depth_Value']).copy()

    # --- CORE DATA PRUNING ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("✂️ Core Data Pruning")
    manual_exclude_input = st.sidebar.text_input("Exclude Depths (e.g., 10.5, 40-45):", value="")
    if manual_exclude_input.strip():
        try:
            items = [item.strip() for item in manual_exclude_input.split(",")]
            for item in items:
                if "-" in item:
                    start_str, end_str = item.split("-")
                    df_raw = df_raw[~((df_raw['Depth_Value'] >= float(start_str)) & (df_raw['Depth_Value'] <= float(end_str)))]
                else:
                    df_raw = df_raw[df_raw['Depth_Value'] != float(item)]
        except ValueError:
            st.sidebar.error("
