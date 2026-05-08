import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import plotly.express as px
import os

# --- PAGE CONFIG ---
st.set_page_config(layout="wide", page_title="XRF AI Tool")

# --- MASTER COLOR MAP ---
COLOR_DISCRETE_MAP = {
    "Beaver Dam": "#1f77b4", "Choptank": "#d62728", "Calvert": "#2ca02c",
    "Unassigned": "#7F7F7F",
    "Cluster 0": "#636EFA", "Cluster 1": "#EF553B", "Cluster 2": "#00CC96",
    "Cluster 3": "#AB63FA", "Cluster 4": "#FFA15A"
}

st.sidebar.title("🛠️ Project Controls")
app_mode = st.sidebar.radio("Analysis Mode", ["Analysis & AI Training", "Run Master AI"])

# --- DATA LOADING ---
col1, col2 = st.columns(2)
with col1:
    uploaded_xrf = st.file_uploader("Upload XRF CSV Files", type="csv", accept_multiple_files=True)
with col2:
    uploaded_gamma = st.file_uploader("Upload Gamma TXT Files", type="txt", accept_multiple_files=True)

# 1. Process Gamma
gamma_data_map = {}
if uploaded_gamma:
    for gf in uploaded_gamma:
        try:
            g_df = pd.read_csv(gf, sep=r'\s+', engine='python', skiprows=1, header=None, usecols=[0, 1], names=['Depth', 'Gamma'])
            g_df = g_df[g_df['Gamma'] > -500].copy()
            prefix = gf.name.split('_')[0].upper()
            gamma_data_map[prefix] = g_df
        except: st.error(f"Error reading Gamma: {gf.name}")

# 2. Process XRF & Sync
if uploaded_xrf:
    all_data = []
    for file in uploaded_xrf:
        temp_df = pd.read_csv(file)
        
        # Identify Depth and Sample ID (Batch)
        temp_df['Depth_Value'] = pd.to_numeric(temp_df['Sample'], errors='coerce')
        
        if 'Batch' in temp_df.columns:
            temp_df['Borehole_ID'] = temp_df['Batch'].astype(str)
        else:
            temp_df['Borehole_ID'] = "Unknown"

        xrf_prefix = file.name.split('_')[0].upper()
        if xrf_prefix in gamma_data_map:
            g_log = gamma_data_map[xrf_prefix]
            temp_df['Gamma_API'] = np.interp(temp_df['Depth_Value'], g_log['Depth'], g_log['Gamma'])
        else:
            temp_df['Gamma_API'] = 0 
            
        all_data.append(temp_df)
    
    df_raw = pd.concat(all_data, ignore_index=True).dropna(subset=['Depth_Value']).copy()

    # Numeric Force for Ratios
    core_elements = ['Zr', 'Sr', 'K', 'S', 'Ca', 'Al']
    for elem in core_elements:
        if elem in df_raw.columns:
            df_raw[elem] = pd.to_numeric(df_raw[elem], errors='coerce').fillna(0)

    # --- RATIO ENGINEERING ---
    if 'Zr' in df_raw.columns and 'Sr' in df_raw.columns:
        df_raw['Zr/Sr'] = (df_raw['Zr'] / df_raw['Sr'].replace(0, np.nan)).fillna(0).replace([np.inf, -np.inf], 0)
    if 'K' in df_raw.columns and 'S' in df_raw.columns:
        df_raw['K/S'] = (df_raw['K'] / df_raw['S'].replace(0, np.nan)).fillna(0).replace([np.inf, -np.inf], 0)
    if 'Ca' in df_raw.columns and 'Al' in df_raw.columns:
        df_raw['Ca/Al'] = (df_raw['Ca'] / df_raw['Al'].replace(0, np.nan)).fillna(0).replace([np.inf, -np.inf], 0)

    # --- UPDATED DROPDOWN FILTERING ---
    # Specifically removing Reading, Type, Batch, and anything with "Sigma"
    exclude_keywords = [
        'READING', 'TYPE', 'TIME', 'SAMPLE', 'UNITS', 'SIGMA', 'CPS', 'UA', 
        'MODE', 'DURATION', 'MAIN', 'LOW', 'HIGH', 'LIGHT', 'USER', 'BATCH', 
        'HEAT', 'LOT', 'NOTE', 'BALANCE', 'BAL', 'BOREHOLE_ID', 'DEPTH_VALUE', 'UNNAMED'
    ]
    
    clean_features = [
        c for c in df_raw.columns 
        if not any(k in c.upper() for k in exclude_keywords)
    ]
    
    # Add back Gamma and Ratios specifically
    final_features = sorted(list(set(clean_features + ['Gamma_API', 'Zr/Sr', 'K/S', 'Ca/Al'])))
    
    # Filter to ensure only columns that actually exist in the dataframe stay in the list
    final_features = [f for f in final_features if f in df_raw.columns]

    for col in final_features:
        df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce').fillna(0)

    # Sidebar Selection - START EMPTY
    selected_features = st.sidebar.multiselect("Select Inputs for AI:", final_features, default=[])

    if len(selected_features) >= 2:
        X_scaled = StandardScaler().fit_transform(df_raw[selected_features])
        pca_obj = PCA(n_components=3, random_state=42)
        df_raw[['PC1', 'PC2', 'PC3']] = pca_obj.fit_transform(X_scaled)

        if app_mode == "Analysis & AI Training":
            num_clusters = st.sidebar.slider("Number of Clusters:", 2, 5, 3)
            df_raw['Cluster_ID'] = KMeans(n_clusters=num_clusters, random_state=42, n_init=10).fit_predict(X_scaled).astype(str)
            
            label_map = {}
            for c in sorted(df_raw['Cluster_ID'].unique()):
                form_name = st.sidebar.selectbox(f"Cluster {c}:", ["Unassigned", "Beaver Dam", "Choptank", "Calvert"], key=f"l_{c}")
                label_map[c] = form_name if form_name != "Unassigned" else f"Cluster {c}"
            df_raw['Display_Label'] = df_raw['Cluster_ID'].map(label_map)
        else:
            df_raw['Display_Label'] = "Unassigned"

        tab1, tab2, tab3 = st.tabs(["3D Space", "Element Drivers", "Stratigraphy"])
        
        with tab1:
            st.plotly_chart(px.scatter_3d(df_raw, x='PC1', y='PC2', z='PC3', color='Display_Label', color_discrete_map=COLOR_DISCRETE_MAP, height=700), use_container_width=True)
        
        with tab2:
            pc_choice = st.radio("Select Axis:", ["PC1", "PC2", "PC3"], horizontal=True)
            loadings = pd.DataFrame(pca_obj.components_.T, columns=['PC1', 'PC2', 'PC3'], index=selected_features)
            st.plotly_chart(px.bar(loadings.reset_index(), x='index', y=pc_choice, color=pc_choice, color_continuous_scale='RdBu_r'), use_container_width=True)
        
        with tab3:
            st.write("### 🕳️ Vertical Stratigraphy")
            fig_strat = px.scatter(
                df_raw, 
                x='Borehole_ID', 
                y='Depth_Value', 
                color='Display_Label', 
                color_discrete_map=COLOR_DISCRETE_MAP,
                hover_data=['Sample'],
                height=850
            )
            fig_strat.update_xaxes(type='category', title="Sample ID (Batch)")
            fig_strat.update_yaxes(autorange="reversed", title="Depth (ft)")
            fig_strat.update_layout(scattermode='overlay')
            fig_strat.update_traces(marker=dict(size=16, line=dict(width=1, color='white')))
            st.plotly_chart(fig_strat, use_container_width=True)
            st.download_button("💾 Download Results", df_raw.to_csv(index=False), "xrf_output.csv")
    else:
        st.info("Please select at least 2 elements or ratios from the sidebar to begin.")