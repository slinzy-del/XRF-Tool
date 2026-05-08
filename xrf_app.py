import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import plotly.express as px
import os
import json

# --- PAGE CONFIG ---
st.set_page_config(layout="wide", page_title="XRF AI Stratigraphy")

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

# 2. Process XRF
if uploaded_xrf:
    all_data = []
    for file in uploaded_xrf:
        temp_df = pd.read_csv(file)
        temp_df['Depth_Value'] = pd.to_numeric(temp_df['Sample'], errors='coerce')
        temp_df['Borehole_ID'] = temp_df['Batch'].astype(str) if 'Batch' in temp_df.columns else "Unknown"
        
        xrf_prefix = file.name.split('_')[0].upper()
        if xrf_prefix in gamma_data_map:
            g_log = gamma_data_map[xrf_prefix]
            temp_df['Gamma_API'] = np.interp(temp_df['Depth_Value'], g_log['Depth'], g_log['Gamma'])
        else:
            temp_df['Gamma_API'] = 0 
        all_data.append(temp_df)
    
    df_raw = pd.concat(all_data, ignore_index=True).dropna(subset=['Depth_Value']).copy()

    # --- GLOBAL NUMERIC CLEANING (Bye Bye <LOD) ---
    for col in df_raw.columns:
        if col not in ['Borehole_ID', 'Display_Label', 'Cluster_ID']:
            df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce').fillna(0)

    # Ratios
    if 'Zr' in df_raw.columns and 'Sr' in df_raw.columns:
        df_raw['Zr/Sr'] = (df_raw['Zr'] / df_raw['Sr'].replace(0, np.nan)).fillna(0)
    if 'K' in df_raw.columns and 'S' in df_raw.columns:
        df_raw['K/S'] = (df_raw['K'] / df_raw['S'].replace(0, np.nan)).fillna(0)
    if 'Ca' in df_raw.columns and 'Al' in df_raw.columns:
        df_raw['Ca/Al'] = (df_raw['Ca'] / df_raw['Al'].replace(0, np.nan)).fillna(0)

    exclude = ['READING', 'TYPE', 'TIME', 'SAMPLE', 'UNITS', 'SIGMA', 'CPS', 'UA', 'MODE', 'DURATION', 'USER', 'BATCH', 'HEAT', 'LOT', 'NOTE', 'BALANCE', 'DEPTH_VALUE', 'BOREHOLE_ID']
    final_features = sorted([c for c in df_raw.columns if not any(k in c.upper() for k in exclude) and "Unnamed" not in c])

    selected_features = []
    
    if app_mode == "Analysis & AI Training":
        selected_features = st.sidebar.multiselect("Select Inputs:", final_features, default=[])
        if len(selected_features) >= 2:
            num_clusters = st.sidebar.slider("Number of Clusters:", 2, 5, 3)
            X = df_raw[selected_features].apply(pd.to_numeric).fillna(0)
            X_scaled = StandardScaler().fit_transform(X)
            
            pca_obj = PCA(n_components=3, random_state=42)
            df_raw[['PC1', 'PC2', 'PC3']] = pca_obj.fit_transform(X_scaled)
            
            km = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
            df_raw['Cluster_ID'] = km.fit_predict(X_scaled).astype(str)
            
            label_map = {}
            for c in sorted(df_raw['Cluster_ID'].unique()):
                label_map[c] = st.sidebar.selectbox(f"Cluster {c}:", ["Unassigned", "Beaver Dam", "Choptank", "Calvert"], key=f"c_{c}")
            df_raw['Display_Label'] = df_raw['Cluster_ID'].map(lambda x: label_map[x] if label_map[x] != "Unassigned" else f"Cluster {x}")

            if st.sidebar.button("💾 Save Model for Master AI"):
                model_data = {"features": selected_features, "n_clusters": num_clusters, "labels": label_map}
                st.sidebar.download_button("Download master_model.json", json.dumps(model_data), "master_model.json")
    else:
        if os.path.exists("master_model.json"):
            with open("master_model.json", "r") as f: m = json.load(f)
            selected_features = m['features']
            X = df_raw[selected_features].apply(pd.to_numeric).fillna(0)
            X_scaled = StandardScaler().fit_transform(X)
            pca_obj = PCA(n_components=3, random_state=42)
            df_raw[['PC1', 'PC2', 'PC3']] = pca_obj.fit_transform(X_scaled)
            km = KMeans(n_clusters=m['n_clusters'], random_state=42, n_init=10)
            df_raw['Cluster_ID'] = km.fit_predict(X_scaled).astype(str)
            df_raw['Display_Label'] = df_raw['Cluster_ID'].map(m['labels'])
        else:
            st.sidebar.error("Missing 'master_model.json'")

    # --- SHOW TABS ONLY IF WE HAVE FEATURES ---
    if len(selected_features) >= 2:
        t1, t2, t3 = st.tabs(["3D Space", "Element Drivers", "Stratigraphy"])
        
        with t1:
            st.plotly_chart(px.scatter_3d(df_raw, x='PC1', y='PC2', z='PC3', color='Display_Label', color_discrete_map=COLOR_DISCRETE_MAP, height=700), use_container_width=True)
        
        with t2:
            st.write("### 🧬 Principle Component Loadings")
            pc_choice = st.radio("Select Axis:", ["PC1", "PC2", "PC3"], horizontal=True)
            loadings = pd.DataFrame(pca_obj.components_.T, columns=['PC1', 'PC2', 'PC3'], index=selected_features)
            fig_load = px.bar(loadings.reset_index(), x='index', y=pc_choice, color=pc_choice, color_continuous_scale='RdBu_r')
            st.plotly_chart(fig_load, use_container_width=True)
            
        with t3:
            fig = px.scatter(df_raw, x='Borehole_ID', y='Depth_Value', color='Display_Label', color_discrete_map=COLOR_DISCRETE_MAP, height=800)
            fig.update_yaxes(autorange="reversed", title="Depth (ft)")
            fig.update_layout(scattermode='overlay', xaxis_title="Sample ID (Batch)")
            st.plotly_chart(fig, use_container_width=True)