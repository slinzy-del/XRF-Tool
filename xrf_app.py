import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import plotly.express as px
import pickle
import os
import datetime

# --- PAGE CONFIG ---
st.set_page_config(layout="wide", page_title="AI XRF + Gamma Stratigraphy")

# --- ORIGINAL PLOTLY COLOR MAP ---
COLOR_DISCRETE_MAP = {
    "Beaver Dam": "#636EFA",  # Plotly Blue
    "Choptank": "#EF553B",    # Plotly Red
    "Calvert": "#00CC96",     # Plotly Green
    "Unassigned": "#7F7F7F",  # Gray
    "Cluster 0": "#636EFA", "0": "#636EFA",
    "Cluster 1": "#EF553B", "1": "#EF553B",
    "Cluster 2": "#00CC96", "2": "#00CC96",
    "Cluster 3": "#AB63FA", "3": "#AB63FA" # Plotly Purple
}

st.sidebar.title("🛠️ Project Controls")

model_path = "master_xrf_brain.pkl"
if os.path.exists(model_path):
    mod_time = datetime.datetime.fromtimestamp(os.path.getmtime(model_path)).strftime('%Y-%m-%d %H:%M')
    st.sidebar.success(f"✅ AI Model Loaded ({mod_time})")
else:
    st.sidebar.warning("⚠️ No AI Model found.")

app_mode = st.sidebar.radio("Analysis Mode", ["Analysis & AI Training", "Run Master AI (Auto-Label)"])

# --- DATA LOADING ---
col1, col2 = st.columns(2)
with col1:
    uploaded_files = st.file_uploader("Upload XRF CSV Files", type="csv", accept_multiple_files=True)
with col2:
    uploaded_txt_gamma = st.file_uploader("Upload Gamma TXT Files", type="txt", accept_multiple_files=True)

# Process TXT Gamma logs
gamma_data_map = {}
if uploaded_txt_gamma:
    for txt_file in uploaded_txt_gamma:
        try:
            # Read space or tab-separated text columns (assuming Column 0 = Depth, Column 1 = Gamma)
            g_df = pd.read_csv(txt_file, sep=r'\s+', engine='python', skiprows=1, header=None, usecols=[0, 1], names=['Depth', 'Gamma'])
            
            # Clean out common invalid tool errors (like massive negative placeholders)
            g_df = g_df[g_df['Gamma'] > -500].dropna().copy()
            
            # Create a lookup key based on the filename prefix (e.g., Ld32-18)
            prefix = txt_file.name.split('.')[0].upper()
            gamma_data_map[prefix] = g_df
            st.toast(f"✅ Loaded Gamma TXT for {txt_file.name}")
        except Exception as e:
            st.error(f"Error parsing TXT Gamma file {txt_file.name}: {e}")

if uploaded_files:
    all_data = []
    for file in uploaded_files:
        temp_df = pd.read_csv(file)
        temp_df = temp_df.assign(
            Source_File=str(file.name),
            Sample_ID=temp_df['Sample'].astype(str),
            Depth_Value=pd.to_numeric(temp_df['Sample'], errors='coerce')
        )
        
        # Match XRF depth values to the continuous Gamma TXT curve
        file_prefix = file.name.split('.')[0].upper()
        matched_key = next((k for k in gamma_data_map if k in file_prefix or file_prefix in k), None)
        
        if matched_key:
            g_log = gamma_data_map[matched_key]
            # Perform linear interpolation to link continuous Gamma measurements directly to XRF intervals
            temp_df['Gamma_API'] = np.interp(temp_df['Depth_Value'], g_log['Depth'], g_log['Gamma'])
        else:
            temp_df['Gamma_API'] = 0.0  # Safe default if no matching log is uploaded

        all_data.append(temp_df)
    
    df_raw = pd.concat(all_data, ignore_index=True).dropna(subset=['Depth_Value']).copy()

    # ---------------------------------------------------------
    # ✂️ CORE DATA PRUNING
    # ---------------------------------------------------------
    st.sidebar.markdown("---")
    st.sidebar.subheader("✂️ Core Data Pruning")
    
    trim_first_reading = st.sidebar.checkbox("Remove first reading from every core", value=False)
    if trim_first_reading:
        df_raw = df_raw.sort_values(by=['Source_File', 'Depth_Value'])
        df_raw = df_raw.groupby('Source_File', as_index=False).tail(-1).reset_index(drop=True)

    manual_exclude_input = st.sidebar.text_input("Exclude Depths (e.g., 10.5, 40-45):", value="")
    if manual_exclude_input.strip():
        try:
            items = [item.strip() for item in manual_exclude_input.split(",")]
            for item in items:
                if "-" in item:
                    start_str, end_str = item.split("-")
                    start_val = float(start_str.strip())
                    end_val = float(end_str.strip())
                    df_raw = df_raw[~((df_raw['Depth_Value'] >= start_val) & (df_raw['Depth_Value'] <= end_val))]
                else:
                    exact_val = float(item)
                    df_raw = df_raw[df_raw['Depth_Value'] != exact_val]
        except ValueError:
            st.sidebar.error("⚠️ Check format! Examples: 12.4 or 40-45")
    # ---------------------------------------------------------

    # Explicit list of words to remove from element dropdown lists
    meta = ['Reading', 'Type', 'Time', 'Sample', 'Units', 'Sigma', 'CPS', 'Mode', 'Duration', 
            'Main', 'Low', 'High', 'Light', 'User', 'Batch', 'Heat', 'Lot', 'Note', 'Balance', 'Bal',
            'Source_File', 'Sample_ID', 'Depth_Value', 'PC1', 'PC2', 'PC3', 'Cluster_ID', 'Display_Label']
    
    elements = [c for c in df_raw.columns if not any(k.upper() in c.upper() for k in meta) 
                and "2-Sigma" not in c and "Unnamed" not in c]
    
    # 🛡️ SAFETY CHECK: Only append Gamma track to features if a valid TXT file was uploaded
    if len(gamma_data_map) > 0 and 'Gamma_API' in df_raw.columns:
        elements = sorted(list(set(elements + ['Gamma_API'])))
    elif 'Gamma_API' in elements:
        elements.remove('Gamma_API')

    # Simplified text prompt above the multi-select input box
    st.sidebar.subheader("Select Features:")
    selected_elements = st.sidebar.multiselect(
        "",  # Kept empty since subheader handles the title cleanly now
        elements, 
        default=[e for e in ['Al', 'Si', 'K', 'Ca', 'Fe', 'Zr', 'Gamma_API'] if e in elements]
    )

    if len(selected_elements) >= 3:
        X_num = df_raw[selected_elements].apply(pd.to_numeric, errors='coerce').fillna(0).copy()
        
        st.sidebar.subheader("🧼 Data Cleaning")
        outlier_sigma = st.sidebar.slider("Outlier Scrub (Z-Score):", 1.0, 15.0, 10.0)
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_num)
        
        pca_obj = PCA(n_components=3, random_state=42)
        pca_scores = pca_obj.fit_transform(X_scaled)
        
        mask = np.abs((pca_scores[:,0] - np.mean(pca_scores[:,0])) / np.std(pca_scores[:,0])) < outlier_sigma
        df = df_raw.loc[mask].copy() 
        final_pca_scores = pca_scores[mask]
        df[['PC1', 'PC2', 'PC3']] = final_pca_scores

        formation_options = ["Unassigned", "Beaver Dam", "Choptank", "Calvert"]

        if app_mode == "Analysis & AI Training":
            num_clusters = st.sidebar.slider("Number of Formations:", 2, 5, 3)
            km = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
            df['Cluster_ID'] = km.fit_predict(final_pca_scores).astype(str)
            
            st.sidebar.subheader("Assign Formations")
            label_map = {c: st.sidebar.selectbox(f"Cluster {c}:", formation_options, key=f"l_{c}") for c in sorted(df['Cluster_ID'].unique())}
            
            df['Display_Label'] = df['Cluster_ID'].map(lambda x: label_map[x] if label_map[x] != "Unassigned" else f"Cluster {x}")
            
            if st.sidebar.button("🧠 Train/Overwrite Master AI"):
                model = RandomForestClassifier(n_estimators=250, random_state=42)
                model.fit(X_scaled[mask], df['Display_Label'])
                with open(model_path, "wb") as f:
                    pickle.dump((model, scaler, selected_elements), f)
                st.rerun()

        else:
            if os.path.exists(model_path):
                with open(model_path, "rb") as f:
                    model, saved_scaler, saved_elements = pickle.load(f)
                
                X_pred = df[saved_elements].apply(pd.to_numeric, errors='coerce').fillna(0)
                X_pred_scaled = saved_scaler.transform(X_pred)
                df['Display_Label'] = model.predict(X_pred_scaled)
            else:
                st.sidebar.error("Train model first!")
                st.stop()

        # --- VISUALIZATION TABS ---
        tab1, tab2, tab3 = st.tabs(["3D Space", "Element Drivers", "Stratigraphy"])

        with tab1:
            st.plotly_chart(px.scatter_3d(
                df, x='PC1', y='PC2', z='PC3', 
                color='Display_Label', 
                color_discrete_map=COLOR_DISCRETE_MAP,
                height=800, template="plotly_dark"
            ), use_container_width=True)

        with tab2:
            loadings = pd.DataFrame(pca_obj.components_.T, columns=['PC1', 'PC2', 'PC3'], index=selected_elements)
            pc_choice = st.radio("Inspect Axis Drivers:", ["PC1", "PC2", "PC3"], horizontal=True)
            st.plotly_chart(px.bar(loadings.reset_index(), x='index', y=pc_choice, color=pc_choice, 
                                   color_continuous_scale='RdBu_r'), use_container_width=True)

        with tab3:
            fig_strat = px.scatter(
                df, x='Source_File', y='Depth_Value', 
                color='Display_Label', 
                color_discrete_map=COLOR_DISCRETE_MAP,
                hover_data=['Sample_ID', 'Gamma_API'] if 'Gamma_API' in df.columns else ['Sample_ID'], height=800
            )
            fig_strat.update_traces(marker=dict(size=14, line=dict(width=1, color='white')))
            fig_strat.update_yaxes(autorange="reversed", title="Depth (ft)")
            fig_strat.update_xaxes(type='category', title="Borehole ID")
            st.plotly_chart(fig_strat, use_container_width=True)
            st.download_button("💾 Export CSV", df.to_csv(index=False), "xrf_strat_results.csv")
