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
        
        # Standardize column naming
        temp_df.columns = [c.strip() for c in temp_df.columns]
        
        # Flexible Depth / Sample column resolution
        sample_col = next((c for c in temp_df.columns if c.upper() in ['SAMPLE', 'DEPTH (IN)', 'DEPTH', 'DEPTH_FT']), 'Sample')
        
        temp_df = temp_df.assign(
            Source_File=str(file.name),
            Sample_ID=temp_df[sample_col].astype(str),
            Depth_Value=pd.to_numeric(temp_df[sample_col], errors='coerce')
        )
        
        # Helper to convert columns cleanly to numeric values
        def to_num(col_name):
            match = next((c for c in temp_df.columns if c.upper() == col_name.upper()), None)
            return pd.to_numeric(temp_df[match], errors='coerce') if match else None

        # --- DYNAMIC CROSS-FORMATION RATIOS ---
        K_val = to_num('K')
        Ca_val = to_num('Ca')
        
        Fe_val = to_num('Fe')
        if Fe_val is None:
            Fe_val = to_num('Iron')
            
        Cr_val = to_num('Cr')
        if Cr_val is None:
            Cr_val = to_num('Chromium')
        
        Zr_val = to_num('Zr')
        if Zr_val is None:
            Zr_val = to_num('Zirconium')
            
        Sr_val = to_num('Sr')
        if Sr_val is None:
            Sr_val = to_num('Strontium')

        # Compute safe ratios
        if K_val is not None and Zr_val is not None:
            temp_df['Ratio_K_Zr'] = (K_val / Zr_val.replace(0, np.nan)).fillna(0)
        if Ca_val is not None and Fe_val is not None:
            temp_df['Ratio_Ca_Fe'] = (Ca_val / Fe_val.replace(0, np.nan)).fillna(0)
        if Sr_val is not None and Cr_val is not None:
            temp_df['Ratio_Sr_Cr'] = (Sr_val / Cr_val.replace(0, np.nan)).fillna(0)

        # Match XRF depth values to continuous Gamma curve
        file_prefix = file.name.split('.')[0].upper()
        matched_key = next((k for k in gamma_data_map if k in file_prefix or file_prefix in k), None)
        
        if matched_key:
            g_log = gamma_data_map[matched_key]
            temp_df['Gamma_API'] = np.interp(temp_df['Depth_Value'], g_log['Depth'], g_log['Gamma'])
        else:
            temp_df['Gamma_API'] = 0.0

        all_data.append(temp_df)
    
    df_raw = pd.concat(all_data, ignore_index=True).dropna(subset=['Depth_Value']).copy()

    # ---------------------------------------------------------
    # ✂️ CORE DATA PRUNING
    # ---------------------------------------------------------
    st.sidebar.markdown("---")
    st.sidebar.subheader("✂️ Core Data Pruning")
    
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

    # Meta column filtering
    meta = ['Reading', 'Type', 'Time', 'Sample', 'Units', 'Sigma', 'CPS', 'Mode', 'Duration', 
            'Main', 'Low', 'High', 'Light', 'User', 'Batch', 'Heat', 'Lot', 'Note', 'Balance', 'Bal',
            'Source_File', 'Sample_ID', 'Depth_Value', 'PC1', 'PC2', 'PC3', 'Cluster_ID', 'Display_Label']
    
    elements = [c for c in df_raw.columns if not any(k.upper() in c.upper() for k in meta) 
                and "2-Sigma" not in c and "Unnamed" not in c]
    
    if len(gamma_data_map) > 0 and 'Gamma_API' in df_raw.columns:
        elements = sorted(list(set(elements + ['Gamma_API'])))
    elif 'Gamma_API' in elements:
        elements.remove('Gamma_API')

    st.sidebar.subheader("Select Features:")
    starting_features = ['Al', 'Ca', 'Fe', 'K', 'Zr', 'Ratio_K_Zr', 'Ratio_Ca_Fe', 'Ratio_Sr_Cr']
    
    selected_elements = st.sidebar.multiselect(
        "", 
        elements, 
        default=[e for e in starting_features if e in elements]
    )

    # ---------------------------------------------------------
    # 🧪 LOG-NORMAL & PREPROCESSING PIPELINE
    # ---------------------------------------------------------
    st.sidebar.subheader("🧪 Scaling Pipeline")
    use_log10 = st.sidebar.checkbox("Apply Log10(x + 1) Transformation", value=True, help="Normalizes log-normal trace distributions and handles 0 values cleanly.")

    if len(selected_elements) >= 3:
        X_num = df_raw[selected_elements].apply(pd.to_numeric, errors='coerce').fillna(0).copy()
        
        # 1. Log10(x + 1) transformation (PAST Equivalent)
        if use_log10:
            X_log = np.log10(X_num + 1)
        else:
            X_log = X_num

        # 2. Data Cleaning on Transformed Space
        st.sidebar.subheader("🧼 Data Cleaning")
        outlier_multiplier = st.sidebar.slider("Outlier Scrub (IQR Multiplier):", 1.5, 10.0, 4.0)
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_log)
        
        pca_obj = PCA(n_components=min(3, len(selected_elements)), random_state=42)
        pca_scores = pca_obj.fit_transform(X_scaled)
        
        # Calculate bounds along PC1 axis for scrubbing
        pc1_scores = pca_scores[:, 0]
        q25, q75 = np.percentile(pc1_scores, [25, 75])
        iqr = q75 - q25
        
        lower_bound = q25 - (outlier_multiplier * iqr)
        upper_bound = q75 + (outlier_multiplier * iqr)
        
        mask = (pc1_scores >= lower_bound) & (pc1_scores <= upper_bound)
        
        df = df_raw.loc[mask].copy() 
        final_pca_scores = pca_scores[mask]
        df[['PC1', 'PC2', 'PC3']] = final_pca_scores

        formation_options = ["Unassigned", "Beaver Dam", "Choptank", "Calvert"]

        if app_mode == "Analysis & AI Training":
            num_clusters = st.sidebar.slider("Number of Formations:", 2, 5, 3)
            km = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
            
            # 🎯 FIX: Cluster directly on full feature matrix X_scaled[mask] (Exact PAST Parity)
            df['Cluster_ID'] = km.fit_predict(X_scaled[mask]).astype(str)
            
            st.sidebar.subheader("Assign Formations")
            label_map = {c: st.sidebar.selectbox(f"Cluster {c}:", formation_options, key=f"l_{c}") for c in sorted(df['Cluster_ID'].unique())}
            
            df['Display_Label'] = df['Cluster_ID'].map(lambda x: label_map[x] if label_map[x] != "Unassigned" else f"Cluster {x}")
            
            if st.sidebar.button("🧠 Train/Overwrite Master AI"):
                model = RandomForestClassifier(n_estimators=250, random_state=42)
                # Fit model on the full-variance scaled features
                model.fit(X_scaled[mask], df['Display_Label'])
                with open(model_path, "wb") as f:
                    pickle.dump((model, scaler, selected_elements, use_log10), f)
                st.rerun()

        else:
            if os.path.exists(model_path):
                with open(model_path, "rb") as f:
                    model, saved_scaler, saved_elements, saved_use_log10 = pickle.load(f)
                
                X_pred = df[saved_elements].apply(pd.to_numeric, errors='coerce').fillna(0)
                if saved_use_log10:
                    X_pred = np.log10(X_pred + 1)
                    
                X_pred_scaled = saved_scaler.transform(X_pred)
                df['Display_Label'] = model.predict(X_pred_scaled)
            else:
                st.sidebar.error("Train model first!")
                st.stop()

        # --- 3-TAB DECOUPLED WORKFLOW ---
        tab1, tab2, tab3 = st.tabs(["🌌 PCA Space & Drivers", "📉 Down-Core Stratigraphy", "🤖 Master AI Training"])

        with tab1:
            st.subheader("1. Principal Component Analysis (Diagnostic View)")
            col_pca, col_loadings = st.columns([3, 2])
            
            with col_pca:
                st.plotly_chart(px.scatter_3d(
                    df, x='PC1', y='PC2', z='PC3', 
                    color='Display_Label', 
                    color_discrete_map=COLOR_DISCRETE_MAP,
                    height=700, template="plotly_dark",
                    title="3D PCA Score Space"
                ), use_container_width=True)
                
            with col_loadings:
                st.subheader("Element Drivers (Loadings)")
                loadings = pd.DataFrame(pca_obj.components_.T, columns=['PC1', 'PC2', 'PC3'], index=selected_elements)
                pc_choice = st.radio("Inspect Axis:", ["PC1", "PC2", "PC3"], horizontal=True)
                st.plotly_chart(px.bar(
                    loadings.reset_index(), x='index', y=pc_choice, color=pc_choice, 
                    color_continuous_scale='RdBu_r', height=550
                ), use_container_width=True)

        with tab2:
            st.subheader("2. Chemostratigraphic Core Log (Full Feature K-Means)")
            fig_strat = px.scatter(
                df, x='Source_File', y='Depth_Value', 
                color='Display_Label', 
                color_discrete_map=COLOR_DISCRETE_MAP,
                hover_data=['Sample_ID', 'Gamma_API'] if 'Gamma_API' in df.columns else ['Sample_ID'], 
                height=800
            )
            fig_strat.update_traces(marker=dict(size=14, line=dict(width=1, color='white')))
            fig_strat.update_yaxes(autorange="reversed", title="Depth (ft)")
            fig_strat.update_xaxes(type='category', title="Borehole ID")
            st.plotly_chart(fig_strat, use_container_width=True)
            st.download_button("💾 Export Core Log CSV", df.to_csv(index=False), "xrf_strat_results.csv")

        with tab3:
            st.subheader("3. Master AI Status & Re-Training Summary")
            st.write("Train your machine learning model on current verified labels to run auto-labeling on new, unclassified boreholes.")
            st.dataframe(df[['Sample_ID', 'Depth_Value', 'Display_Label'] + selected_elements].head(20))

else:
    st.info("Please upload XRF CSV file(s) from the sidebar to begin analysis.")
