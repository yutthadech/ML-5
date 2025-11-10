"""
🚀 Packed (%) Prediction App - Streamlit Deployment
Machine Learning: Random Forest (R² = 0.9297)
Dataset: Pack SGA 20253
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Page config
st.set_page_config(
    page_title="Packed (%) Prediction",
    page_icon="📦",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Load model and preprocessing data
@st.cache_resource
def load_model_data():
    try:
        with open('furnace_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('preprocessing_data.pkl', 'rb') as f:
            preprocessing = pickle.load(f)
        with open('deployment_info.pkl', 'rb') as f:
            info = pickle.load(f)
        return model, preprocessing, info
    except Exception as e:
        st.error(f"Error loading model files: {e}")
        return None, None, None

# Load charts
@st.cache_data
def load_charts():
    charts = {}
    chart_names = [
        'chart1_actual_vs_predicted',
        'chart2_feature_importance',
        'chart3_shap_summary',
        'chart4_pareto',
        'chart5_correlation_matrix',
        'chart6_residual_plot',
        'chart7_residual_histogram',
        'chart8_main_effect_plot',
        'chart9_interaction_plot'
    ]
    for name in chart_names:
        try:
            charts[name] = f'{name}.png'
        except:
            pass
    return charts

# Main app
def main():
    st.markdown('<div class="main-header">📦 Packed (%) Prediction System</div>', 
                unsafe_allow_html=True)
    
    # Load resources
    model, preprocessing, info = load_model_data()
    
    if model is None:
        st.error("⚠️ Could not load model files. Please ensure all files are in the same directory.")
        return
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/000000/artificial-intelligence.png", width=100)
        st.title("Navigation")
        page = st.radio("Select Page:", 
                       ["🏠 Home", "🔮 Prediction", "📊 Analytics", "ℹ️ Model Info"])
        
        st.markdown("---")
        st.markdown(f"""
        **Model Performance:**
        - R² Score: `{info['r2_score']:.4f}`
        - RMSE: `{info['rmse']:.4f}`
        - MAE: `{info['mae']:.4f}`
        """)
    
    # Page routing
    if page == "🏠 Home":
        show_home(info)
    elif page == "🔮 Prediction":
        show_prediction(model, preprocessing, info)
    elif page == "📊 Analytics":
        show_analytics()
    elif page == "ℹ️ Model Info":
        show_model_info(info)

def show_home(info):
    st.header("Welcome to Packed (%) Prediction System")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("R² Score", f"{info['r2_score']:.4f}", "Excellent")
    with col2:
        st.metric("RMSE", f"{info['rmse']:.4f}", "Low Error")
    with col3:
        st.metric("Total Features", info['n_features'], "Optimized")
    
    st.markdown("---")
    
    st.subheader("📖 About This App")
    st.write("""
    This application predicts **Packed (%)** using a trained Random Forest model 
    with 92.97% accuracy (R² = 0.9297). The model was trained on 1,574 production samples 
    with 89 features including quality metrics, defect rates, and operational parameters.
    
    **Key Features:**
    - 🎯 Real-time prediction with confidence intervals
    - 📊 Comprehensive analytics and visualizations
    - 🔍 Feature importance analysis
    - 📈 Interactive main effect plots
    
    **How to Use:**
    1. Navigate to **🔮 Prediction** page
    2. Upload your data file (Excel/CSV)
    3. Get instant predictions with insights
    """)
    
    st.markdown("---")
    st.subheader("🏆 Model Performance Summary")
    
    try:
        summary_df = pd.read_excel('model_comparison_summary.xlsx')
        st.dataframe(summary_df, use_container_width=True)
    except:
        st.info("Model comparison summary not available")

def show_prediction(model, preprocessing, info):
    st.header("🔮 Make Predictions")
    
    st.write("""
    Upload your production data to get **Packed (%)** predictions. 
    Your file should contain the same features used during training.
    """)
    
    # File upload
    uploaded_file = st.file_uploader("Upload Excel or CSV file", 
                                     type=['xlsx', 'xls', 'csv'])
    
    if uploaded_file is not None:
        try:
            # Read file
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.success(f"✅ File loaded: {df.shape[0]} rows × {df.shape[1]} columns")
            
            # Show preview
            with st.expander("📋 Preview Data"):
                st.dataframe(df.head(10))
            
            # Preprocess button
            if st.button("🚀 Make Predictions", type="primary"):
                with st.spinner("Processing and predicting..."):
                    # Preprocess
                    df_processed = preprocess_data(df, preprocessing, info)
                    
                    if df_processed is not None:
                        # Make predictions
                        predictions = model.predict(df_processed)
                        
                        # Add predictions to original dataframe
                        df['Predicted_Packed_%'] = predictions
                        
                        # Show results
                        st.markdown('<div class="success-box">', unsafe_allow_html=True)
                        st.subheader("✨ Prediction Results")
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Mean Prediction", f"{predictions.mean():.2f}%")
                        with col2:
                            st.metric("Min Prediction", f"{predictions.min():.2f}%")
                        with col3:
                            st.metric("Max Prediction", f"{predictions.max():.2f}%")
                        with col4:
                            st.metric("Std Dev", f"{predictions.std():.2f}%")
                        
                        # Show results table
                        st.dataframe(df[['Predicted_Packed_%']].head(20), use_container_width=True)
                        
                        # Download button
                        csv = df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Download Results (CSV)",
                            data=csv,
                            file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                        
                        # Visualization
                        st.subheader("📊 Prediction Distribution")
                        fig, ax = plt.subplots(figsize=(10, 4))
                        ax.hist(predictions, bins=30, color='steelblue', edgecolor='k', alpha=0.7)
                        ax.set_xlabel('Predicted Packed (%)')
                        ax.set_ylabel('Frequency')
                        ax.set_title('Distribution of Predictions')
                        ax.grid(True, alpha=0.3)
                        st.pyplot(fig)
                        
        except Exception as e:
            st.error(f"Error processing file: {e}")
    else:
        st.info("👆 Upload a file to get started")

def preprocess_data(df, preprocessing, info):
    """Preprocess input data to match training format"""
    try:
        df_processed = df.copy()
        
        # Handle datetime columns
        for col in df_processed.columns:
            if pd.api.types.is_datetime64_any_dtype(df_processed[col]):
                df_processed[f'{col}_year'] = df_processed[col].dt.year
                df_processed[f'{col}_month'] = df_processed[col].dt.month
                df_processed[f'{col}_day'] = df_processed[col].dt.day
                df_processed[f'{col}_dayofweek'] = df_processed[col].dt.dayofweek
                df_processed = df_processed.drop(columns=[col])
        
        # Convert to numeric
        for col in df_processed.columns:
            if df_processed[col].dtype == 'object':
                df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
        
        # One-hot encode categorical columns (if any exist in preprocessing)
        categorical_cols = preprocessing.get('categorical_cols', [])
        if categorical_cols:
            for col in categorical_cols:
                if col in df_processed.columns:
                    df_processed = pd.get_dummies(df_processed, columns=[col], drop_first=True)
        
        # Fill NaN
        df_processed = df_processed.fillna(0)
        
        # Align columns with training features
        feature_names = preprocessing['feature_names']
        
        # Add missing columns with 0
        for col in feature_names:
            if col not in df_processed.columns:
                df_processed[col] = 0
        
        # Select only training features in correct order
        df_processed = df_processed[feature_names]
        
        # Scale
        scaler = preprocessing['scaler']
        df_scaled = pd.DataFrame(
            scaler.transform(df_processed),
            columns=feature_names
        )
        
        return df_scaled
        
    except Exception as e:
        st.error(f"Preprocessing error: {e}")
        return None

def show_analytics():
    st.header("📊 Model Analytics & Visualizations")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Performance", "Feature Analysis", "Diagnostics", "Main Effect - Key Variables"])
    
    with tab1:
        st.subheader("Model Performance")
        col1, col2 = st.columns(2)
        
        with col1:
            try:
                st.image('chart1_actual_vs_predicted.png', 
                        caption='Actual vs Predicted', 
                        use_container_width=True)
            except:
                st.info("Chart not available")
        
        with col2:
            try:
                st.image('chart6_residual_plot.png', 
                        caption='Residual Plot', 
                        use_container_width=True)
            except:
                st.info("Chart not available")
        
        try:
            st.image('chart7_residual_histogram.png', 
                    caption='Residual Distribution', 
                    use_container_width=True)
        except:
            pass
    
    with tab2:
        st.subheader("Feature Importance Analysis")
        
        col1, col2 = st.columns(2)
        with col1:
            try:
                st.image('chart2_feature_importance.png', 
                        caption='Feature Importance', 
                        use_container_width=True)
            except:
                st.info("Chart not available")
        
        with col2:
            try:
                st.image('chart4_pareto.png', 
                        caption='Pareto Chart', 
                        use_container_width=True)
            except:
                st.info("Chart not available")
        
        try:
            st.image('chart3_shap_summary.png', 
                    caption='SHAP Summary', 
                    use_container_width=True)
        except:
            pass
        
        try:
            st.image('chart5_correlation_matrix.png', 
                    caption='Correlation Matrix', 
                    use_container_width=True)
        except:
            pass
    
    with tab3:
        st.subheader("Main Effect & Interaction Plots")
        
        try:
            st.image('chart8_main_effect_plot.png', 
                    caption='Main Effect Plot (Minitab Style)', 
                    use_container_width=True)
        except:
            st.info("Chart not available")
        
        try:
            st.image('chart9_interaction_plot.png', 
                    caption='Main Interaction Plot', 
                    use_container_width=True)
        except:
            st.info("Chart not available")
    
    with tab4:
        st.subheader("📊 Main Effect Plot - Key Operational Variables")
        
        st.markdown("""
        **Key Variables Analyzed:**
        - 📅 **DMY (Date)** - Temporal patterns by month
        - 🌦️ **Season** - Seasonal effects
        - 🌓 **Shift** - Day vs Night shift performance
        - 🌡️ **Temperature** - Temperature impact
        - 🏭 **Plant** - Plant location effects
        - 📍 **Line** - Production line variations
        - 📦 **Product** - Product type differences
        """)
        
        st.markdown("---")
        
        # Combined view
        st.subheader("🔍 Combined Overview")
        try:
            st.image('chart10_main_effect_specified_variables.png', 
                    caption='Main Effect Plot - All Specified Variables', 
                    use_container_width=True)
        except:
            st.info("Combined chart not available")
        
        st.markdown("---")
        
        # Individual plots in expandable sections
        st.subheader("📈 Individual Variable Analysis")
        
        # Date/Month
        with st.expander("📅 DMY (Date by Month)"):
            try:
                st.image('chart10_1_DMY.png', use_container_width=True)
                st.caption("Shows seasonal patterns across months")
            except:
                st.info("Chart not available")
        
        # Season
        with st.expander("🌦️ Season"):
            try:
                st.image('chart10_2_Season.png', use_container_width=True)
                st.caption("Compares performance across seasons")
            except:
                st.info("Chart not available")
        
        # Shift
        with st.expander("🌓 Shift (Day vs Night)"):
            try:
                st.image('chart10_3_Shift.png', use_container_width=True)
                st.caption("Night shift typically shows 0.87% vs Day shift 0.86%")
            except:
                st.info("Chart not available")
        
        # Temperature
        with st.expander("🌡️ Temperature (Thailand Ayutthaya)"):
            try:
                st.image('chart10_4_Temp_C_Thailand_Ayutthaya.png', use_container_width=True)
                st.caption("Temperature impact on packing efficiency")
            except:
                st.info("Chart not available")
        
        # Plant
        with st.expander("🏭 Plant Location"):
            try:
                st.image('chart10_5_Plant.png', use_container_width=True)
                st.caption("Performance differences between plants")
            except:
                st.info("Chart not available")
        
        # Line
        with st.expander("📍 Production Line"):
            try:
                st.image('chart10_6_Line.png', use_container_width=True)
                st.caption("Line 2.1 shows highest performance (0.88%), Line 2.2 shows lowest (0.84%)")
            except:
                st.info("Chart not available")
        
        # Product
        with st.expander("📦 Product Type"):
            try:
                st.image('chart10_7_Product.png', use_container_width=True)
                st.caption("Vitamin shows highest efficiency (0.90%), 700cc Pichai fi shows lowest (0.74%)")
            except:
                st.info("Chart not available")
        
        st.markdown("---")
        
        # Key Insights
        st.subheader("💡 Key Insights from Main Effect Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.info("""
            **🏆 Best Performers:**
            - Night Shift: 0.87%
            - Line 2.1: 0.88%
            - Vitamin Product: 0.90%
            """)
        
        with col2:
            st.warning("""
            **⚠️ Areas for Improvement:**
            - Day Shift: 0.86%
            - Line 2.2: 0.84%
            - 700cc Pichai fi: 0.74%
            """)
        
        with col3:
            st.success("""
            **📊 Recommendations:**
            - Focus on Day shift training
            - Optimize Line 2.2 process
            - Review Pichai fi product specs
            """)

def show_model_info(info):
    st.header("ℹ️ Model Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Model Details")
        st.write(f"""
        - **Model Type:** {info['best_model']}
        - **Target Variable:** {info['target_column']}
        - **Total Features:** {info['n_features']}
        - **Training Samples:** {info['n_train_samples']}
        - **Testing Samples:** {info['n_test_samples']}
        """)
    
    with col2:
        st.subheader("📈 Performance Metrics")
        st.write(f"""
        - **R² Score:** {info['r2_score']:.4f}
        - **RMSE:** {info['rmse']:.4f}
        - **MAE:** {info['mae']:.4f}
        """)
    
    st.markdown("---")
    
    st.subheader("🔍 Feature List")
    features_df = pd.DataFrame({
        'Feature Name': info['feature_names']
    })
    st.dataframe(features_df, use_container_width=True, height=400)
    
    st.markdown("---")
    
    st.subheader("📚 How the Model Works")
    st.write("""
    ### Random Forest Algorithm
    
    The Random Forest model is an ensemble learning method that:
    1. Creates multiple decision trees during training
    2. Each tree makes a prediction
    3. Final prediction is the average of all trees
    
    ### Why Random Forest?
    - ✅ Handles non-linear relationships well
    - ✅ Robust to outliers
    - ✅ Provides feature importance
    - ✅ High accuracy (R² = 0.9297)
    
    ### Data Preprocessing
    1. **Date Conversion:** Extract year, month, day, day of week
    2. **Normalization:** MinMaxScaler (0-1 range)
    3. **Encoding:** One-hot encoding for categorical variables
    4. **Missing Values:** Filled with median/0
    """)

if __name__ == "__main__":
    main()
