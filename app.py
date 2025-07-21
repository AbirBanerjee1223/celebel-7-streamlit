import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
from streamlit_shap import st_shap
import warnings

# --- Page Configuration ---
st.set_page_config(
    page_title="Ames House Price Estimator",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Suppress the specific SHAP/Matplotlib warning ---
warnings.filterwarnings("ignore", message="FigureCanvasAgg is non-interactive, and thus cannot be shown")


# --- Resource Loading ---
@st.cache_resource
def load_resources():
    with open('ames_housing_pipeline.pkl', 'rb') as f:
        pipeline = pickle.load(f)
    with open('ames_model_features.pkl', 'rb') as f:
        features = pickle.load(f)
    df_train = pd.read_csv('train.csv')
    X_train = df_train.drop(['SalePrice', 'Id'], axis=1)

    # Recreate the exact same engineered features for the background data
    X_train['TotalBath'] = X_train['FullBath'] + (0.5 * X_train['HalfBath']) + X_train['BsmtFullBath'] + (0.5 * X_train['BsmtHalfBath'])
    X_train['TotalSF'] = X_train['TotalBsmtSF'] + X_train['1stFlrSF'] + X_train['2ndFlrSF']
    X_train['HouseAge'] = X_train['YrSold'] - X_train['YearBuilt']
    X_train['AgeSinceRemod'] = X_train['YrSold'] - X_train['YearRemodAdd']
    X_train['TotalPorchSF'] = X_train['OpenPorchSF'] + X_train['EnclosedPorch'] + X_train['3SsnPorch'] + X_train['ScreenPorch']

    return pipeline, features, X_train

pipeline, feature_names, X_train = load_resources()

# --- SHAP Explainer Initialization ---
@st.cache_resource
def get_explainer(_pipeline, _X_train):
    preprocessor = _pipeline.named_steps['preprocessor']
    model = _pipeline.named_steps['regressor']
    X_train_transformed = preprocessor.transform(_X_train).toarray()
    explainer = shap.TreeExplainer(model, X_train_transformed, feature_names=preprocessor.get_feature_names_out())
    return explainer

explainer = get_explainer(pipeline, X_train)

# --- UI: Title and Introduction ---
st.title("🏠 Ames House Price Estimator")
st.markdown("""
Welcome! I'm an AI-powered estimator built to predict house prices in Ames, Iowa. 
Just describe the house using the options on the left, and I'll give you an estimated price and explain what features mattered most!
""")
st.divider()

# --- UI: Sidebar for User Input ---
with st.sidebar:
    st.header("Tell Me About the House 📝")
    unique_vals = {
        'Neighborhood': X_train['Neighborhood'].unique().tolist(),
        'ExterQual': ['Ex', 'Gd', 'TA', 'Fa'],
        'KitchenQual': ['Ex', 'Gd', 'TA', 'Fa'],
    }

    def user_input_features():
        inputs = {}
        st.subheader("Key Property Details")
        inputs['OverallQual'] = st.select_slider('Overall Quality', options=range(1, 11), value=7, help="How would you rate the overall material and finish of the house? (1=Poor, 10=Excellent)")
        inputs['GrLivArea'] = st.number_input('Living Area (sq. ft)', min_value=300, max_value=5000, value=1700, step=50, help="Above grade (ground) living area square feet.")
        inputs['YearBuilt'] = st.number_input('Year Built', min_value=1870, max_value=2010, value=2005, step=1)
        
        st.subheader("Rooms & Spaces")
        inputs['GarageCars'] = st.select_slider('Garage Capacity (cars)', options=[0, 1, 2, 3, 4], value=2)
        inputs['FullBath'] = st.select_slider('Full Bathrooms', options=[0, 1, 2, 3, 4], value=2)
        inputs['HalfBath'] = st.select_slider('Half Bathrooms', options=[0, 1, 2], value=1)
        
        has_basement = st.toggle('Has a Basement?', value=True)
        if has_basement:
            inputs['TotalBsmtSF'] = st.number_input('Basement Area (sq. ft)', min_value=0, max_value=6000, value=864, step=50)
            inputs['BsmtFullBath'] = st.select_slider('Basement Full Bathrooms', options=[0, 1, 2, 3], value=0)
            inputs['BsmtHalfBath'] = st.select_slider('Basement Half Bathrooms', options=[0, 1, 2], value=0)
        else:
            inputs['TotalBsmtSF'] = 0
            inputs['BsmtFullBath'] = 0
            inputs['BsmtHalfBath'] = 0

        st.subheader("Location & Other Features")
        inputs['Neighborhood'] = st.selectbox('Neighborhood', options=unique_vals['Neighborhood'], index=unique_vals['Neighborhood'].index('CollgCr'))
        inputs['KitchenQual'] = st.selectbox('Kitchen Quality 🧑‍🍳', options=unique_vals['KitchenQual'])
        
        full_feature_df = pd.DataFrame(columns=feature_names).astype(X_train[feature_names].dtypes.to_dict())

        for key, value in inputs.items():
            if key in full_feature_df.columns:
                full_feature_df.at[0, key] = value
        
        current_year = 2010
        full_feature_df['YrSold'] = current_year
        full_feature_df['YearRemodAdd'] = inputs.get('YearBuilt')
        
        # **THE FIX IS HERE**: Replace inplace=True with direct assignment
        full_feature_df['1stFlrSF'] = full_feature_df['1stFlrSF'].fillna(inputs.get('GrLivArea', 1500))
        full_feature_df['2ndFlrSF'] = full_feature_df['2ndFlrSF'].fillna(0)
        for porch_col in ['OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch']:
            full_feature_df[porch_col] = full_feature_df[porch_col].fillna(0)
        
        full_feature_df['TotalBath'] = full_feature_df['FullBath'] + (0.5 * full_feature_df['HalfBath']) + full_feature_df['BsmtFullBath'] + (0.5 * full_feature_df['BsmtHalfBath'])
        full_feature_df['TotalSF'] = full_feature_df['TotalBsmtSF'] + full_feature_df['1stFlrSF'] + full_feature_df['2ndFlrSF']
        full_feature_df['HouseAge'] = full_feature_df['YrSold'] - full_feature_df['YearBuilt']
        full_feature_df['AgeSinceRemod'] = full_feature_df['YrSold'] - full_feature_df['YearRemodAdd']
        full_feature_df['TotalPorchSF'] = full_feature_df['OpenPorchSF'] + full_feature_df['EnclosedPorch'] + full_feature_df['3SsnPorch'] + full_feature_df['ScreenPorch']

        for col in feature_names:
            if pd.isna(full_feature_df.at[0, col]):
                if pd.api.types.is_numeric_dtype(X_train[col].dtype):
                    full_feature_df.at[0, col] = X_train[col].median()
                else:
                    full_feature_df.at[0, col] = X_train[col].mode()[0]
        
        return full_feature_df[feature_names]

    input_df = user_input_features()

# --- Main Panel for Displaying Results ---
col1, col2 = st.columns([1, 1.5])
with col1:
    st.subheader("Your Selections")
    display_df = input_df[['OverallQual', 'GrLivArea', 'YearBuilt', 'GarageCars', 'FullBath', 'TotalBsmtSF', 'Neighborhood', 'KitchenQual']].T
    display_df.columns = ['Value']
    # **THE SECOND FIX IS HERE**: Ensure all values are strings for consistent display
    st.dataframe(display_df.astype(str), use_container_width=True)
    predict_button = st.button("✨ Predict Price!", type="primary", use_container_width=True)

with col2:
    st.subheader("Prediction & Explanation")
    if predict_button:
        with st.spinner('Analyzing the data...'):
            log_prediction = pipeline.predict(input_df)
            prediction = np.expm1(log_prediction)[0]
            st.success(f"Estimated Sale Price: **${prediction:,.2f}**")
            
            st.markdown("#### What influenced this price?")
            st.info("The chart below shows which features pushed the price up (in red) and which pulled it down (in blue), starting from the average price.", icon="💡")
            
            input_transformed = pipeline.named_steps['preprocessor'].transform(input_df).toarray()
            shap_values = explainer(input_transformed)
            
            st_shap(shap.plots.waterfall(shap_values[0]), height=400)
    else:
        st.info("Click the 'Predict Price!' button to see the magic happen.")