"""
Streamlit Frontend for Dynamic Airline Ticket Price Forecasting
Professional UI with visualization charts and expandable code sections
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, date
import joblib
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append('src')

from src.data.data_acquisition import DataAcquisitionManager
from src.data.preprocessing import FlightDataPreprocessor
from src.models.ml_models import ModelManager, QUANTUM_AVAILABLE
from src.utils.price_analysis import PriceAlertBot, FuelPriceSimulator

# Page configuration
st.set_page_config(
    page_title="✈️ Dynamic Airline Price Forecasting",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: bold;
    }
    
    .section-header {
        font-size: 1.5rem;
        color: #2c3e50;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
        margin: 1rem 0;
    }
    
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3498db;
        margin: 0.5rem 0;
    }
    
    .code-container {
        background-color: #f1f3f4;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
        font-family: 'Courier New', monospace;
        font-size: 0.9rem;
        overflow-x: auto;
    }
    
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.25rem;
        padding: 0.75rem;
        color: #155724;
    }
    
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.25rem;
        padding: 0.75rem;
        color: #856404;
    }
    
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 0.25rem;
        padding: 0.75rem;
        color: #721c24;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'preprocessor' not in st.session_state:
    st.session_state.preprocessor = None
if 'model_manager' not in st.session_state:
    st.session_state.model_manager = None
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None

@st.cache_data
def load_and_process_data():
    """Load and process the airline data"""
    try:
        preprocessor = FlightDataPreprocessor()
        X_train, X_val, X_test, y_train, y_val, y_test, df_full = preprocessor.process_full_pipeline()
        
        return {
            'preprocessor': preprocessor,
            'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
            'y_train': y_train, 'y_val': y_val, 'y_test': y_test,
            'df_full': df_full,
            'feature_columns': preprocessor.feature_columns
        }
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def display_code_section(title: str, code: str, language: str = "python"):
    """Display expandable code section"""
    with st.expander(f"📋 View Code: {title}"):
        st.code(code, language=language)

def create_data_overview_visualizations(df: pd.DataFrame):
    """Create data overview visualizations"""
    st.markdown('<div class="section-header">📊 Dataset Overview</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", len(df))
    with col2:
        st.metric("Features", len(df.columns))
    with col3:
        st.metric("Airlines", df['airline'].nunique() if 'airline' in df.columns else 0)
    with col4:
        st.metric("Routes", len(df[['from', 'to']].drop_duplicates()) if 'from' in df.columns and 'to' in df.columns else 0)
    
    # Price distribution
    col1, col2 = st.columns(2)
    
    with col1:
        if 'price' in df.columns:
            fig = px.histogram(df, x='price', nbins=50, title='Price Distribution')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if 'airline' in df.columns and 'price' in df.columns:
            fig = px.box(df, x='airline', y='price', title='Price by Airline')
            fig.update_xaxis(tickangle=45)
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    # Seasonal trends
    if 'date' in df.columns and 'price' in df.columns:
        df_temp = df.copy()
        df_temp['date'] = pd.to_datetime(df_temp['date'])
        df_temp['month'] = df_temp['date'].dt.month
        monthly_avg = df_temp.groupby('month')['price'].mean().reset_index()
        
        fig = px.line(monthly_avg, x='month', y='price', title='Seasonal Price Trends')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Display code for data loading
    data_loading_code = '''
# Data Loading and Basic Analysis
import pandas as pd
import plotly.express as px

# Load datasets
clean_df = pd.read_csv("Kaggle Airline Dataset/Clean_Dataset.csv")
business_df = pd.read_csv("Kaggle Airline Dataset/business.csv")
economy_df = pd.read_csv("Kaggle Airline Dataset/economy.csv")

# Combine datasets
combined_df = preprocess_and_combine_datasets(clean_df, business_df, economy_df)

# Basic statistics
print(f"Dataset shape: {combined_df.shape}")
print(f"Price range: ${combined_df['price'].min():.2f} - ${combined_df['price'].max():.2f}")

# Visualizations
fig = px.histogram(combined_df, x='price', title='Price Distribution')
fig.show()
'''
    display_code_section("Data Loading and Analysis", data_loading_code)

def create_preprocessing_visualizations(preprocessor, df_full):
    """Create preprocessing visualizations"""
    st.markdown('<div class="section-header">🔧 Data Preprocessing</div>', unsafe_allow_html=True)
    
    # Feature importance (if available)
    if hasattr(preprocessor, 'feature_columns'):
        st.subheader("Feature Engineering Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Total Features Created:** {len(preprocessor.feature_columns)}")
            st.write("**Feature Categories:**")
            
            feature_categories = {
                'Temporal': [f for f in preprocessor.feature_columns if any(x in f for x in ['year', 'month', 'day', 'week', 'season'])],
                'Route': [f for f in preprocessor.feature_columns if any(x in f for x in ['from', 'to', 'route', 'major'])],
                'External': [f for f in preprocessor.feature_columns if any(x in f for x in ['fuel', 'holiday'])],
                'Airline': [f for f in preprocessor.feature_columns if any(x in f for x in ['airline', 'flight'])],
                'Time': [f for f in preprocessor.feature_columns if any(x in f for x in ['dep_time', 'arr_time', 'duration'])]
            }
            
            for category, features in feature_categories.items():
                if features:
                    st.write(f"- **{category}:** {len(features)} features")
        
        with col2:
            # Correlation heatmap for top features
            if len(preprocessor.feature_columns) > 5:
                numeric_features = df_full[preprocessor.feature_columns[:10]].select_dtypes(include=[np.number])
                if not numeric_features.empty:
                    fig, ax = plt.subplots(figsize=(10, 8))
                    sns.heatmap(numeric_features.corr(), annot=True, cmap='coolwarm', center=0, ax=ax)
                    st.pyplot(fig)
    
    # Missing values analysis
    if not df_full.empty:
        missing_data = df_full.isnull().sum()
        missing_data = missing_data[missing_data > 0]
        
        if not missing_data.empty:
            fig = px.bar(x=missing_data.index, y=missing_data.values, 
                        title='Missing Values by Feature')
            st.plotly_chart(fig, use_container_width=True)
    
    # Display preprocessing code
    preprocessing_code = '''
# Data Preprocessing Pipeline
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.model_selection import train_test_split

class FlightDataPreprocessor:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = RobustScaler()
    
    def clean_data(self, df):
        # Remove duplicates
        df = df.drop_duplicates()
        
        # Handle missing values
        df = self.handle_missing_values(df)
        
        # Clean price column
        df['price'] = df['price'].str.replace(',', '').astype(float)
        
        # Remove outliers using IQR
        Q1 = df['price'].quantile(0.25)
        Q3 = df['price'].quantile(0.75)
        IQR = Q3 - Q1
        df = df[(df['price'] >= Q1 - 1.5*IQR) & (df['price'] <= Q3 + 1.5*IQR)]
        
        return df
    
    def feature_engineering(self, df):
        # Temporal features
        df['date'] = pd.to_datetime(df['date'])
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day_of_week'] = df['date'].dt.dayofweek
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Route features
        major_cities = ['Delhi', 'Mumbai', 'Bangalore', 'Chennai']
        df['from_major_city'] = df['from'].isin(major_cities).astype(int)
        df['to_major_city'] = df['to'].isin(major_cities).astype(int)
        
        # External features (fuel prices, holidays)
        df = self.add_external_features(df)
        
        return df
    
    def encode_features(self, df):
        categorical_cols = ['airline', 'from', 'to', 'class']
        
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
            df[col + '_encoded'] = self.label_encoders[col].fit_transform(df[col])
        
        return df
'''
    display_code_section("Data Preprocessing Pipeline", preprocessing_code)

def train_models_section():
    """Model training section"""
    st.markdown('<div class="section-header">🤖 Model Training</div>', unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.warning("Please load and process data first.")
        return
    
    if st.button("🚀 Train All Models", type="primary"):
        with st.spinner("Training models... This may take several minutes."):
            try:
                # Initialize model manager
                model_manager = ModelManager()
                data = st.session_state.processed_data
                
                # Initialize models
                models = model_manager.initialize_models(n_features=len(data['feature_columns']))
                
                # Train all models
                results = model_manager.train_all_models(
                    data['X_train'], data['y_train'],
                    data['X_val'], data['y_val']
                )
                
                st.session_state.model_manager = model_manager
                st.session_state.models_trained = True
                st.session_state.training_results = results
                
                st.success("✅ All models trained successfully!")
                
            except Exception as e:
                st.error(f"❌ Error training models: {str(e)}")
    
    # Display training results
    if st.session_state.models_trained and hasattr(st.session_state, 'training_results'):
        st.subheader("📈 Model Performance Comparison")
        
        # Create comparison DataFrame
        comparison_data = []
        for model_name, result in st.session_state.training_results.items():
            if 'val_metrics' in result:
                metrics = result['val_metrics']
                comparison_data.append({
                    'Model': model_name.replace('_', ' ').title(),
                    'RMSE': metrics.get('rmse', 0),
                    'MAE': metrics.get('mae', 0),
                    'R²': metrics.get('r2', 0),
                    'MAPE (%)': metrics.get('mape', 0)
                })
        
        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            
            # Display metrics table
            st.dataframe(comparison_df, use_container_width=True)
            
            # Visualize model comparison
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.bar(comparison_df, x='Model', y='RMSE', 
                           title='Model Comparison - RMSE')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.bar(comparison_df, x='Model', y='R²', 
                           title='Model Comparison - R² Score')
                st.plotly_chart(fig, use_container_width=True)
            
            # Best model highlight
            best_model_idx = comparison_df['RMSE'].idxmin()
            best_model = comparison_df.loc[best_model_idx]
            
            st.markdown(f"""
            <div class="success-box">
                🏆 <strong>Best Performing Model:</strong> {best_model['Model']}<br>
                📊 <strong>RMSE:</strong> {best_model['RMSE']:.2f}<br>
                📈 <strong>R² Score:</strong> {best_model['R²']:.3f}<br>
                📉 <strong>MAPE:</strong> {best_model['MAPE (%)']:.2f}%
            </div>
            """, unsafe_allow_html=True)
    
    # Display model training code
    model_training_code = '''
# Model Training Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from tensorflow import keras
from qiskit_machine_learning.algorithms import VQR

class ModelManager:
    def __init__(self):
        self.models = {}
    
    def initialize_models(self):
        # Linear Regression
        self.models['linear_regression'] = Ridge(alpha=1.0)
        
        # Random Forest
        self.models['random_forest'] = RandomForestRegressor(
            n_estimators=100, max_depth=20, random_state=42
        )
        
        # XGBoost
        self.models['xgboost'] = xgb.XGBRegressor(
            n_estimators=100, max_depth=6, learning_rate=0.1
        )
        
        # LSTM Neural Network
        lstm_model = keras.Sequential([
            keras.layers.LSTM(50, return_sequences=True),
            keras.layers.Dropout(0.2),
            keras.layers.LSTM(25),
            keras.layers.Dense(1)
        ])
        lstm_model.compile(optimizer='adam', loss='mse')
        self.models['lstm'] = lstm_model
        
        # Quantum Hybrid Model
        if QUANTUM_AVAILABLE:
            self.models['quantum_hybrid'] = self.create_quantum_model()
    
    def train_all_models(self, X_train, y_train, X_val, y_val):
        results = {}
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            
            if name == 'lstm':
                # Special handling for LSTM
                X_seq, y_seq = self.create_sequences(X_train, y_train)
                model.fit(X_seq, y_seq, epochs=50, validation_split=0.2)
            else:
                # Standard training
                model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            r2 = r2_score(y_val, y_pred)
            
            results[name] = {'rmse': rmse, 'r2': r2}
        
        return results
'''
    display_code_section("Model Training Code", model_training_code)

def price_prediction_section():
    """Price prediction interface"""
    st.markdown('<div class="section-header">💰 Price Prediction</div>', unsafe_allow_html=True)
    
    if not st.session_state.models_trained:
        st.warning("Please train models first.")
        return
    
    st.subheader("✈️ Predict Flight Price")
    
    # Input form
    col1, col2, col3 = st.columns(3)
    
    with col1:
        airline = st.selectbox("Airline", 
                              ['SpiceJet', 'IndiGo', 'Air India', 'Vistara', 'AirAsia', 'GO_FIRST'])
        from_city = st.selectbox("From", 
                                ['Delhi', 'Mumbai', 'Bangalore', 'Chennai', 'Kolkata', 'Hyderabad'])
        to_city = st.selectbox("To", 
                              ['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Kolkata', 'Hyderabad'])
    
    with col2:
        departure_date = st.date_input("Departure Date", 
                                     min_value=date.today(),
                                     max_value=date.today() + timedelta(days=365))
        departure_time = st.selectbox("Departure Time", 
                                    ['Early Morning', 'Morning', 'Afternoon', 'Evening', 'Night'])
        flight_class = st.selectbox("Class", ['Economy', 'Business'])
    
    with col3:
        stops = st.selectbox("Stops", ['zero', 'one', 'two+'])
        duration = st.number_input("Duration (hours)", min_value=0.5, max_value=24.0, value=2.5)
        current_fuel_price = st.number_input("Current Fuel Price ($/barrel)", 
                                           min_value=50.0, max_value=200.0, value=100.0)
    
    if st.button("🔮 Predict Price", type="primary"):
        try:
            # Create feature vector
            features = create_prediction_features(
                airline, from_city, to_city, departure_date, departure_time,
                flight_class, stops, duration, current_fuel_price
            )
            
            # Get predictions from all models
            model_manager = st.session_state.model_manager
            predictions = {}
            
            for name, model in model_manager.trained_models.items():
                try:
                    pred = model.predict(pd.DataFrame([features]))[0]
                    predictions[name] = pred
                except Exception as e:
                    st.warning(f"Could not get prediction from {name}: {str(e)}")
            
            if predictions:
                # Display predictions
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("🎯 Model Predictions")
                    for model_name, price in predictions.items():
                        st.metric(model_name.replace('_', ' ').title(), f"₹{price:,.0f}")
                
                with col2:
                    # Ensemble predictions
                    avg_price = np.mean(list(predictions.values()))
                    min_price = min(predictions.values())
                    max_price = max(predictions.values())
                    
                    st.subheader("📊 Ensemble Results")
                    st.metric("Average Prediction", f"₹{avg_price:,.0f}")
                    st.metric("Price Range", f"₹{min_price:,.0f} - ₹{max_price:,.0f}")
                    st.metric("Uncertainty", f"±₹{(max_price - min_price)/2:,.0f}")
                
                # Visualization
                pred_df = pd.DataFrame(list(predictions.items()), 
                                     columns=['Model', 'Predicted Price'])
                
                fig = px.bar(pred_df, x='Model', y='Predicted Price',
                           title='Price Predictions by Model')
                fig.add_hline(y=avg_price, line_dash="dash", 
                            annotation_text=f"Average: ₹{avg_price:,.0f}")
                st.plotly_chart(fig, use_container_width=True)
                
                # Confidence intervals
                st.subheader("🎯 Prediction Confidence")
                confidence_score = calculate_prediction_confidence(predictions)
                
                confidence_color = "green" if confidence_score > 0.8 else "orange" if confidence_score > 0.6 else "red"
                confidence_text = "High" if confidence_score > 0.8 else "Medium" if confidence_score > 0.6 else "Low"
                
                st.markdown(f"""
                <div style="background-color: {confidence_color}20; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid {confidence_color};">
                    <strong>Confidence Level:</strong> {confidence_text} ({confidence_score:.2f})<br>
                    <strong>Recommendation:</strong> {generate_price_recommendation(avg_price, confidence_score)}
                </div>
                """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
    
    # Display prediction code
    prediction_code = '''
# Price Prediction Implementation
def predict_flight_price(features):
    predictions = {}
    
    # Linear Regression Prediction
    lr_pred = linear_model.predict([features])[0]
    predictions['Linear Regression'] = lr_pred
    
    # Random Forest Prediction
    rf_pred = rf_model.predict([features])[0]
    predictions['Random Forest'] = rf_pred
    
    # XGBoost Prediction
    xgb_pred = xgb_model.predict([features])[0]
    predictions['XGBoost'] = xgb_pred
    
    # LSTM Prediction (requires sequence)
    lstm_features = create_sequence(features)
    lstm_pred = lstm_model.predict(lstm_features)[0]
    predictions['LSTM'] = lstm_pred
    
    # Quantum Hybrid Prediction
    quantum_features = create_quantum_features(features)
    quantum_pred = quantum_model.predict(quantum_features)[0]
    predictions['Quantum Hybrid'] = quantum_pred
    
    # Ensemble prediction
    ensemble_pred = np.mean(list(predictions.values()))
    
    return predictions, ensemble_pred

# Feature Engineering for Prediction
def create_prediction_features(airline, from_city, to_city, date, time, class_type):
    features = {}
    
    # Temporal features
    dt = pd.to_datetime(date)
    features['year'] = dt.year
    features['month'] = dt.month
    features['day_of_week'] = dt.dayofweek
    features['is_weekend'] = int(dt.dayofweek >= 5)
    
    # Route features
    major_cities = ['Delhi', 'Mumbai', 'Bangalore', 'Chennai']
    features['from_major_city'] = int(from_city in major_cities)
    features['to_major_city'] = int(to_city in major_cities)
    
    # External features
    fuel_price = get_current_fuel_price()
    features['fuel_price'] = fuel_price
    
    holiday_info = get_holiday_info(date)
    features['is_holiday'] = holiday_info['is_holiday']
    features['holiday_impact'] = holiday_info['impact_score']
    
    return features
'''
    display_code_section("Price Prediction Implementation", prediction_code)

def price_alert_section():
    """Price alert bot section"""
    st.markdown('<div class="section-header">🚨 Price Alert Bot</div>', unsafe_allow_html=True)
    
    if not st.session_state.models_trained:
        st.warning("Please train models first.")
        return
    
    st.subheader("📈 Optimal Booking Time Analysis")
    
    # Input for route analysis
    col1, col2 = st.columns(2)
    
    with col1:
        alert_from = st.selectbox("From City", 
                                ['Delhi', 'Mumbai', 'Bangalore', 'Chennai'], 
                                key="alert_from")
        alert_to = st.selectbox("To City", 
                              ['Mumbai', 'Delhi', 'Bangalore', 'Chennai'], 
                              key="alert_to")
    
    with col2:
        target_departure = st.date_input("Target Departure Date",
                                       min_value=date.today() + timedelta(days=1),
                                       max_value=date.today() + timedelta(days=365),
                                       value=date.today() + timedelta(days=30))
    
    if st.button("🔍 Analyze Optimal Booking Time", type="primary"):
        try:
            # Initialize price alert bot
            data_manager = DataAcquisitionManager()
            alert_bot = PriceAlertBot(st.session_state.model_manager, data_manager)
            
            route = f"{alert_from}-{alert_to}"
            
            # Get optimal booking recommendation
            with st.spinner("Analyzing historical patterns and predicting optimal booking time..."):
                recommendation = alert_bot.predict_optimal_booking_time(
                    route, str(target_departure)
                )
            
            if 'error' not in recommendation:
                # Display recommendation
                st.markdown(f"""
                <div class="success-box">
                    <h4>🎯 Optimal Booking Recommendation</h4>
                    <strong>Best Booking Date:</strong> {recommendation['optimal_booking_date']}<br>
                    <strong>Days in Advance:</strong> {recommendation['optimal_days_ahead']} days<br>
                    <strong>Predicted Price:</strong> ₹{recommendation['predicted_lowest_price']:,.0f}<br>
                    <strong>Confidence:</strong> {recommendation['confidence_score']:.2f}<br><br>
                    <strong>💡 Recommendation:</strong> {recommendation['recommendation']}
                </div>
                """, unsafe_allow_html=True)
                
                # Visualize booking scenarios
                scenarios_df = pd.DataFrame(recommendation['all_scenarios'])
                
                fig = px.line(scenarios_df, x='days_ahead', y='predicted_price',
                            title='Price Prediction by Booking Window',
                            labels={'days_ahead': 'Days Before Departure', 
                                   'predicted_price': 'Predicted Price (₹)'})
                
                # Highlight optimal point
                optimal_point = scenarios_df[scenarios_df['days_ahead'] == recommendation['optimal_days_ahead']]
                fig.add_scatter(x=optimal_point['days_ahead'], y=optimal_point['predicted_price'],
                              mode='markers', marker=dict(size=15, color='red'),
                              name='Optimal Booking Time')
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Confidence visualization
                fig_conf = px.line(scenarios_df, x='days_ahead', y='confidence_score',
                                 title='Prediction Confidence by Booking Window')
                st.plotly_chart(fig_conf, use_container_width=True)
            
            else:
                st.error(recommendation['error'])
        
        except Exception as e:
            st.error(f"Error in price alert analysis: {str(e)}")
    
    # Display price alert code
    alert_code = '''
# Price Alert Bot Implementation
class PriceAlertBot:
    def __init__(self, model_manager, data_manager):
        self.model_manager = model_manager
        self.data_manager = data_manager
    
    def predict_optimal_booking_time(self, route, departure_date):
        target_date = pd.to_datetime(departure_date)
        current_date = datetime.now()
        
        booking_scenarios = []
        
        # Generate scenarios for different booking windows
        for days_ahead in range(1, 91, 7):
            booking_date = target_date - timedelta(days=days_ahead)
            
            if booking_date < current_date:
                continue
            
            # Create feature vector
            features = self.create_prediction_features(
                route, booking_date, days_ahead
            )
            
            # Get prediction
            best_model = self.model_manager.get_best_model()
            predicted_price = best_model.predict([features])[0]
            
            # Calculate confidence
            confidence = self.calculate_confidence(days_ahead)
            
            booking_scenarios.append({
                'booking_date': booking_date,
                'days_ahead': days_ahead,
                'predicted_price': predicted_price,
                'confidence_score': confidence
            })
        
        # Find optimal booking time
        optimal_scenario = min(booking_scenarios, 
                             key=lambda x: x['predicted_price'])
        
        return {
            'optimal_booking_date': optimal_scenario['booking_date'],
            'optimal_days_ahead': optimal_scenario['days_ahead'],
            'predicted_lowest_price': optimal_scenario['predicted_price'],
            'confidence_score': optimal_scenario['confidence_score'],
            'all_scenarios': booking_scenarios,
            'recommendation': self.generate_recommendation(optimal_scenario)
        }
    
    def analyze_historical_patterns(self, route_data):
        patterns = {
            'seasonal_trends': self.analyze_seasonal_trends(route_data),
            'weekly_patterns': self.analyze_weekly_patterns(route_data),
            'booking_window_analysis': self.analyze_booking_window(route_data),
            'holiday_impact': self.analyze_holiday_impact(route_data)
        }
        return patterns
'''
    display_code_section("Price Alert Bot Implementation", alert_code)

def fuel_simulation_section():
    """Fuel price impact simulation"""
    st.markdown('<div class="section-header">⛽ Fuel Price Impact Simulation</div>', unsafe_allow_html=True)
    
    if not st.session_state.models_trained:
        st.warning("Please train models first.")
        return
    
    st.subheader("📊 Simulate Fuel Price Impact on Ticket Prices")
    
    # Base scenario inputs
    col1, col2 = st.columns(2)
    
    with col1:
        base_airline = st.selectbox("Airline", 
                                  ['SpiceJet', 'IndiGo', 'Air India', 'Vistara'], 
                                  key="fuel_airline")
        base_route = st.selectbox("Route", 
                                ['Delhi-Mumbai', 'Mumbai-Bangalore', 'Delhi-Chennai'], 
                                key="fuel_route")
        base_date = st.date_input("Travel Date", 
                                value=date.today() + timedelta(days=30),
                                key="fuel_date")
    
    with col2:
        current_fuel_price = st.slider("Current Fuel Price ($/barrel)", 
                                     min_value=60, max_value=150, value=100)
        simulation_range = st.slider("Simulation Range (±%)", 
                                   min_value=10, max_value=50, value=30)
    
    if st.button("🧪 Run Fuel Price Simulation", type="primary"):
        try:
            # Initialize fuel simulator
            fuel_simulator = FuelPriceSimulator(st.session_state.model_manager)
            
            # Create base features
            base_features = create_simulation_features(
                base_airline, base_route, base_date, current_fuel_price
            )
            
            # Generate fuel price scenarios
            scenarios = fuel_simulator.create_fuel_scenarios(
                current_fuel_price, simulation_range / 100
            )
            
            with st.spinner("Running fuel price impact simulation..."):
                results = fuel_simulator.simulate_fuel_impact(
                    pd.DataFrame([base_features]), scenarios
                )
            
            if results:
                # Display summary
                summary = results['summary']
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Price Range", 
                            f"₹{summary['price_range']['min']:,.0f} - ₹{summary['price_range']['max']:,.0f}")
                with col2:
                    st.metric("Max Price Impact", f"{summary['max_price_impact']:.1f}%")
                with col3:
                    st.metric("Sensitivity Level", summary['sensitivity_level'])
                
                # Visualization
                scenarios_df = pd.DataFrame(results['scenarios'])
                
                # Price vs Fuel Price
                fig = px.scatter(scenarios_df, x='fuel_price', y='predicted_price',
                               title='Ticket Price vs Fuel Price',
                               labels={'fuel_price': 'Fuel Price ($/barrel)',
                                      'predicted_price': 'Predicted Ticket Price (₹)'})
                
                # Add trend line
                from scipy import stats
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    scenarios_df['fuel_price'], scenarios_df['predicted_price']
                )
                line_x = np.array([scenarios_df['fuel_price'].min(), scenarios_df['fuel_price'].max()])
                line_y = slope * line_x + intercept
                
                fig.add_scatter(x=line_x, y=line_y, mode='lines', name=f'Trend (R²={r_value**2:.3f})')
                st.plotly_chart(fig, use_container_width=True)
                
                # Price impact visualization
                fig2 = px.bar(scenarios_df, x='fuel_change_pct', y='price_impact_pct',
                            title='Price Impact by Fuel Price Change',
                            labels={'fuel_change_pct': 'Fuel Price Change (%)',
                                   'price_impact_pct': 'Ticket Price Impact (%)'})
                st.plotly_chart(fig2, use_container_width=True)
                
                # Elasticity analysis
                st.subheader("📈 Price Elasticity Analysis")
                avg_elasticity = summary['average_elasticity']
                
                elasticity_interpretation = {
                    "Very Low": avg_elasticity < 0.2,
                    "Low": 0.2 <= avg_elasticity < 0.5,
                    "Moderate": 0.5 <= avg_elasticity < 1.0,
                    "High": 1.0 <= avg_elasticity < 2.0,
                    "Very High": avg_elasticity >= 2.0
                }
                
                elasticity_level = next(level for level, condition in elasticity_interpretation.items() if condition)
                
                st.markdown(f"""
                <div class="metric-card">
                    <strong>Average Price Elasticity:</strong> {avg_elasticity:.3f}<br>
                    <strong>Elasticity Level:</strong> {elasticity_level}<br>
                    <strong>Interpretation:</strong> A 1% increase in fuel prices leads to a {avg_elasticity:.3f}% increase in ticket prices.
                </div>
                """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Error in fuel price simulation: {str(e)}")
    
    # Display simulation code
    simulation_code = '''
# Fuel Price Impact Simulation
class FuelPriceSimulator:
    def __init__(self, model_manager):
        self.model_manager = model_manager
    
    def simulate_fuel_impact(self, base_features, fuel_scenarios):
        results = {}
        base_fuel_price = base_features['fuel_price_usd_per_barrel'].iloc[0]
        
        # Get best model
        best_model = self.model_manager.get_best_model()
        
        for fuel_price in fuel_scenarios:
            # Create modified features
            modified_features = base_features.copy()
            modified_features['fuel_price_usd_per_barrel'] = fuel_price
            modified_features['fuel_price_inr_per_liter'] = fuel_price * 0.53
            
            # Get prediction
            predicted_price = best_model.predict(modified_features)[0]
            
            # Calculate impact
            fuel_change_pct = ((fuel_price - base_fuel_price) / base_fuel_price) * 100
            
            results[fuel_price] = {
                'predicted_price': predicted_price,
                'fuel_change_pct': fuel_change_pct,
                'price_elasticity': self.calculate_elasticity(
                    base_fuel_price, fuel_price, base_features, modified_features
                )
            }
        
        return self.format_results(results, base_fuel_price)
    
    def calculate_elasticity(self, base_fuel, new_fuel, base_features, new_features):
        best_model = self.model_manager.get_best_model()
        
        base_price = best_model.predict(base_features)[0]
        new_price = best_model.predict(new_features)[0]
        
        fuel_change_pct = (new_fuel - base_fuel) / base_fuel
        price_change_pct = (new_price - base_price) / base_price
        
        return price_change_pct / fuel_change_pct if fuel_change_pct != 0 else 0
    
    def create_fuel_scenarios(self, base_price, range_pct=0.3):
        scenarios = []
        for pct_change in np.arange(-range_pct, range_pct + 0.1, 0.1):
            new_price = base_price * (1 + pct_change)
            scenarios.append(round(new_price, 2))
        return scenarios
'''
    display_code_section("Fuel Price Impact Simulation", simulation_code)

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<div class="main-header">✈️ Dynamic Airline Ticket Price Forecasting</div>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem; color: #666;">
        AI-Powered Flight Price Prediction with Advanced Analytics | Featuring 5 ML Models including Quantum Hybrid
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("🧭 Navigation")
    page = st.sidebar.selectbox(
        "Select Section",
        ["📊 Data Overview", "🔧 Preprocessing", "🤖 Model Training", 
         "💰 Price Prediction", "🚨 Price Alert Bot", "⛽ Fuel Impact Simulation"]
    )
    
    # System status
    st.sidebar.markdown("### 📈 System Status")
    st.sidebar.write(f"Data Loaded: {'✅' if st.session_state.data_loaded else '❌'}")
    st.sidebar.write(f"Models Trained: {'✅' if st.session_state.models_trained else '❌'}")
    st.sidebar.write(f"Quantum Available: {'✅' if QUANTUM_AVAILABLE else '❌'}")
    
    # Load data if not already loaded
    if not st.session_state.data_loaded:
        with st.spinner("Loading and processing data..."):
            try:
                processed_data = load_and_process_data()
                if processed_data:
                    st.session_state.processed_data = processed_data
                    st.session_state.preprocessor = processed_data['preprocessor']
                    st.session_state.data_loaded = True
                    st.sidebar.success("Data loaded successfully!")
            except Exception as e:
                st.sidebar.error(f"Error loading data: {str(e)}")
    
    # Page routing
    if page == "📊 Data Overview":
        if st.session_state.data_loaded:
            create_data_overview_visualizations(st.session_state.processed_data['df_full'])
        else:
            st.warning("Loading data...")
    
    elif page == "🔧 Preprocessing":
        if st.session_state.data_loaded:
            create_preprocessing_visualizations(
                st.session_state.processed_data['preprocessor'],
                st.session_state.processed_data['df_full']
            )
        else:
            st.warning("Please wait for data to load.")
    
    elif page == "🤖 Model Training":
        train_models_section()
    
    elif page == "💰 Price Prediction":
        price_prediction_section()
    
    elif page == "🚨 Price Alert Bot":
        price_alert_section()
    
    elif page == "⛽ Fuel Impact Simulation":
        fuel_simulation_section()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <strong>Dynamic Airline Price Forecasting System</strong><br>
        Built with Streamlit, Scikit-learn, XGBoost, TensorFlow, and Qiskit<br>
        🚀 Powered by Advanced ML and Quantum Computing
    </div>
    """, unsafe_allow_html=True)

# Helper functions
def create_prediction_features(airline, from_city, to_city, departure_date, departure_time, 
                             flight_class, stops, duration, fuel_price):
    """Create feature vector for prediction"""
    # This is a simplified version - in production, use the preprocessor
    dt = pd.to_datetime(departure_date)
    features = {
        'year': dt.year,
        'month': dt.month,
        'day_of_week': dt.dayofweek,
        'is_weekend': int(dt.dayofweek >= 5),
        'fuel_price_usd_per_barrel': fuel_price,
        'fuel_price_inr_per_liter': fuel_price * 0.53,
        'is_holiday': 0,  # Simplified
        'holiday_impact_score': 0,
        'is_holiday_season': 0,
        'from_major_city': int(from_city in ['Delhi', 'Mumbai', 'Bangalore', 'Chennai']),
        'to_major_city': int(to_city in ['Delhi', 'Mumbai', 'Bangalore', 'Chennai']),
        'major_to_major': int(from_city in ['Delhi', 'Mumbai', 'Bangalore', 'Chennai'] and 
                             to_city in ['Delhi', 'Mumbai', 'Bangalore', 'Chennai']),
        'route_popularity': 1000,
        'dep_time_slot': {'Early Morning': 1, 'Morning': 2, 'Afternoon': 3, 'Evening': 4, 'Night': 5}.get(departure_time, 2),
        'arr_time_slot': 4,
        'is_peak_departure': int(departure_time in ['Morning', 'Evening']),
        'flight_hours': duration,
        'total_flight_minutes': duration * 60,
        'airline_market_share': 5000
    }
    return features

def create_simulation_features(airline, route, travel_date, fuel_price):
    """Create features for fuel simulation"""
    from_city, to_city = route.split('-')
    return create_prediction_features(
        airline, from_city, to_city, travel_date, 'Morning', 'Economy', 'zero', 2.5, fuel_price
    )

def calculate_prediction_confidence(predictions):
    """Calculate confidence score based on prediction variance"""
    if len(predictions) < 2:
        return 0.5
    
    values = list(predictions.values())
    cv = np.std(values) / np.mean(values)  # Coefficient of variation
    
    # Lower CV = higher confidence
    confidence = max(0, 1 - cv)
    return min(1, confidence)

def generate_price_recommendation(price, confidence):
    """Generate price recommendation based on prediction"""
    if confidence > 0.8:
        return f"High confidence prediction. Expected price: ₹{price:,.0f}"
    elif confidence > 0.6:
        return f"Moderate confidence. Price estimate: ₹{price:,.0f} (±10%)"
    else:
        return f"Low confidence. Price range: ₹{price*0.85:,.0f} - ₹{price*1.15:,.0f}"

if __name__ == "__main__":
    main()