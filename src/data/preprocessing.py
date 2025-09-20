"""
Data preprocessing pipeline for airline price prediction
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta
import re
import logging
from typing import Tuple, Dict, List, Optional
from pathlib import Path
import joblib

from .data_acquisition import DataAcquisitionManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FlightDataPreprocessor:
    """Comprehensive flight data preprocessing pipeline"""
    
    def __init__(self, cache_dir: str = "data/processed"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.label_encoders = {}
        self.scaler = None
        self.feature_columns = []
        self.target_column = 'price'
        
        self.data_manager = DataAcquisitionManager(cache_dir)
        
    def load_airline_data(self, data_path: str = "Kaggle Airline Dataset") -> pd.DataFrame:
        """Load and combine all airline datasets"""
        logger.info("Loading airline datasets...")
        
        # Load all three datasets
        clean_df = pd.read_csv(f"{data_path}/Clean_Dataset.csv")
        business_df = pd.read_csv(f"{data_path}/business.csv")
        economy_df = pd.read_csv(f"{data_path}/economy.csv")
        
        logger.info(f"Loaded datasets - Clean: {clean_df.shape}, Business: {business_df.shape}, Economy: {economy_df.shape}")
        
        # Standardize the datasets
        clean_df = self._standardize_clean_dataset(clean_df)
        business_df = self._standardize_business_dataset(business_df)
        economy_df = self._standardize_economy_dataset(economy_df)
        
        # Combine all datasets
        combined_df = pd.concat([clean_df, business_df, economy_df], ignore_index=True)
        
        logger.info(f"Combined dataset shape: {combined_df.shape}")
        return combined_df
    
    def _standardize_clean_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize the clean dataset format"""
        df = df.copy()
        
        # Create date column (assuming this is recent data)
        if 'days_left' in df.columns:
            base_date = datetime(2024, 1, 1)
            df['date'] = base_date + pd.to_timedelta(df['days_left'], unit='D')
        else:
            df['date'] = datetime(2024, 1, 1)  # Default date
            
        # Standardize column names
        column_mapping = {
            'source_city': 'from',
            'destination_city': 'to',
            'departure_time': 'dep_time',
            'arrival_time': 'arr_time',
            'stops': 'stop'
        }
        
        df = df.rename(columns=column_mapping)
        
        # Add missing columns with defaults
        if 'ch_code' not in df.columns:
            df['ch_code'] = df['airline'].str[:2]
        if 'num_code' not in df.columns:
            df['num_code'] = df['flight'].str.extract(r'(\d+)').fillna('000')
        if 'time_taken' not in df.columns:
            df['time_taken'] = df['duration'].apply(self._convert_duration_to_standard)
            
        return df
    
    def _standardize_business_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize the business dataset format"""
        df = df.copy()
        
        # Convert date format
        if 'ggldate' in df.columns:
            df['date'] = pd.to_datetime(df['ggldate'], format='%d-%m-%Y', errors='coerce')
        
        # Clean price column
        if 'price' in df.columns:
            df['price'] = df['price'].astype(str).str.replace(',', '').str.replace('"', '')
            df['price'] = pd.to_numeric(df['price'], errors='coerce')
        
        # Add class column
        df['class'] = 'Business'
        
        return df
    
    def _standardize_economy_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize the economy dataset format"""
        df = df.copy()
        
        # Convert date format
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y', errors='coerce')
        
        # Clean price column
        if 'price' in df.columns:
            df['price'] = df['price'].astype(str).str.replace(',', '').str.replace('"', '')
            df['price'] = pd.to_numeric(df['price'], errors='coerce')
        
        # Add class column if not present
        if 'class' not in df.columns:
            df['class'] = 'Economy'
        
        return df
    
    def _convert_duration_to_standard(self, duration) -> str:
        """Convert duration to standard format (Xh Ym)"""
        if pd.isna(duration):
            return "2h 00m"
        
        duration_str = str(duration)
        
        # If already in decimal format (like 2.17), convert to hours and minutes
        try:
            decimal_hours = float(duration_str)
            hours = int(decimal_hours)
            minutes = int((decimal_hours - hours) * 60)
            return f"{hours}h {minutes:02d}m"
        except:
            # If already in standard format, return as is
            return duration_str
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Comprehensive data cleaning"""
        logger.info("Starting data cleaning...")
        
        df_clean = df.copy()
        
        # Remove duplicates
        initial_shape = df_clean.shape
        df_clean = df_clean.drop_duplicates()
        logger.info(f"Removed {initial_shape[0] - df_clean.shape[0]} duplicate rows")
        
        # Handle missing values
        df_clean = self._handle_missing_values(df_clean)
        
        # Clean price column
        df_clean = self._clean_price_column(df_clean)
        
        # Clean time columns
        df_clean = self._clean_time_columns(df_clean)
        
        # Remove outliers
        df_clean = self._remove_outliers(df_clean)
        
        logger.info(f"Data cleaning completed. Final shape: {df_clean.shape}")
        return df_clean
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset"""
        df = df.copy()
        
        # Fill missing categorical values
        categorical_columns = ['airline', 'from', 'to', 'class', 'stop']
        for col in categorical_columns:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
        
        # Fill missing numerical values
        numerical_columns = ['price', 'duration']
        for col in numerical_columns:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())
        
        # Fill missing time values
        time_columns = ['dep_time', 'arr_time']
        for col in time_columns:
            if col in df.columns:
                df[col] = df[col].fillna('Unknown')
        
        return df
    
    def _clean_price_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean the price column"""
        df = df.copy()
        
        if 'price' in df.columns:
            # Remove commas, quotes, and currency symbols
            df['price'] = df['price'].astype(str)
            df['price'] = df['price'].str.replace(r'[^\d.]', '', regex=True)
            df['price'] = pd.to_numeric(df['price'], errors='coerce')
            
            # Remove rows with invalid prices
            df = df[df['price'] > 0]
            
            # Convert to INR if needed (assuming some prices might be in other currencies)
            # This is a simple heuristic - prices less than 1000 might be in thousands
            df.loc[df['price'] < 1000, 'price'] *= 1000
        
        return df
    
    def _clean_time_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean time-related columns"""
        df = df.copy()
        
        # Standardize time format
        time_mappings = {
            'Early_Morning': 'Early Morning',
            'Late_Night': 'Night',
            'early_morning': 'Early Morning',
            'late_night': 'Night'
        }
        
        for col in ['dep_time', 'arr_time']:
            if col in df.columns:
                df[col] = df[col].replace(time_mappings)
        
        return df
    
    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove outliers using IQR method"""
        df = df.copy()
        
        if 'price' in df.columns:
            Q1 = df['price'].quantile(0.25)
            Q3 = df['price'].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            initial_count = len(df)
            df = df[(df['price'] >= lower_bound) & (df['price'] <= upper_bound)]
            logger.info(f"Removed {initial_count - len(df)} price outliers")
        
        return df
    
    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Comprehensive feature engineering"""
        logger.info("Starting feature engineering...")
        
        df_features = df.copy()
        
        # Temporal features
        df_features = self._create_temporal_features(df_features)
        
        # Route features
        df_features = self._create_route_features(df_features)
        
        # Time-based features
        df_features = self._create_time_features(df_features)
        
        # Duration features
        df_features = self._create_duration_features(df_features)
        
        # Airline features
        df_features = self._create_airline_features(df_features)
        
        # External data features (fuel prices, holidays)
        df_features = self._add_external_features(df_features)
        
        logger.info(f"Feature engineering completed. New shape: {df_features.shape}")
        return df_features
    
    def _create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create temporal features from date"""
        df = df.copy()
        
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            
            df['year'] = df['date'].dt.year
            df['month'] = df['date'].dt.month
            df['quarter'] = df['date'].dt.quarter
            df['day'] = df['date'].dt.day
            df['day_of_week'] = df['date'].dt.dayofweek
            df['day_of_year'] = df['date'].dt.dayofyear
            df['week_of_year'] = df['date'].dt.isocalendar().week
            
            # Seasonal features
            df['season'] = df['month'].map({
                12: 'Winter', 1: 'Winter', 2: 'Winter',
                3: 'Spring', 4: 'Spring', 5: 'Spring',
                6: 'Summer', 7: 'Summer', 8: 'Summer',
                9: 'Autumn', 10: 'Autumn', 11: 'Autumn'
            })
            
            # Weekend indicator
            df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
            
            # Month-end indicator
            df['is_month_end'] = (df['day'] >= 25).astype(int)
            
        return df
    
    def _create_route_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create route-based features"""
        df = df.copy()
        
        if 'from' in df.columns and 'to' in df.columns:
            # Route popularity
            route_counts = df.groupby(['from', 'to']).size()
            df['route_popularity'] = df.apply(lambda x: route_counts.get((x['from'], x['to']), 0), axis=1)
            
            # Major cities indicator
            major_cities = ['Delhi', 'Mumbai', 'Bangalore', 'Chennai', 'Kolkata', 'Hyderabad']
            df['from_major_city'] = df['from'].isin(major_cities).astype(int)
            df['to_major_city'] = df['to'].isin(major_cities).astype(int)
            df['major_to_major'] = (df['from_major_city'] & df['to_major_city']).astype(int)
            
            # Route type
            df['route_type'] = 'Domestic'  # Assuming all routes are domestic
        
        return df
    
    def _create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features"""
        df = df.copy()
        
        # Time slot encoding
        time_slot_mapping = {
            'Early Morning': 1, 'Morning': 2, 'Afternoon': 3, 
            'Evening': 4, 'Night': 5, 'Late Night': 6
        }
        
        if 'dep_time' in df.columns:
            df['dep_time_slot'] = df['dep_time'].map(time_slot_mapping).fillna(3)
        
        if 'arr_time' in df.columns:
            df['arr_time_slot'] = df['arr_time'].map(time_slot_mapping).fillna(3)
        
        # Peak time indicators
        peak_times = ['Morning', 'Evening']
        if 'dep_time' in df.columns:
            df['is_peak_departure'] = df['dep_time'].isin(peak_times).astype(int)
        
        return df
    
    def _create_duration_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create duration-based features"""
        df = df.copy()
        
        if 'time_taken' in df.columns:
            # Extract hours and minutes from duration
            duration_pattern = r'(\d+)h\s*(\d+)m'
            duration_matches = df['time_taken'].str.extract(duration_pattern)
            
            df['flight_hours'] = pd.to_numeric(duration_matches[0], errors='coerce').fillna(2)
            df['flight_minutes'] = pd.to_numeric(duration_matches[1], errors='coerce').fillna(0)
            df['total_flight_minutes'] = df['flight_hours'] * 60 + df['flight_minutes']
            
            # Duration categories
            df['duration_category'] = pd.cut(df['total_flight_minutes'], 
                                           bins=[0, 120, 240, 480, float('inf')],
                                           labels=['Short', 'Medium', 'Long', 'Very Long'])
        
        elif 'duration' in df.columns:
            # Handle decimal duration format
            df['flight_hours'] = df['duration'].fillna(2)
            df['total_flight_minutes'] = df['flight_hours'] * 60
            df['duration_category'] = pd.cut(df['total_flight_minutes'], 
                                           bins=[0, 120, 240, 480, float('inf')],
                                           labels=['Short', 'Medium', 'Long', 'Very Long'])
        
        return df
    
    def _create_airline_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create airline-based features"""
        df = df.copy()
        
        if 'airline' in df.columns:
            # Airline type categorization
            budget_airlines = ['SpiceJet', 'IndiGo', 'GO_FIRST', 'AirAsia']
            premium_airlines = ['Vistara', 'Air India']
            
            df['airline_type'] = 'Other'
            df.loc[df['airline'].isin(budget_airlines), 'airline_type'] = 'Budget'
            df.loc[df['airline'].isin(premium_airlines), 'airline_type'] = 'Premium'
            
            # Airline market share (based on frequency in dataset)
            airline_counts = df['airline'].value_counts()
            df['airline_market_share'] = df['airline'].map(airline_counts)
        
        return df
    
    def _add_external_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add external features (fuel prices, holidays)"""
        df = df.copy()
        
        if 'date' in df.columns:
            logger.info("Adding external features...")
            df = self.data_manager.enrich_flight_data(df, 'date')
        
        return df
    
    def encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features"""
        logger.info("Encoding categorical features...")
        
        df_encoded = df.copy()
        
        categorical_columns = [
            'airline', 'from', 'to', 'class', 'stop', 'dep_time', 'arr_time',
            'season', 'duration_category', 'airline_type', 'route_type'
        ]
        
        for col in categorical_columns:
            if col in df_encoded.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    df_encoded[col + '_encoded'] = self.label_encoders[col].fit_transform(df_encoded[col].astype(str))
                else:
                    # Handle unseen categories
                    unique_values = df_encoded[col].astype(str).unique()
                    known_values = set(self.label_encoders[col].classes_)
                    
                    for value in unique_values:
                        if value not in known_values:
                            # Add new category
                            self.label_encoders[col].classes_ = np.append(self.label_encoders[col].classes_, value)
                    
                    df_encoded[col + '_encoded'] = self.label_encoders[col].transform(df_encoded[col].astype(str))
        
        return df_encoded
    
    def prepare_features_for_modeling(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Prepare final feature set for modeling"""
        logger.info("Preparing features for modeling...")
        
        # Select features for modeling
        feature_columns = []
        
        # Numerical features
        numerical_features = [
            'year', 'month', 'quarter', 'day', 'day_of_week', 'day_of_year', 'week_of_year',
            'is_weekend', 'is_month_end', 'route_popularity', 'from_major_city', 'to_major_city',
            'major_to_major', 'dep_time_slot', 'arr_time_slot', 'is_peak_departure',
            'flight_hours', 'flight_minutes', 'total_flight_minutes', 'airline_market_share',
            'fuel_price_usd_per_barrel', 'fuel_price_inr_per_liter', 'is_holiday',
            'holiday_impact_score', 'is_holiday_season'
        ]
        
        # Encoded categorical features
        encoded_features = [col for col in df.columns if col.endswith('_encoded')]
        
        # Combine all features
        all_potential_features = numerical_features + encoded_features
        
        # Select only existing features
        for feature in all_potential_features:
            if feature in df.columns:
                feature_columns.append(feature)
        
        self.feature_columns = feature_columns
        
        # Create feature matrix
        X = df[feature_columns].copy()
        
        # Handle any remaining missing values
        X = X.fillna(X.median())
        
        logger.info(f"Selected {len(feature_columns)} features for modeling")
        return X, feature_columns
    
    def scale_features(self, X: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Scale features using RobustScaler"""
        if fit:
            self.scaler = RobustScaler()
            X_scaled = pd.DataFrame(
                self.scaler.fit_transform(X),
                columns=X.columns,
                index=X.index
            )
            logger.info("Fitted and transformed features")
        else:
            if self.scaler is None:
                raise ValueError("Scaler not fitted. Call with fit=True first.")
            X_scaled = pd.DataFrame(
                self.scaler.transform(X),
                columns=X.columns,
                index=X.index
            )
            logger.info("Transformed features using existing scaler")
        
        return X_scaled
    
    def split_data(self, X: pd.DataFrame, y: pd.Series, 
                   test_size: float = 0.2, val_size: float = 0.2,
                   random_state: int = 42) -> Tuple:
        """Split data into train, validation, and test sets"""
        
        # First split: train+val and test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=None
        )
        
        # Second split: train and val
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state
        )
        
        logger.info(f"Data split - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def save_preprocessor(self, filepath: str = "models/saved/preprocessor.pkl"):
        """Save the preprocessor state"""
        preprocessor_state = {
            'label_encoders': self.label_encoders,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'target_column': self.target_column
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(preprocessor_state, filepath)
        logger.info(f"Preprocessor saved to {filepath}")
    
    def load_preprocessor(self, filepath: str = "models/saved/preprocessor.pkl"):
        """Load the preprocessor state"""
        preprocessor_state = joblib.load(filepath)
        
        self.label_encoders = preprocessor_state['label_encoders']
        self.scaler = preprocessor_state['scaler']
        self.feature_columns = preprocessor_state['feature_columns']
        self.target_column = preprocessor_state['target_column']
        
        logger.info(f"Preprocessor loaded from {filepath}")
    
    def process_full_pipeline(self, data_path: str = "Kaggle Airline Dataset") -> Tuple:
        """Run the complete preprocessing pipeline"""
        logger.info("Starting full preprocessing pipeline...")
        
        # Load data
        df = self.load_airline_data(data_path)
        
        # Clean data
        df_clean = self.clean_data(df)
        
        # Feature engineering
        df_features = self.feature_engineering(df_clean)
        
        # Encode categorical features
        df_encoded = self.encode_categorical_features(df_features)
        
        # Prepare features for modeling
        X, feature_columns = self.prepare_features_for_modeling(df_encoded)
        
        # Scale features
        X_scaled = self.scale_features(X, fit=True)
        
        # Extract target variable
        y = df_encoded[self.target_column]
        
        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X_scaled, y)
        
        # Save preprocessor
        self.save_preprocessor()
        
        # Save processed data
        processed_data = {
            'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
            'y_train': y_train, 'y_val': y_val, 'y_test': y_test,
            'feature_columns': feature_columns,
            'full_dataset': df_encoded
        }
        
        joblib.dump(processed_data, self.cache_dir / "processed_data.pkl")
        logger.info("Processed data saved")
        
        logger.info("Full preprocessing pipeline completed successfully!")
        
        return X_train, X_val, X_test, y_train, y_val, y_test, df_encoded

if __name__ == "__main__":
    # Test the preprocessing pipeline
    preprocessor = FlightDataPreprocessor()
    X_train, X_val, X_test, y_train, y_val, y_test, df_full = preprocessor.process_full_pipeline()
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Feature columns: {len(preprocessor.feature_columns)}")
    print(f"Full dataset shape: {df_full.shape}")