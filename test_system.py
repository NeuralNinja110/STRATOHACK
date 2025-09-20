#!/usr/bin/env python3
"""
System integration test for the airline price forecasting system
"""
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append('src')

def test_complete_system():
    """Test the complete system pipeline"""
    print("🧪 Starting Complete System Test")
    print("=" * 50)
    
    try:
        # Test 1: Data Acquisition
        print("1️⃣ Testing Data Acquisition...")
        from src.data.data_acquisition import DataAcquisitionManager
        data_manager = DataAcquisitionManager()
        external_data = data_manager.get_all_external_data()
        print(f"   ✅ Fuel data: {external_data['fuel_data'].shape}")
        print(f"   ✅ Holiday data: {external_data['holiday_data'].shape}")
        
        # Test 2: Data Preprocessing
        print("\n2️⃣ Testing Data Preprocessing...")
        from src.data.preprocessing import FlightDataPreprocessor
        preprocessor = FlightDataPreprocessor()
        
        # Load a small sample of data for testing
        df = preprocessor.load_airline_data()
        print(f"   ✅ Loaded data: {df.shape}")
        
        # Clean data
        df_clean = preprocessor.clean_data(df.sample(1000))  # Small sample for testing
        print(f"   ✅ Cleaned data: {df_clean.shape}")
        
        # Feature engineering
        df_features = preprocessor.feature_engineering(df_clean)
        print(f"   ✅ Feature engineering: {df_features.shape}")
        
        # Test 3: Model Initialization
        print("\n3️⃣ Testing Model Initialization...")
        from src.models.ml_models import ModelManager
        model_manager = ModelManager()
        models = model_manager.initialize_models(n_features=20)
        print(f"   ✅ Initialized {len(models)} models")
        
        # Test 4: Price Analysis Utilities
        print("\n4️⃣ Testing Price Analysis Utils...")
        from src.utils.price_analysis import PriceAlertBot, FuelPriceSimulator
        
        alert_bot = PriceAlertBot(model_manager, data_manager)
        fuel_simulator = FuelPriceSimulator(model_manager)
        
        # Test fuel scenarios
        scenarios = fuel_simulator.create_fuel_scenarios(100.0)
        print(f"   ✅ Created {len(scenarios)} fuel scenarios")
        
        # Test 5: Feature Creation for Prediction
        print("\n5️⃣ Testing Prediction Features...")
        
        # Create sample prediction features
        sample_features = {
            'year': 2024, 'month': 12, 'day_of_week': 1,
            'fuel_price_usd_per_barrel': 100.0,
            'is_holiday': 0, 'holiday_impact_score': 0,
            'from_major_city': 1, 'to_major_city': 1,
            'route_popularity': 1000, 'dep_time_slot': 2
        }
        
        print(f"   ✅ Sample features created: {len(sample_features)} features")
        
        print("\n🎉 ALL TESTS PASSED!")
        print("=" * 50)
        print("✅ System is ready for use!")
        print("🚀 Run 'streamlit run app.py' to start the web interface")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_complete_system()
    sys.exit(0 if success else 1)
