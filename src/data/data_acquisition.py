"""
Data acquisition module for fuel prices and holiday data
"""
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import holidays
import json
import os
from typing import Dict, List, Optional
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FuelDataCollector:
    """Collects and processes fuel price data"""
    
    def __init__(self, cache_dir: str = "data/processed"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def get_synthetic_fuel_data(self, start_date: str = "2022-01-01", 
                              end_date: str = "2024-12-31") -> pd.DataFrame:
        """
        Generate synthetic fuel price data based on realistic patterns
        Since IATA fuel data requires subscription, we'll create realistic synthetic data
        """
        cache_file = self.cache_dir / "fuel_prices.csv"
        
        if cache_file.exists():
            logger.info("Loading cached fuel data")
            return pd.read_csv(cache_file, parse_dates=['date'])
        
        logger.info("Generating synthetic fuel price data")
        
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Base price around $80-120 per barrel with realistic fluctuations
        base_price = 100
        np.random.seed(42)  # For reproducibility
        
        # Generate realistic fuel price patterns
        fuel_prices = []
        for i, date in enumerate(date_range):
            # Seasonal pattern (higher in summer)
            seasonal_factor = 1 + 0.1 * np.sin(2 * np.pi * date.dayofyear / 365)
            
            # Random volatility
            volatility = np.random.normal(0, 0.02)
            
            # Trend component (slight increase over time)
            trend = 0.0001 * i
            
            # Crisis simulation (COVID-like event in 2020, Russia-Ukraine in 2022)
            crisis_factor = 1.0
            if date.year == 2020 and 3 <= date.month <= 6:
                crisis_factor = 0.7  # COVID oil crash
            elif date.year == 2022 and date.month >= 3:
                crisis_factor = 1.3  # Geopolitical tensions
            
            price = base_price * seasonal_factor * (1 + volatility) * (1 + trend) * crisis_factor
            fuel_prices.append({
                'date': date,
                'fuel_price_usd_per_barrel': round(price, 2),
                'fuel_price_inr_per_liter': round(price * 0.53, 2),  # Approximate conversion
                'price_change_pct': round(volatility * 100, 2)
            })
        
        fuel_df = pd.DataFrame(fuel_prices)
        fuel_df.to_csv(cache_file, index=False)
        logger.info(f"Fuel data cached to {cache_file}")
        
        return fuel_df
    
    def get_fuel_price_for_date(self, date: str) -> float:
        """Get fuel price for a specific date"""
        fuel_data = self.get_synthetic_fuel_data()
        target_date = pd.to_datetime(date)
        
        # Find closest date
        closest_idx = (fuel_data['date'] - target_date).abs().idxmin()
        return fuel_data.iloc[closest_idx]['fuel_price_usd_per_barrel']

class HolidayDataCollector:
    """Collects and processes holiday data for India"""
    
    def __init__(self, cache_dir: str = "data/processed"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def get_indian_holidays(self, years: List[int] = None) -> pd.DataFrame:
        """Get Indian holiday data for specified years"""
        if years is None:
            years = list(range(2022, 2025))
            
        cache_file = self.cache_dir / "indian_holidays.csv"
        
        if cache_file.exists():
            logger.info("Loading cached holiday data")
            return pd.read_csv(cache_file, parse_dates=['date'])
        
        logger.info("Generating Indian holiday data")
        
        all_holidays = []
        
        for year in years:
            # Get Indian holidays
            india_holidays = holidays.India(years=year)
            
            # Add additional Indian holidays not in the library
            additional_holidays = self._get_additional_indian_holidays(year)
            
            for date, name in india_holidays.items():
                all_holidays.append({
                    'date': date,
                    'holiday_name': name,
                    'is_national_holiday': True,
                    'holiday_type': 'National',
                    'year': year
                })
            
            for date, name in additional_holidays.items():
                all_holidays.append({
                    'date': date,
                    'holiday_name': name,
                    'is_national_holiday': False,
                    'holiday_type': 'Regional/Festival',
                    'year': year
                })
        
        holiday_df = pd.DataFrame(all_holidays)
        
        # Add holiday impact scores (how much they affect travel demand)
        holiday_df['holiday_impact_score'] = holiday_df['holiday_name'].map(
            self._get_holiday_impact_scores()
        ).fillna(3)  # Default medium impact
        
        holiday_df.to_csv(cache_file, index=False)
        logger.info(f"Holiday data cached to {cache_file}")
        
        return holiday_df
    
    def _get_additional_indian_holidays(self, year: int) -> Dict:
        """Get additional Indian holidays not in the standard library"""
        # These are approximate dates - in real implementation, use proper calendar
        additional = {}
        
        # Add some major festivals (approximate dates)
        if year == 2022:
            additional.update({
                datetime(2022, 3, 18): "Holi",
                datetime(2022, 4, 14): "Baisakhi",
                datetime(2022, 8, 19): "Janmashtami",
                datetime(2022, 9, 10): "Ganesh Chaturthi",
                datetime(2022, 10, 5): "Dussehra",
                datetime(2022, 10, 24): "Diwali",
                datetime(2022, 11, 8): "Guru Nanak Jayanti"
            })
        elif year == 2023:
            additional.update({
                datetime(2023, 3, 8): "Holi",
                datetime(2023, 4, 14): "Baisakhi",
                datetime(2023, 9, 7): "Janmashtami",
                datetime(2023, 9, 19): "Ganesh Chaturthi",
                datetime(2023, 10, 24): "Dussehra",
                datetime(2023, 11, 12): "Diwali",
                datetime(2023, 11, 27): "Guru Nanak Jayanti"
            })
        elif year == 2024:
            additional.update({
                datetime(2024, 3, 25): "Holi",
                datetime(2024, 4, 14): "Baisakhi",
                datetime(2024, 8, 26): "Janmashtami",
                datetime(2024, 9, 7): "Ganesh Chaturthi",
                datetime(2024, 10, 12): "Dussehra",
                datetime(2024, 11, 1): "Diwali",
                datetime(2024, 11, 15): "Guru Nanak Jayanti"  
            })
            
        return additional
    
    def _get_holiday_impact_scores(self) -> Dict[str, int]:
        """Get impact scores for different holidays (1-5 scale)"""
        return {
            'Diwali': 5,
            'Holi': 4,
            'Dussehra': 4,
            'Christmas Day': 5,
            'New Year\'s Day': 4,
            'Independence Day': 3,
            'Republic Day': 3,
            'Gandhi Jayanti': 3,
            'Eid al-Fitr': 4,
            'Eid al-Adha': 4,
            'Janmashtami': 3,
            'Ganesh Chaturthi': 3,
            'Baisakhi': 3,
            'Guru Nanak Jayanti': 2,
            'Good Friday': 2,
            'Ram Navami': 2,
            'Maha Shivratri': 2
        }
    
    def is_holiday(self, date: str) -> tuple:
        """Check if a date is a holiday and return impact score"""
        holiday_data = self.get_indian_holidays()
        target_date = pd.to_datetime(date).date()
        
        holiday_row = holiday_data[holiday_data['date'].dt.date == target_date]
        
        if not holiday_row.empty:
            return True, holiday_row.iloc[0]['holiday_impact_score'], holiday_row.iloc[0]['holiday_name']
        return False, 0, None
    
    def get_holiday_season_info(self, date: str) -> Dict:
        """Get information about holiday season (before/after major holidays)"""
        target_date = pd.to_datetime(date)
        holiday_data = self.get_indian_holidays()
        
        # Check for holidays within ±7 days
        date_range = pd.date_range(target_date - timedelta(days=7), 
                                 target_date + timedelta(days=7))
        
        nearby_holidays = holiday_data[holiday_data['date'].isin(date_range)]
        
        if not nearby_holidays.empty:
            holiday_info = nearby_holidays.iloc[0]
            days_to_holiday = (holiday_info['date'] - target_date).days
            
            return {
                'is_holiday_season': True,
                'days_to_holiday': days_to_holiday,
                'holiday_name': holiday_info['holiday_name'],
                'holiday_impact': holiday_info['holiday_impact_score'],
                'season_type': 'pre_holiday' if days_to_holiday > 0 else 'post_holiday'
            }
        
        return {'is_holiday_season': False}

class DataAcquisitionManager:
    """Main class to manage all data acquisition"""
    
    def __init__(self, cache_dir: str = "data/processed"):
        self.fuel_collector = FuelDataCollector(cache_dir)
        self.holiday_collector = HolidayDataCollector(cache_dir)
        
    def get_all_external_data(self) -> Dict[str, pd.DataFrame]:
        """Get all external data sources"""
        logger.info("Collecting all external data...")
        
        return {
            'fuel_data': self.fuel_collector.get_synthetic_fuel_data(),
            'holiday_data': self.holiday_collector.get_indian_holidays()
        }
    
    def enrich_flight_data(self, flight_df: pd.DataFrame, 
                          date_column: str = 'date') -> pd.DataFrame:
        """Enrich flight data with fuel prices and holiday information"""
        logger.info("Enriching flight data with external sources...")
        
        enriched_df = flight_df.copy()
        
        # Add fuel price information
        fuel_data = self.fuel_collector.get_synthetic_fuel_data()
        
        # Convert date column to datetime if not already
        if date_column in enriched_df.columns:
            enriched_df[date_column] = pd.to_datetime(enriched_df[date_column])
            
            # Merge with fuel data
            enriched_df = enriched_df.merge(
                fuel_data[['date', 'fuel_price_usd_per_barrel', 'fuel_price_inr_per_liter']],
                left_on=date_column, right_on='date', how='left'
            ).drop('date', axis=1)
        
        # Add holiday information
        holiday_data = self.holiday_collector.get_indian_holidays()
        
        # Create holiday features
        if date_column in enriched_df.columns:
            enriched_df['is_holiday'] = False
            enriched_df['holiday_impact_score'] = 0
            enriched_df['is_holiday_season'] = False
            
            for idx, row in enriched_df.iterrows():
                is_hol, impact, name = self.holiday_collector.is_holiday(str(row[date_column].date()))
                season_info = self.holiday_collector.get_holiday_season_info(str(row[date_column].date()))
                
                enriched_df.at[idx, 'is_holiday'] = is_hol
                enriched_df.at[idx, 'holiday_impact_score'] = impact
                enriched_df.at[idx, 'is_holiday_season'] = season_info['is_holiday_season']
        
        logger.info(f"Enriched flight data with {len(enriched_df.columns) - len(flight_df.columns)} new features")
        return enriched_df

if __name__ == "__main__":
    # Test the data acquisition
    manager = DataAcquisitionManager()
    external_data = manager.get_all_external_data()
    
    print("Fuel data shape:", external_data['fuel_data'].shape)
    print("Holiday data shape:", external_data['holiday_data'].shape)
    print("\nFuel data sample:")
    print(external_data['fuel_data'].head())
    print("\nHoliday data sample:")
    print(external_data['holiday_data'].head())