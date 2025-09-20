"""
Price Alert Bot and Fuel Price Impact Simulation
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path

from ..data.data_acquisition import DataAcquisitionManager
from ..models.ml_models import ModelManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PriceAlertBot:
    """AI-powered price alert system"""
    
    def __init__(self, model_manager: ModelManager, data_manager: DataAcquisitionManager):
        self.model_manager = model_manager
        self.data_manager = data_manager
        self.historical_patterns = {}
        
    def analyze_historical_patterns(self, df: pd.DataFrame, route: str) -> Dict:
        """Analyze historical price patterns for a specific route"""
        logger.info(f"Analyzing historical patterns for route: {route}")
        
        route_data = df[df['route'] == route].copy() if 'route' in df.columns else df.copy()
        
        if route_data.empty:
            return {}
        
        patterns = {
            'seasonal_trends': self._analyze_seasonal_trends(route_data),
            'weekly_patterns': self._analyze_weekly_patterns(route_data),
            'booking_window_analysis': self._analyze_booking_window(route_data),
            'holiday_impact': self._analyze_holiday_impact(route_data),
            'fuel_price_correlation': self._analyze_fuel_correlation(route_data)
        }
        
        self.historical_patterns[route] = patterns
        return patterns
    
    def _analyze_seasonal_trends(self, df: pd.DataFrame) -> Dict:
        """Analyze seasonal price trends"""
        if 'date' not in df.columns or df.empty:
            return {}
        
        df['month'] = pd.to_datetime(df['date']).dt.month
        monthly_avg = df.groupby('month')['price'].agg(['mean', 'std', 'count']).reset_index()
        
        # Find peak and off-peak seasons
        peak_months = monthly_avg.nlargest(3, 'mean')['month'].tolist()
        off_peak_months = monthly_avg.nsmallest(3, 'mean')['month'].tolist()
        
        return {
            'monthly_averages': monthly_avg.to_dict('records'),
            'peak_months': peak_months,
            'off_peak_months': off_peak_months,
            'seasonal_variation': (monthly_avg['mean'].max() - monthly_avg['mean'].min()) / monthly_avg['mean'].mean()
        }
    
    def _analyze_weekly_patterns(self, df: pd.DataFrame) -> Dict:
        """Analyze weekly price patterns"""
        if 'date' not in df.columns or df.empty:
            return {}
        
        df['day_of_week'] = pd.to_datetime(df['date']).dt.day_name()
        weekly_avg = df.groupby('day_of_week')['price'].agg(['mean', 'std', 'count']).reset_index()
        
        # Reorder by day of week
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        weekly_avg['day_of_week'] = pd.Categorical(weekly_avg['day_of_week'], categories=day_order, ordered=True)
        weekly_avg = weekly_avg.sort_values('day_of_week')
        
        cheapest_days = weekly_avg.nsmallest(2, 'mean')['day_of_week'].tolist()
        expensive_days = weekly_avg.nlargest(2, 'mean')['day_of_week'].tolist()
        
        return {
            'weekly_averages': weekly_avg.to_dict('records'),
            'cheapest_days': cheapest_days,
            'most_expensive_days': expensive_days
        }
    
    def _analyze_booking_window(self, df: pd.DataFrame) -> Dict:
        """Analyze optimal booking window"""
        if 'days_left' not in df.columns or df.empty:
            return {}
        
        # Group by booking window
        booking_windows = pd.cut(df['days_left'], 
                               bins=[0, 7, 14, 30, 60, 90, float('inf')],
                               labels=['0-7 days', '8-14 days', '15-30 days', '31-60 days', '61-90 days', '90+ days'])
        
        window_analysis = df.groupby(booking_windows)['price'].agg(['mean', 'std', 'count']).reset_index()
        
        optimal_window = window_analysis.loc[window_analysis['mean'].idxmin(), 'days_left']
        
        return {
            'booking_window_analysis': window_analysis.to_dict('records'),
            'optimal_booking_window': str(optimal_window),
            'price_trend_by_window': window_analysis[['days_left', 'mean']].to_dict('records')
        }
    
    def _analyze_holiday_impact(self, df: pd.DataFrame) -> Dict:
        """Analyze holiday impact on prices"""
        if 'is_holiday' not in df.columns or df.empty:
            return {}
        
        holiday_comparison = df.groupby('is_holiday')['price'].agg(['mean', 'std', 'count']).reset_index()
        
        if len(holiday_comparison) == 2:
            price_increase = ((holiday_comparison.loc[1, 'mean'] - holiday_comparison.loc[0, 'mean']) / 
                            holiday_comparison.loc[0, 'mean']) * 100
        else:
            price_increase = 0
        
        return {
            'holiday_vs_normal': holiday_comparison.to_dict('records'),
            'holiday_price_increase_pct': price_increase,
            'holiday_season_impact': df.groupby('is_holiday_season')['price'].mean().to_dict() if 'is_holiday_season' in df.columns else {}
        }
    
    def _analyze_fuel_correlation(self, df: pd.DataFrame) -> Dict:
        """Analyze fuel price correlation"""
        if 'fuel_price_usd_per_barrel' not in df.columns or df.empty:
            return {}
        
        correlation = df['price'].corr(df['fuel_price_usd_per_barrel'])
        
        # Linear relationship analysis
        fuel_price_bins = pd.cut(df['fuel_price_usd_per_barrel'], bins=5)
        fuel_impact = df.groupby(fuel_price_bins)['price'].mean()
        
        return {
            'fuel_price_correlation': correlation,
            'fuel_price_impact': fuel_impact.to_dict(),
            'elasticity': self._calculate_price_elasticity(df)
        }
    
    def _calculate_price_elasticity(self, df: pd.DataFrame) -> float:
        """Calculate price elasticity with respect to fuel prices"""
        if len(df) < 10:
            return 0
        
        # Simple elasticity calculation
        fuel_change = (df['fuel_price_usd_per_barrel'].max() - df['fuel_price_usd_per_barrel'].min()) / df['fuel_price_usd_per_barrel'].mean()
        price_change = (df['price'].max() - df['price'].min()) / df['price'].mean()
        
        return price_change / fuel_change if fuel_change != 0 else 0
    
    def predict_optimal_booking_time(self, route: str, departure_date: str, 
                                   current_fuel_price: float = None) -> Dict:
        """Predict optimal booking time for a specific route and date"""
        logger.info(f"Predicting optimal booking time for {route} on {departure_date}")
        
        target_date = pd.to_datetime(departure_date)
        current_date = datetime.now()
        
        # Generate booking scenarios for different advance booking days
        booking_scenarios = []
        
        for days_ahead in range(1, 91, 7):  # 1 to 90 days, weekly intervals
            booking_date = target_date - timedelta(days=days_ahead)
            
            if booking_date < current_date:
                continue
            
            # Create feature vector for prediction
            features = self._create_prediction_features(
                route, str(booking_date.date()), days_ahead, current_fuel_price
            )
            
            # Get prediction from best model
            if self.model_manager.trained_models:
                best_model_name, best_model = self.model_manager.get_best_model()
                predicted_price = best_model.predict(pd.DataFrame([features]))[0]
            else:
                # Fallback to historical average with adjustments
                predicted_price = self._estimate_price_from_patterns(route, booking_date, days_ahead)
            
            booking_scenarios.append({
                'booking_date': booking_date.strftime('%Y-%m-%d'),
                'days_ahead': days_ahead,
                'predicted_price': predicted_price,
                'confidence_score': self._calculate_confidence_score(days_ahead)
            })
        
        # Find optimal booking time
        if booking_scenarios:
            optimal_scenario = min(booking_scenarios, key=lambda x: x['predicted_price'])
            
            return {
                'optimal_booking_date': optimal_scenario['booking_date'],
                'optimal_days_ahead': optimal_scenario['days_ahead'],
                'predicted_lowest_price': optimal_scenario['predicted_price'],
                'confidence_score': optimal_scenario['confidence_score'],
                'all_scenarios': booking_scenarios,
                'recommendation': self._generate_booking_recommendation(optimal_scenario, booking_scenarios)
            }
        
        return {'error': 'No valid booking scenarios found'}
    
    def _create_prediction_features(self, route: str, booking_date: str, 
                                  days_ahead: int, fuel_price: float = None) -> Dict:
        """Create feature vector for price prediction"""
        booking_dt = pd.to_datetime(booking_date)
        
        # Basic temporal features
        features = {
            'year': booking_dt.year,
            'month': booking_dt.month,
            'quarter': booking_dt.quarter,
            'day': booking_dt.day,
            'day_of_week': booking_dt.dayofweek,
            'day_of_year': booking_dt.dayofyear,
            'week_of_year': booking_dt.isocalendar().week,
            'is_weekend': int(booking_dt.dayofweek >= 5),
            'is_month_end': int(booking_dt.day >= 25),
            'days_left': days_ahead
        }
        
        # Seasonal features
        season_map = {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3}
        features['season_encoded'] = season_map.get(booking_dt.month, 0)
        
        # Holiday information
        is_holiday, holiday_impact, _ = self.data_manager.holiday_collector.is_holiday(booking_date)
        holiday_season = self.data_manager.holiday_collector.get_holiday_season_info(booking_date)
        
        features.update({
            'is_holiday': int(is_holiday),
            'holiday_impact_score': holiday_impact,
            'is_holiday_season': int(holiday_season['is_holiday_season'])
        })
        
        # Fuel price
        if fuel_price is None:
            fuel_price = self.data_manager.fuel_collector.get_fuel_price_for_date(booking_date)
        
        features['fuel_price_usd_per_barrel'] = fuel_price
        features['fuel_price_inr_per_liter'] = fuel_price * 0.53
        
        # Route-specific features (simplified)
        route_parts = route.split('-') if '-' in route else ['Delhi', 'Mumbai']
        major_cities = ['Delhi', 'Mumbai', 'Bangalore', 'Chennai', 'Kolkata', 'Hyderabad']
        
        features.update({
            'from_major_city': int(route_parts[0] in major_cities),
            'to_major_city': int(route_parts[1] in major_cities if len(route_parts) > 1 else 1),
            'major_to_major': int(len(route_parts) > 1 and route_parts[0] in major_cities and route_parts[1] in major_cities),
            'route_popularity': 1000,  # Default value
            'dep_time_slot': 2,  # Default morning
            'arr_time_slot': 4,  # Default evening
            'is_peak_departure': 1,
            'flight_hours': 2.5,
            'flight_minutes': 30,
            'total_flight_minutes': 150,
            'airline_market_share': 5000  # Default value
        })
        
        return features
    
    def _estimate_price_from_patterns(self, route: str, booking_date: pd.Timestamp, days_ahead: int) -> float:
        """Estimate price using historical patterns"""
        base_price = 7000  # Default base price
        
        if route in self.historical_patterns:
            patterns = self.historical_patterns[route]
            
            # Seasonal adjustment
            if 'seasonal_trends' in patterns and 'monthly_averages' in patterns['seasonal_trends']:
                month_data = [m for m in patterns['seasonal_trends']['monthly_averages'] if m['month'] == booking_date.month]
                if month_data:
                    seasonal_multiplier = month_data[0]['mean'] / 7000
                    base_price *= seasonal_multiplier
            
            # Booking window adjustment
            if 'booking_window_analysis' in patterns:
                if days_ahead <= 7:
                    base_price *= 1.2  # Last minute premium
                elif days_ahead <= 14:
                    base_price *= 1.1
                elif 30 <= days_ahead <= 60:
                    base_price *= 0.9  # Sweet spot discount
        
        return base_price
    
    def _calculate_confidence_score(self, days_ahead: int) -> float:
        """Calculate confidence score for prediction"""
        # Higher confidence for mid-range predictions
        if 14 <= days_ahead <= 60:
            return 0.8 + (0.2 * (1 - abs(days_ahead - 37) / 23))
        elif days_ahead < 14:
            return 0.6 + (0.2 * days_ahead / 14)
        else:
            return 0.5 + (0.3 * (90 - days_ahead) / 30)
    
    def _generate_booking_recommendation(self, optimal_scenario: Dict, all_scenarios: List[Dict]) -> str:
        """Generate human-readable booking recommendation"""
        days_ahead = optimal_scenario['days_ahead']
        price = optimal_scenario['predicted_price']
        confidence = optimal_scenario['confidence_score']
        
        if days_ahead <= 7:
            urgency = "Book immediately!"
        elif days_ahead <= 14:
            urgency = "Book within the next week"
        elif days_ahead <= 30:
            urgency = "Book within the next 2-3 weeks"
        else:
            urgency = "You have time to wait for better deals"
        
        confidence_text = "high" if confidence > 0.7 else "medium" if confidence > 0.5 else "low"
        
        # Check for price trends
        prices = [s['predicted_price'] for s in all_scenarios]
        if len(prices) > 1:
            trend = "increasing" if prices[-1] > prices[0] else "decreasing"
        else:
            trend = "stable"
        
        recommendation = f"{urgency} The optimal booking time is {days_ahead} days in advance " \
                        f"with a predicted price of ₹{price:.0f}. Confidence: {confidence_text}. " \
                        f"Price trend: {trend}."
        
        return recommendation

class FuelPriceSimulator:
    """Simulate impact of fuel price changes on ticket prices"""
    
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        
    def simulate_fuel_impact(self, base_features: pd.DataFrame, 
                           fuel_price_scenarios: List[float]) -> Dict:
        """Simulate price changes for different fuel price scenarios"""
        logger.info("Running fuel price impact simulation...")
        
        results = {}
        base_fuel_price = base_features['fuel_price_usd_per_barrel'].iloc[0] if 'fuel_price_usd_per_barrel' in base_features.columns else 100
        
        # Get best model for predictions
        if self.model_manager.trained_models:
            best_model_name, best_model = self.model_manager.get_best_model()
        else:
            logger.warning("No trained models available for simulation")
            return {}
        
        for fuel_price in fuel_price_scenarios:
            # Create modified features
            modified_features = base_features.copy()
            modified_features['fuel_price_usd_per_barrel'] = fuel_price
            modified_features['fuel_price_inr_per_liter'] = fuel_price * 0.53
            
            # Get prediction
            try:
                predicted_price = best_model.predict(modified_features)[0]
                
                # Calculate impact
                fuel_change_pct = ((fuel_price - base_fuel_price) / base_fuel_price) * 100
                
                results[fuel_price] = {
                    'predicted_price': predicted_price,
                    'fuel_change_pct': fuel_change_pct,
                    'price_elasticity': self._calculate_elasticity(base_fuel_price, fuel_price, 
                                                                 base_features, modified_features)
                }
                
            except Exception as e:
                logger.error(f"Error in simulation for fuel price {fuel_price}: {e}")
        
        return self._format_simulation_results(results, base_fuel_price)
    
    def _calculate_elasticity(self, base_fuel: float, new_fuel: float,
                            base_features: pd.DataFrame, new_features: pd.DataFrame) -> float:
        """Calculate price elasticity"""
        try:
            if self.model_manager.trained_models:
                best_model_name, best_model = self.model_manager.get_best_model()
                
                base_price = best_model.predict(base_features)[0]
                new_price = best_model.predict(new_features)[0]
                
                fuel_change_pct = (new_fuel - base_fuel) / base_fuel
                price_change_pct = (new_price - base_price) / base_price
                
                return price_change_pct / fuel_change_pct if fuel_change_pct != 0 else 0
            
        except Exception as e:
            logger.error(f"Error calculating elasticity: {e}")
            
        return 0
    
    def _format_simulation_results(self, results: Dict, base_fuel_price: float) -> Dict:
        """Format simulation results for display"""
        if not results:
            return {}
        
        # Find baseline (closest to original fuel price)
        baseline_fuel = min(results.keys(), key=lambda x: abs(x - base_fuel_price))
        baseline_price = results[baseline_fuel]['predicted_price']
        
        formatted_results = {
            'base_fuel_price': base_fuel_price,
            'baseline_price': baseline_price,
            'scenarios': [],
            'summary': {}
        }
        
        for fuel_price, data in sorted(results.items()):
            price_impact = ((data['predicted_price'] - baseline_price) / baseline_price) * 100
            
            formatted_results['scenarios'].append({
                'fuel_price': fuel_price,
                'predicted_price': data['predicted_price'],
                'fuel_change_pct': data['fuel_change_pct'],
                'price_impact_pct': price_impact,
                'elasticity': data['price_elasticity']
            })
        
        # Summary statistics
        prices = [s['predicted_price'] for s in formatted_results['scenarios']]
        fuel_changes = [s['fuel_change_pct'] for s in formatted_results['scenarios']]
        price_impacts = [s['price_impact_pct'] for s in formatted_results['scenarios']]
        
        formatted_results['summary'] = {
            'price_range': {'min': min(prices), 'max': max(prices)},
            'max_fuel_impact': max(fuel_changes) if fuel_changes else 0,
            'max_price_impact': max(price_impacts) if price_impacts else 0,
            'average_elasticity': np.mean([s['elasticity'] for s in formatted_results['scenarios']]),
            'sensitivity_level': self._assess_sensitivity_level(price_impacts)
        }
        
        return formatted_results
    
    def _assess_sensitivity_level(self, price_impacts: List[float]) -> str:
        """Assess sensitivity level based on price impacts"""
        max_impact = max(abs(p) for p in price_impacts) if price_impacts else 0
        
        if max_impact < 5:
            return "Low"
        elif max_impact < 15:
            return "Medium"
        else:
            return "High"
    
    def create_fuel_scenarios(self, base_fuel_price: float, 
                            scenario_range: float = 0.3) -> List[float]:
        """Create realistic fuel price scenarios"""
        scenarios = []
        
        # Create scenarios from -30% to +30% in 10% increments
        for pct_change in np.arange(-scenario_range, scenario_range + 0.1, 0.1):
            new_price = base_fuel_price * (1 + pct_change)
            scenarios.append(round(new_price, 2))
        
        return scenarios

if __name__ == "__main__":
    # Test the modules
    from ..data.data_acquisition import DataAcquisitionManager
    from ..models.ml_models import ModelManager
    
    data_manager = DataAcquisitionManager()
    model_manager = ModelManager()
    
    # Test PriceAlertBot
    alert_bot = PriceAlertBot(model_manager, data_manager)
    print("PriceAlertBot initialized successfully")
    
    # Test FuelPriceSimulator
    fuel_simulator = FuelPriceSimulator(model_manager)
    scenarios = fuel_simulator.create_fuel_scenarios(100.0)
    print(f"Created {len(scenarios)} fuel price scenarios: {scenarios}")