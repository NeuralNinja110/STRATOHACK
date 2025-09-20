"""
Machine Learning Models for Airline Price Prediction
Includes: Linear Regression, Random Forest, XGBoost, LSTM, and Quantum Hybrid Model
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import joblib
import logging
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Quantum imports (with fallback if not available)
try:
    from qiskit import QuantumCircuit, transpile
    from qiskit_aer import AerSimulator
    from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
    from qiskit_machine_learning.algorithms import VQR
    from qiskit_machine_learning.neural_networks import CircuitQNN
    from qiskit.algorithms.optimizers import COBYLA
    QUANTUM_AVAILABLE = True
except ImportError:
    QUANTUM_AVAILABLE = False
    logging.warning("Quantum libraries not available. Quantum model will use classical approximation.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelBase:
    """Base class for all models"""
    
    def __init__(self, name: str):
        self.name = name
        self.model = None
        self.is_trained = False
        self.metrics = {}
        
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs):
        raise NotImplementedError
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_trained:
            raise ValueError(f"Model {self.name} is not trained yet.")
        return self.model.predict(X)
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Evaluate model performance"""
        y_pred = self.predict(X)
        
        metrics = {
            'mae': mean_absolute_error(y, y_pred),
            'mse': mean_squared_error(y, y_pred),
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'r2': r2_score(y, y_pred),
            'mape': np.mean(np.abs((y - y_pred) / y)) * 100
        }
        
        self.metrics = metrics
        return metrics
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            'model': self.model,
            'name': self.name,
            'is_trained': self.is_trained,
            'metrics': self.metrics
        }, filepath)
        logger.info(f"Model {self.name} saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.name = model_data['name']
        self.is_trained = model_data['is_trained']
        self.metrics = model_data.get('metrics', {})
        logger.info(f"Model {self.name} loaded from {filepath}")

class LinearRegressionModel(ModelBase):
    """Linear Regression with Ridge regularization"""
    
    def __init__(self, alpha: float = 1.0):
        super().__init__("Linear Regression")
        self.alpha = alpha
        
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, 
              X_val: pd.DataFrame = None, y_val: pd.Series = None, **kwargs):
        """Train Linear Regression model"""
        logger.info(f"Training {self.name} model...")
        
        self.model = Ridge(alpha=self.alpha, random_state=42)
        self.model.fit(X_train, y_train)
        
        self.is_trained = True
        
        # Evaluate on validation set if provided
        if X_val is not None and y_val is not None:
            val_metrics = self.evaluate(X_val, y_val)
            logger.info(f"Validation metrics: {val_metrics}")
        
        logger.info(f"{self.name} training completed")

class RandomForestModel(ModelBase):
    """Random Forest Regressor"""
    
    def __init__(self, n_estimators: int = 100, max_depth: int = None, 
                 min_samples_split: int = 2, min_samples_leaf: int = 1):
        super().__init__("Random Forest")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        
    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_val: pd.DataFrame = None, y_val: pd.Series = None, **kwargs):
        """Train Random Forest model"""
        logger.info(f"Training {self.name} model...")
        
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Evaluate on validation set if provided
        if X_val is not None and y_val is not None:
            val_metrics = self.evaluate(X_val, y_val)
            logger.info(f"Validation metrics: {val_metrics}")
        
        logger.info(f"{self.name} training completed")
    
    def get_feature_importance(self, feature_names: List[str]) -> pd.DataFrame:
        """Get feature importance"""
        if not self.is_trained:
            raise ValueError("Model is not trained yet.")
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df

class XGBoostModel(ModelBase):
    """XGBoost Regressor"""
    
    def __init__(self, n_estimators: int = 100, max_depth: int = 6,
                 learning_rate: float = 0.1, subsample: float = 0.8,
                 colsample_bytree: float = 0.8):
        super().__init__("XGBoost")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        
    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_val: pd.DataFrame = None, y_val: pd.Series = None, **kwargs):
        """Train XGBoost model"""
        logger.info(f"Training {self.name} model...")
        
        self.model = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            random_state=42,
            n_jobs=-1
        )
        
        # Use early stopping if validation data is provided
        eval_set = [(X_val, y_val)] if X_val is not None and y_val is not None else None
        
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            early_stopping_rounds=10 if eval_set else None,
            verbose=False
        )
        
        self.is_trained = True
        
        # Evaluate on validation set if provided
        if X_val is not None and y_val is not None:
            val_metrics = self.evaluate(X_val, y_val)
            logger.info(f"Validation metrics: {val_metrics}")
        
        logger.info(f"{self.name} training completed")
    
    def get_feature_importance(self, feature_names: List[str]) -> pd.DataFrame:
        """Get feature importance"""
        if not self.is_trained:
            raise ValueError("Model is not trained yet.")
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df

class LSTMModel(ModelBase):
    """LSTM Neural Network for time series prediction"""
    
    def __init__(self, sequence_length: int = 30, lstm_units: int = 50,
                 dropout_rate: float = 0.2, epochs: int = 50, batch_size: int = 32):
        super().__init__("LSTM Neural Network")
        self.sequence_length = sequence_length
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.scaler_X = None
        self.scaler_y = None
        
    def _create_sequences(self, X: pd.DataFrame, y: pd.Series = None) -> Tuple:
        """Create sequences for LSTM input"""
        X_sequences = []
        y_sequences = []
        
        X_array = X.values
        y_array = y.values if y is not None else None
        
        for i in range(self.sequence_length, len(X_array)):
            X_sequences.append(X_array[i-self.sequence_length:i])
            if y_array is not None:
                y_sequences.append(y_array[i])
        
        X_sequences = np.array(X_sequences)
        y_sequences = np.array(y_sequences) if y_array is not None else None
        
        return X_sequences, y_sequences
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_val: pd.DataFrame = None, y_val: pd.Series = None, **kwargs):
        """Train LSTM model"""
        logger.info(f"Training {self.name} model...")
        
        # Create sequences
        X_train_seq, y_train_seq = self._create_sequences(X_train, y_train)
        
        if X_val is not None and y_val is not None:
            X_val_seq, y_val_seq = self._create_sequences(X_val, y_val)
            validation_data = (X_val_seq, y_val_seq)
        else:
            validation_data = None
        
        # Build LSTM model
        self.model = keras.Sequential([
            layers.LSTM(self.lstm_units, return_sequences=True, 
                       input_shape=(self.sequence_length, X_train.shape[1])),
            layers.Dropout(self.dropout_rate),
            layers.LSTM(self.lstm_units // 2, return_sequences=False),
            layers.Dropout(self.dropout_rate),
            layers.Dense(50, activation='relu'),
            layers.Dense(25, activation='relu'),
            layers.Dense(1)
        ])
        
        self.model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        # Train the model
        history = self.model.fit(
            X_train_seq, y_train_seq,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_data=validation_data,
            verbose=0,
            callbacks=[
                keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5)
            ]
        )
        
        self.is_trained = True
        self.training_history = history.history
        
        logger.info(f"{self.name} training completed")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions with LSTM model"""
        if not self.is_trained:
            raise ValueError(f"Model {self.name} is not trained yet.")
        
        X_seq, _ = self._create_sequences(X)
        predictions = self.model.predict(X_seq, verbose=0)
        
        return predictions.flatten()
    
    def save_model(self, filepath: str):
        """Save LSTM model"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # Save Keras model
        self.model.save(f"{filepath}_keras_model")
        
        # Save other attributes
        model_data = {
            'name': self.name,
            'is_trained': self.is_trained,
            'metrics': self.metrics,
            'sequence_length': self.sequence_length,
            'lstm_units': self.lstm_units,
            'dropout_rate': self.dropout_rate,
            'epochs': self.epochs,
            'batch_size': self.batch_size
        }
        
        joblib.dump(model_data, f"{filepath}_metadata.pkl")
        logger.info(f"LSTM model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load LSTM model"""
        # Load Keras model
        self.model = keras.models.load_model(f"{filepath}_keras_model")
        
        # Load other attributes
        model_data = joblib.load(f"{filepath}_metadata.pkl")
        self.name = model_data['name']
        self.is_trained = model_data['is_trained']
        self.metrics = model_data.get('metrics', {})
        self.sequence_length = model_data['sequence_length']
        self.lstm_units = model_data['lstm_units']
        self.dropout_rate = model_data['dropout_rate']
        self.epochs = model_data['epochs']
        self.batch_size = model_data['batch_size']
        
        logger.info(f"LSTM model loaded from {filepath}")

class QuantumHybridModel(ModelBase):
    """Quantum Hybrid Model (VQR or classical approximation)"""
    
    def __init__(self, n_qubits: int = 4, n_features: int = None, n_layers: int = 2):
        super().__init__("Quantum Hybrid")
        self.n_qubits = min(n_qubits, n_features if n_features else 4)
        self.n_features = n_features
        self.n_layers = n_layers
        self.feature_map = None
        self.ansatz = None
        self.classical_model = None
        
    def _create_quantum_circuit(self):
        """Create quantum circuit for VQR"""
        if not QUANTUM_AVAILABLE:
            return None
        
        # Feature map
        self.feature_map = ZZFeatureMap(feature_dimension=self.n_qubits, reps=1)
        
        # Ansatz
        self.ansatz = RealAmplitudes(num_qubits=self.n_qubits, reps=self.n_layers)
        
        return self.feature_map, self.ansatz
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_val: pd.DataFrame = None, y_val: pd.Series = None, **kwargs):
        """Train Quantum Hybrid model"""
        logger.info(f"Training {self.name} model...")
        
        if QUANTUM_AVAILABLE:
            try:
                self._train_quantum_model(X_train, y_train, X_val, y_val)
            except Exception as e:
                logger.warning(f"Quantum training failed: {e}. Falling back to classical approximation.")
                self._train_classical_approximation(X_train, y_train, X_val, y_val)
        else:
            self._train_classical_approximation(X_train, y_train, X_val, y_val)
        
        logger.info(f"{self.name} training completed")
    
    def _train_quantum_model(self, X_train, y_train, X_val, y_val):
        """Train actual quantum model"""
        # Reduce dimensionality to fit quantum circuit
        if X_train.shape[1] > self.n_qubits:
            from sklearn.decomposition import PCA
            self.pca = PCA(n_components=self.n_qubits)
            X_train_reduced = self.pca.fit_transform(X_train)
            X_val_reduced = self.pca.transform(X_val) if X_val is not None else None
        else:
            X_train_reduced = X_train.values
            X_val_reduced = X_val.values if X_val is not None else None
            self.pca = None
        
        # Create quantum circuit
        feature_map, ansatz = self._create_quantum_circuit()
        
        # Create quantum neural network
        circuit = feature_map.compose(ansatz)
        
        # Use AerSimulator
        simulator = AerSimulator()
        
        # Create VQR
        optimizer = COBYLA(maxiter=100)
        
        # This is a simplified quantum model setup
        # In practice, you would need more sophisticated quantum ML setup
        self.model = {
            'type': 'quantum',
            'circuit': circuit,
            'optimizer': optimizer,
            'pca': self.pca
        }
        
        # For demonstration, we'll use a classical approximation with quantum-inspired features
        self._train_classical_approximation(X_train, y_train, X_val, y_val)
        
        self.is_trained = True
    
    def _train_classical_approximation(self, X_train, y_train, X_val, y_val):
        """Train classical approximation of quantum model"""
        # Use quantum-inspired feature transformations
        X_train_quantum = self._quantum_inspired_features(X_train)
        X_val_quantum = self._quantum_inspired_features(X_val) if X_val is not None else None
        
        # Use a neural network to approximate quantum behavior
        self.classical_model = keras.Sequential([
            layers.Dense(64, activation='tanh', input_shape=(X_train_quantum.shape[1],)),
            layers.Dropout(0.2),
            layers.Dense(32, activation='tanh'),
            layers.Dropout(0.2),
            layers.Dense(16, activation='tanh'),
            layers.Dense(1)
        ])
        
        self.classical_model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        # Train the classical approximation
        validation_data = (X_val_quantum, y_val) if X_val_quantum is not None else None
        
        self.classical_model.fit(
            X_train_quantum, y_train,
            epochs=50,
            batch_size=32,
            validation_data=validation_data,
            verbose=0,
            callbacks=[keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)]
        )
        
        self.model = {
            'type': 'classical_approximation',
            'classical_model': self.classical_model
        }
        
        self.is_trained = True
    
    def _quantum_inspired_features(self, X: pd.DataFrame) -> np.ndarray:
        """Create quantum-inspired features"""
        X_array = X.values
        
        # Apply quantum-inspired transformations
        quantum_features = []
        
        # 1. Amplitude encoding inspired features
        for i in range(min(4, X_array.shape[1])):
            # Normalize to [0, 2π]
            normalized = (X_array[:, i] - X_array[:, i].min()) / (X_array[:, i].max() - X_array[:, i].min() + 1e-8) * 2 * np.pi
            quantum_features.append(np.sin(normalized))
            quantum_features.append(np.cos(normalized))
        
        # 2. Entanglement-inspired features
        for i in range(min(2, X_array.shape[1] - 1)):
            for j in range(i + 1, min(4, X_array.shape[1])):
                # Create interaction terms
                quantum_features.append(X_array[:, i] * X_array[:, j])
                quantum_features.append(np.sin(X_array[:, i]) * np.cos(X_array[:, j]))
        
        # 3. Phase-inspired features
        for i in range(min(3, X_array.shape[1])):
            phase = np.arctan2(X_array[:, i], X_array[:, (i + 1) % X_array.shape[1]])
            quantum_features.append(np.sin(phase))
            quantum_features.append(np.cos(phase))
        
        return np.column_stack(quantum_features)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions with quantum hybrid model"""
        if not self.is_trained:
            raise ValueError(f"Model {self.name} is not trained yet.")
        
        if self.model['type'] == 'classical_approximation':
            X_quantum = self._quantum_inspired_features(X)
            return self.classical_model.predict(X_quantum, verbose=0).flatten()
        else:
            # Quantum prediction would go here
            # For now, fallback to classical
            X_quantum = self._quantum_inspired_features(X)
            return self.classical_model.predict(X_quantum, verbose=0).flatten()

class ModelManager:
    """Manager class for all models"""
    
    def __init__(self, models_dir: str = "models/saved"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.models = {}
        self.trained_models = {}
        self.model_metrics = {}
        
    def initialize_models(self, n_features: int = None) -> Dict[str, ModelBase]:
        """Initialize all models"""
        logger.info("Initializing all models...")
        
        self.models = {
            'linear_regression': LinearRegressionModel(alpha=1.0),
            'random_forest': RandomForestModel(n_estimators=100, max_depth=20),
            'xgboost': XGBoostModel(n_estimators=100, max_depth=6, learning_rate=0.1),
            'lstm': LSTMModel(sequence_length=30, lstm_units=50, epochs=50),
            'quantum_hybrid': QuantumHybridModel(n_qubits=4, n_features=n_features)
        }
        
        logger.info(f"Initialized {len(self.models)} models")
        return self.models
    
    def train_all_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                        X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, Dict]:
        """Train all models and return metrics"""
        logger.info("Training all models...")
        
        results = {}
        
        for name, model in self.models.items():
            try:
                logger.info(f"Training {name}...")
                
                # Train model
                model.train(X_train, y_train, X_val, y_val)
                
                # Evaluate model
                train_metrics = model.evaluate(X_train, y_train)
                val_metrics = model.evaluate(X_val, y_val)
                
                # Store results
                results[name] = {
                    'model': model,
                    'train_metrics': train_metrics,
                    'val_metrics': val_metrics
                }
                
                # Save model
                model_path = self.models_dir / f"{name}_model.pkl"
                if name == 'lstm':
                    model.save_model(str(model_path).replace('.pkl', ''))
                else:
                    model.save_model(model_path)
                
                self.trained_models[name] = model
                self.model_metrics[name] = val_metrics
                
                logger.info(f"{name} training completed. Val RMSE: {val_metrics['rmse']:.2f}")
                
            except Exception as e:
                logger.error(f"Error training {name}: {e}")
                results[name] = {'error': str(e)}
        
        logger.info("All models training completed")
        return results
    
    def get_best_model(self, metric: str = 'rmse') -> Tuple[str, ModelBase]:
        """Get the best performing model"""
        if not self.model_metrics:
            raise ValueError("No trained models available")
        
        # For RMSE and MAE, lower is better
        if metric in ['rmse', 'mae', 'mse', 'mape']:
            best_model_name = min(self.model_metrics, key=lambda x: self.model_metrics[x][metric])
        else:  # For R2, higher is better
            best_model_name = max(self.model_metrics, key=lambda x: self.model_metrics[x][metric])
        
        return best_model_name, self.trained_models[best_model_name]
    
    def compare_models(self) -> pd.DataFrame:
        """Compare all trained models"""
        if not self.model_metrics:
            raise ValueError("No trained models available")
        
        comparison_df = pd.DataFrame(self.model_metrics).T
        comparison_df = comparison_df.round(4)
        comparison_df = comparison_df.sort_values('rmse')
        
        return comparison_df
    
    def predict_ensemble(self, X: pd.DataFrame, method: str = 'average') -> np.ndarray:
        """Make ensemble predictions"""
        if not self.trained_models:
            raise ValueError("No trained models available")
        
        predictions = {}
        
        for name, model in self.trained_models.items():
            try:
                pred = model.predict(X)
                predictions[name] = pred
            except Exception as e:
                logger.warning(f"Error predicting with {name}: {e}")
        
        if not predictions:
            raise ValueError("No models could make predictions")
        
        pred_df = pd.DataFrame(predictions)
        
        if method == 'average':
            return pred_df.mean(axis=1).values
        elif method == 'weighted':
            # Weight by inverse RMSE
            weights = {}
            for name in pred_df.columns:
                if name in self.model_metrics:
                    weights[name] = 1 / (self.model_metrics[name]['rmse'] + 1e-8)
            
            total_weight = sum(weights.values())
            weighted_pred = np.zeros(len(pred_df))
            
            for name, weight in weights.items():
                weighted_pred += pred_df[name].values * (weight / total_weight)
            
            return weighted_pred
        else:
            return pred_df.median(axis=1).values

if __name__ == "__main__":
    # Test model initialization
    manager = ModelManager()
    models = manager.initialize_models(n_features=20)
    
    print(f"Initialized models: {list(models.keys())}")
    print(f"Quantum libraries available: {QUANTUM_AVAILABLE}")