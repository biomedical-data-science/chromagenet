import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import joblib
import os


def make_model(
    input_shape,
    dense_units=(256, 64, 16),
    n_output_units=1,
    output_activation="sigmoid",
    l2_reg=None,
):
    """
    Build the ChromAgeNet convolutional neural network.

    Args:
        input_shape (tuple): Shape of the input images, e.g. (128, 128, 1).
        dense_units (tuple): Number of units of each dense layer in the head.
        n_output_units (int): Number of output units (1 for binary).
        output_activation (str): Activation of the output layer
            ("sigmoid" for binary, "softmax" for categorical).
        l2_reg: Optional regularizer for the separable convolutions.

    Returns:
        keras.Model: The uncompiled model.
    """
    inputs = keras.Input(shape=input_shape)

    # Entry block
    x = layers.Rescaling(1.0 / 255)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, 5, strides=2, padding="same")(x)
    x = layers.Activation("leaky_relu")(x)

    previous_block_activation = x  # Set aside residual

    for size in [64, 64, 128, 128]:
        x = layers.Activation("leaky_relu")(x)
        x = layers.SeparableConv2D(
            size,
            3,
            padding="same",
            depthwise_regularizer=l2_reg,
            pointwise_regularizer=l2_reg,
        )(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("swish")(x)
        x = layers.SeparableConv2D(
            size,
            3,
            padding="same",
            depthwise_regularizer=l2_reg,
            pointwise_regularizer=l2_reg,
        )(x)
        x = layers.SpatialDropout2D(0.1)(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = layers.SeparableConv2D(256, 3, padding="same")(x)
    x = layers.SpatialDropout2D(0.1)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("swish")(x)

    x = layers.GlobalAveragePooling2D()(x)
    for n_units in dense_units:
        x = layers.Dense(n_units, activation="relu")(x)
    output = layers.Dense(units=n_output_units, activation=output_activation)(x)

    return keras.Model(inputs, output)


class CalibratedModel:
    """
    A wrapper class that combines a Keras model with beta calibration
    to create an integrated prediction pipeline.
    """
    
    def __init__(self, keras_model=None, calibrator=None):
        """
        Initialize with an optional keras model and calibrator
        """
        self.keras_model = keras_model
        self.calibrator = calibrator
        
    def predict_proba(self, X, apply_cal=True):
        """
        Generate calibrated probability predictions
        """
        if self.keras_model is None or self.calibrator is None:
            raise ValueError("Model and calibrator must be set before prediction")
            
        # Get raw predictions from Keras model
        raw_pred = self.keras_model.predict(X)

        if apply_cal:
            # Handle different output shapes based on binary vs multi-class
            if raw_pred.ndim > 1 and raw_pred.shape[1] > 1:
                # Multi-class case
                return self.calibrator.predict(raw_pred)
            else:
                # Binary classification case - reshape if needed
                if raw_pred.ndim > 1:
                    raw_pred = raw_pred.flatten()
                
                # Apply calibration
                calibrated = self.calibrator.predict(raw_pred.reshape(-1, 1))
                
                # Return probabilities in the expected format [1-p, p]
                return np.vstack((1 - calibrated, calibrated)).T
        else:
            raw_pred = raw_pred.flatten()
            return np.column_stack((1 - raw_pred, raw_pred))
    
    def predict(self, X, threshold=0.5, apply_cal=True):
        """
        Generate class predictions using the calibrated probabilities
        """
        probs = self.predict_proba(X, apply_cal)
        
        # Handle binary vs multi-class
        if probs.shape[1] == 2:
            return (probs[:, 1] >= threshold).astype(int)
        else:
            return np.argmax(probs, axis=1)
    
    def save(self, filepath):
        """
        Save both the Keras model and calibrator to disk
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save Keras model
        model_path = filepath + '.keras'
        self.keras_model.save(model_path)
        
        # Save calibrator
        calibrator_path = filepath + '.joblib'
        joblib.dump(self.calibrator, calibrator_path)
        
        print(f"Model saved to {model_path}")
        print(f"Calibrator saved to {calibrator_path}")
    
    @classmethod
    def load(cls, filepath):
        """
        Load both Keras model and calibrator from disk
        """
        # Load Keras model
        model_path = filepath + '.keras'
        keras_model = keras.models.load_model(model_path)
        
        # Load calibrator
        calibrator_path = filepath + '.joblib'
        calibrator = joblib.load(calibrator_path)
        
        # Create and return a new instance
        instance = cls(keras_model, calibrator)
        return instance

