import logging
from abc import ABC, abstractmethod

import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

class Model(ABC):
    """
    Abstract base class for all models.
    """

    @abstractmethod
    def train(self, x_train, y_train):
        """
        Trains the model on the given data.

        Args:
            x_train: Training data
            y_train: Target data
        """
        pass

# Implementing Linear Regression:
class LinearRegressionModel(Model):
    """
    LinearRegressionModel that implements the Model interface.
    """

    def train(self, x_train, y_train, **kwargs):
        try:
            reg = LinearRegression(**kwargs)
            reg.fit(x_train, y_train)
            logging.info("LR Model Training Complete!")
            return reg
        except Exception as err:
            logging.error("Error in training model: {}", format(err))
            raise err

# Implementing LightGBM Model: 
class LightGBMModel(Model):
    """
    LightGBMModel that implements the Model interface.
    """

    def train(self, x_train, y_train, **kwargs):
        reg = LGBMRegressor(**kwargs)
        reg.fit(x_train, y_train)
        logging.info("LR Model Training Complete!")
        return reg

# Implementing Random Forest Model:
class RandomForestModel(Model):
    """
    RandomForestModel that implements the Model interface.
    """

    def train(self, x_train, y_train, **kwargs):
        reg = RandomForestRegressor(**kwargs)
        reg.fit(x_train, y_train)
        logging.info("LR Model Training Complete!")
        return reg