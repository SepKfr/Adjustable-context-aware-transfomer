import base
import pandas as pd
import sklearn.preprocessing
import utils
import numpy as np
import random

np.random.seed(21)
random.seed(21)

GenericDataFormatter = base.GenericDataFormatter
DataTypes = base.DataTypes
InputTypes = base.InputTypes


class AirQualityFormatter(GenericDataFormatter):

    _column_definition = [
        ('hours_from_start', DataTypes.REAL_VALUED, InputTypes.TIME),
        ('CO', DataTypes.REAL_VALUED, InputTypes.TARGET),
        ('PM2.5', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
        ('PM10', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
        ('SO2', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
        ('NO2', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
        ('O3', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
        ('TEMP', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
        ('PRES', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
        ('DEWP', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
        ('RAIN', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
        ('WSPM', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
        ('day_of_week', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
        ('id', DataTypes.REAL_VALUED, InputTypes.ID),
    ]

    def split_data(self, df, valid_boundary=1200, test_boundary=1360):
        """Splits data frame into training-validation-test data frames.
        This also calibrates scaling object, and transforms data for each split.
        Args:
          df: Source data frame to split.
          valid_boundary: Starting year for validation data
          test_boundary: Starting year for test data
        Returns:
          Tuple of transformed (train, valid, test) data.
        """

        print('Formatting train-valid-test splits.')

        index = df['days_from_start']
        train = df.loc[index < valid_boundary]
        valid = df.loc[(index >= valid_boundary - 7) & (index < test_boundary)]
        test = df.loc[index >= test_boundary - 7]

        self.set_scalers(train)

        return (self.transform_inputs(data) for data in [train, valid, test])

    def set_scalers(self, df):
        """Calibrates scalers using the data supplied.
        Args:
          df: Data to use to calibrate scalers.
        """
        print('Setting scalers with training data...')

        column_definitions = self.get_column_definition()
        id_column = utils.get_single_col_by_input_type(InputTypes.ID,
                                                       column_definitions)
        target_column = utils.get_single_col_by_input_type(InputTypes.TARGET,
                                                           column_definitions)

        # Extract identifiers in case required
        self.identifiers = list(df[id_column].unique())

        # Format real scalers
        real_inputs = utils.extract_cols_from_data_type(
            DataTypes.REAL_VALUED, column_definitions,
            {InputTypes.ID, InputTypes.TIME})

        data = df[real_inputs].values
        self._real_scalers = sklearn.preprocessing.StandardScaler().fit(data)
        self._target_scaler = sklearn.preprocessing.StandardScaler().fit(
            df[[target_column]].values)  # used for predictions

    def transform_inputs(self, df):
        """Performs feature transformations.
        This includes both feature engineering, preprocessing and normalisation.
        Args:
          df: Data frame to transform.
        Returns:
          Transformed data frame.
        """
        output = df.copy()

        if self._real_scalers is None:
            raise ValueError('Scalers have not been set!')

        column_definitions = self.get_column_definition()

        real_inputs = utils.extract_cols_from_data_type(
            DataTypes.REAL_VALUED, column_definitions,
            {InputTypes.ID, InputTypes.TIME})

        # Format real inputs
        output[real_inputs] = self._real_scalers.transform(df[real_inputs].values)

        return output

    def format_predictions(self, predictions):
        """Reverts any normalisation to give predictions in original scale.
        Args:
          predictions: Dataframe of model predictions.
        Returns:
          Data frame of unnormalised predictions.
        """

        output = predictions.copy()

        column_names = predictions.columns

        for col in column_names:
            if col not in {'identifier'}:
                output[col] = self._target_scaler.inverse_transform(predictions[col])

        return output

    # Default params
    def get_fixed_params(self):
        """Returns fixed model parameters for experiments."""

        fixed_params = {
            'total_time_steps': 6 * 24,
            'num_encoder_steps': 5 * 24,
            'num_epochs': 100,
            'early_stopping_patience': 5,
            'multiprocessing_workers': 5
        }

        return fixed_params

    def get_default_model_params(self):
        """Returns default optimised model parameters."""

        model_params = {
            'dropout_rate': 0.3,
            'hidden_layer_size': 320,
            'learning_rate': 0.001,
            'minibatch_size': 128,
            'max_gradient_norm': 100.,
            'num_heads': 4,
            'stack_size': 1
        }

        return model_params

    def get_num_samples_for_calibration(self):
        """Gets the default number of training and validation samples.
        Use to sub-sample the data for network calibration and a value of -1 uses
        all available samples.
        Returns:
          Tuple of (training samples, validation samples)
        """
        return 45000, 480

