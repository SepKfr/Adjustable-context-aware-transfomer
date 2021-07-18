from Utils import base
from data.electricity import ElectricityFormatter
import numpy as np
import random

np.random.seed(21)
random.seed(21)

DataFormatter = ElectricityFormatter
DataTypes = base.DataTypes
InputTypes = base.InputTypes


class SolarFormatter(ElectricityFormatter):

    _column_definition = [
        ('hours_from_start', DataTypes.REAL_VALUED, InputTypes.TIME),
        ('day_of_week', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
        ('hour', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
        ('Power(MW)', DataTypes.REAL_VALUED, InputTypes.TARGET),
        ('id', DataTypes.REAL_VALUED, InputTypes.ID),
        ('categorical_id', DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT),
        ('capacity', DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT),

    ]

    def split_data(self, df, valid_boundary=214, test_boundary=289):
        print('Formatting train-valid-test splits.')

        index = df['days_from_start']
        train = df.loc[index < valid_boundary]
        valid = df.loc[(index >= valid_boundary - 7) & (index < test_boundary)]
        test = df.loc[index >= test_boundary - 7]

        self.set_scalers(train)

        return (self.transform_inputs(data) for data in [train, valid, test])

    def get_fixed_params(self):
        """Returns fixed model parameters for experiments."""

        fixed_params = {
            'total_time_steps': 8 * 24,
            'num_encoder_steps': 7 * 24,
            'num_epochs': 100,
            'early_stopping_patience': 5,
            'multiprocessing_workers': 5
        }

        return fixed_params

    def get_default_model_params(self):
        """Returns default optimised model parameters."""

        model_params = {
            'dropout_rate': 0.3,
            'hidden_layer_size': [16, 32],
            'learning_rate': 0.001,
            'minibatch_size': 64,
            'max_gradient_norm': 100.,
            'num_heads': 8,
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
        return 64000, 8000
