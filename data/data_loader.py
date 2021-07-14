import wget
import pyunpack
import os
import pandas as pd
import numpy as np
import argparse
import sys
import random
import gc
import glob

import air_quality, electricity, traffic, watershed, solar

np.random.seed(21)
random.seed(21)


class ExperimentConfig(object):
    default_experiments = ['electricity', 'traffic', 'air_quality', 'favorita', 'watershed', 'solar']

    def __init__(self, experiment='electricity', root_folder=None):

        if experiment not in self.default_experiments:
            raise ValueError('Unrecognised experiment={}'.format(experiment))

        if root_folder is None:
            root_folder = os.path.join(
                os.path.dirname(os.path.realpath(__file__)), '../..', 'outputs')
            print('Using root folder {}'.format(root_folder))

        self.root_folder = root_folder
        self.experiment = experiment
        self.data_folder = os.path.join(root_folder, '', experiment)

        for relevant_directory in [
            self.root_folder, self.data_folder
        ]:
            if not os.path.exists(relevant_directory):
                os.makedirs(relevant_directory)

    @property
    def data_csv_path(self):
        csv_map = {
            'electricity': 'hourly_electricity.csv',
            'traffic': 'hourly_traffic.csv',
            'air_quality': 'hourly_air_quality.csv',
            'favorita': 'favorita_consolidated.csv',
            'watershed': 'watershed.csv',
            'solar': 'solar.csv'
        }

        return os.path.join(self.data_folder, csv_map[self.experiment])

    def make_data_formatter(self):
        """Gets a data formatter object for experiment.
        Returns:
          Default DataFormatter per experiment.
        """

        data_formatter_class = {
            'electricity': electricity.ElectricityFormatter,
            'traffic': traffic.TrafficFormatter,
            'air_quality': air_quality.AirQualityFormatter,
            'watershed': watershed.WatershedFormatter,
            'solar': solar.SolarFormatter,
        }

        return data_formatter_class[self.experiment]()


def download_from_url(url, output_path):
    """Downloads a file from url."""

    print('Pulling data from {} to {}'.format(url, output_path))
    wget.download(url, output_path)
    print('done')


def unzip(zip_path, output_file, data_folder):
    """Unzips files and checks successful completion."""

    print('Unzipping file: {}'.format(zip_path))
    pyunpack.Archive(zip_path).extractall(data_folder)

    # Checks if unzip was successful
    '''if not os.path.exists(output_file):
        raise ValueError(
            'Error in unzipping process! {} not found.'.format(output_file))'''


def download_and_unzip(url, zip_path, csv_path, data_folder):
    """Downloads and unzips an online csv file.
    Args:
    url: Web address
    zip_path: Path to download zip file
    csv_path: Expected path to csv file
    data_folder: Folder in which data is stored.
    """

    download_from_url(url, zip_path)

    unzip(zip_path, csv_path, data_folder)

    print('Done.')


def process_watershed(config):

    """Process watershed dataset
    Args:
    config: Default experiment config for Watershed
    """
    sites = ['BDC', 'BEF', 'DCF', 'GOF', 'HBF', 'LMP', 'MCQ', 'SBM', 'TPB', 'WHB']
    data_path = config.data_folder
    df_list = []

    for i, site in enumerate(sites):

        df = pd.read_csv('{}/{}_WQual_Level4.csv'.format(data_path, site), index_col=0, sep=',')
        df_list.append(df.iloc[0::4, :])

    output = pd.concat(df_list, axis=0)
    output.index = pd.to_datetime(output.Date)
    output.sort_index(inplace=True)
    output = output.dropna(axis=1, how='all')
    output = output.fillna(0.)

    start_date = pd.to_datetime('2013-03-28')
    earliest_time = start_date
    output = output[output.index >= start_date]

    date = output.index
    output['day_of_week'] = date.dayofweek
    output['id'] = output['Site']
    output['categorical_id'] = output['Site']
    output['hours_from_start'] = (date - earliest_time).seconds / 60 / 60 + (
            date - earliest_time).days * 24

    output['days_from_start'] = (date - earliest_time).days
    output = output[output['Site'] != 0.0]
    output = output.fillna('na')
    output = output[output['days_from_start'] != 'na']
    output.to_csv("watershed.csv")

    print('Done.')


def download_air_quality(args):

    """Downloads air quality dataset from UCI repository"""

    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00501/PRSA2017_Data_20130301-20170228.zip'

    sites = ['Wanshouxigong', 'Wanliu', 'Shunyi', 'Nongzhanguan', 'Huairou', 'Gucheng',
             'Guanyuan', 'Dongsi', 'Dingling', 'Changping', 'Aotizhongxin']
    data_folder = args.data_folder
    data_path = os.path.join(data_folder, 'PRSA_Data_20130301-20170228')
    zip_path = data_path + '.zip'
    download_and_unzip(url, zip_path, data_path, data_folder)
    df_list = []

    for i, site in enumerate(sites):

        df = pd.read_csv('{}/PRSA_Data_{}_20130301-20170228.csv'.format(data_path, site),
                         index_col=0, sep=',')
        df_list.append(df)

    output = pd.concat(df_list, axis=0)
    output.index = pd.to_datetime(output[['year','month','day']])
    output.sort_index(inplace=True)
    earliest_time = output.index.min()

    start_date = min(output.fillna(method='ffill').dropna().index)
    end_date = max(output.fillna(method='bfill').dropna().index)

    active_range = (output.index >= start_date) & (output.index <= end_date)
    output = output[active_range].fillna(0.)

    date = output.index

    output['day_of_week'] = date.dayofweek
    output['id'] = output['station']
    output['categorical_id'] = output['station']
    output['hours_from_start'] = (date - earliest_time).seconds / 60 / 60 + (
            date - earliest_time).days * 24
    output['days_from_start'] = (date - earliest_time).days
    output.to_csv("air_quality.csv")

    print('Done.')


def download_solar(args):

    url = 'https://www.nrel.gov/grid/assets/downloads/al-pv-2006.zip'
    data_folder = args.data_folder
    csv_path = os.path.join(data_folder, 'al-pv-2006')
    zip_path = csv_path + '.zip'

    download_and_unzip(url, zip_path, csv_path, data_folder)

    files = [
        'Actual_30.45_-88.25_2006_UPV_70MW_5_Min', 'Actual_30.55_-87.55_2006_UPV_80MW_5_Min',
        'Actual_30.55_-87.75_2006_DPV_36MW_5_Min', 'Actual_30.55_-88.15_2006_DPV_38MW_5_Min',
        'Actual_30.55_-88.25_2006_DPV_38MW_5_Min', 'Actual_30.65_-87.55_2006_UPV_50MW_5_Min',
        'Actual_30.65_-87.65_2006_DPV_36MW_5_Min', 'Actual_30.65_-87.75_2006_DPV_36MW_5_Min',
        'Actual_30.65_-87.85_2006_DPV_36MW_5_Min', 'Actual_30.65_-88.05_2006_DPV_38MW_5_Min',
        'Actual_30.65_-88.15_2006_DPV_38MW_5_Min', 'Actual_30.65_-88.25_2006_DPV_38MW_5_Min',
        'Actual_30.65_-88.35_2006_UPV_10MW_5_Min', 'Actual_30.75_-87.75_2006_DPV_36MW_5_Min',
        'Actual_30.75_-87.95_2006_UPV_30MW_5_Min', 'Actual_30.75_-88.05_2006_DPV_38MW_5_Min',
        'Actual_30.75_-88.15_2006_DPV_38MW_5_Min', 'Actual_30.75_-88.25_2006_DPV_38MW_5_Min',
        'Actual_30.85_-88.15_2006_DPV_38MW_5_Min', 'Actual_31.05_-85.55_2006_UPV_20MW_5_Min',
        'Actual_31.05_-85.65_2006_UPV_70MW_5_Min', 'Actual_31.05_-85.75_2006_UPV_60MW_5_Min',
        'Actual_31.15_-85.15_2006_DPV_34MW_5_Min', 'Actual_31.15_-85.25_2006_DPV_34MW_5_Min',
        'Actual_31.15_-86.65_2006_UPV_60MW_5_Min', 'Actual_31.15_-87.65_2006_UPV_30MW_5_Min',
        'Actual_31.25_-85.25_2006_DPV_34MW_5_Min', 'Actual_31.25_-85.55_2006_UPV_50MW_5_Min',
        'Actual_31.25_-85.65_2006_UPV_20MW_5_Min', 'Actual_31.35_-86.85_2006_UPV_20MW_5_Min',
        'Actual_31.35_-87.05_2006_UPV_70MW_5_Min', 'Actual_31.35_-87.65_2006_UPV_100MW_5_Min',
        'Actual_31.35_-88.05_2006_UPV_80MW_5_Min', 'Actual_31.45_-85.45_2006_UPV_50MW_5_Min',
        'Actual_31.45_-85.75_2006_UPV_30MW_5_Min', 'Actual_31.95_-86.45_2006_UPV_30MW_5_Min',
        'Actual_32.05_-85.95_2006_UPV_30MW_5_Min', 'Actual_32.15_-86.25_2006_DPV_39MW_5_Min',
        'Actual_32.25_-85.25_2006_UPV_10MW_5_Min', 'Actual_32.25_-86.15_2006_DPV_39MW_5_Min',
        'Actual_32.25_-86.25_2006_DPV_39MW_5_Min', 'Actual_32.25_-86.25_2006_UPV_60MW_5_Min',
        'Actual_32.25_-86.35_2006_DPV_39MW_5_Min', 'Actual_32.35_-86.15_2006_DPV_39MW_5_Min',
        'Actual_32.35_-86.25_2006_DPV_39MW_5_Min', 'Actual_32.55_-85.35_2006_DPV_35MW_5_Min',
        'Actual_32.55_-85.35_2006_UPV_50MW_5_Min', 'Actual_32.55_-86.05_2006_DPV_27MW_5_Min',
        'Actual_32.55_-86.15_2006_DPV_27MW_5_Min', 'Actual_32.55_-86.55_2006_UPV_40MW_5_Min',
        'Actual_32.65_-85.25_2006_DPV_35MW_5_Min', 'Actual_32.65_-85.35_2006_DPV_35MW_5_Min',
        'Actual_32.65_-85.85_2006_UPV_30MW_5_Min', 'Actual_32.65_-86.15_2006_DPV_27MW_5_Min',
        'Actual_32.75_-85.35_2006_DPV_35MW_5_Min', 'Actual_32.75_-85.45_2006_UPV_90MW_5_Min',
        'Actual_32.95_-85.45_2006_UPV_100MW_5_Min', 'Actual_33.15_-86.55_2006_DPV_35MW_5_Min',
        'Actual_33.15_-86.65_2006_DPV_35MW_5_Min', 'Actual_33.15_-87.55_2006_DPV_39MW_5_Min',
        'Actual_33.25_-86.55_2006_DPV_35MW_5_Min', 'Actual_33.25_-86.65_2006_DPV_35MW_5_Min',
        'Actual_33.25_-86.75_2006_DPV_35MW_5_Min', 'Actual_33.25_-87.45_2006_DPV_39MW_5_Min',
        'Actual_33.25_-87.55_2006_DPV_39MW_5_Min', 'Actual_33.25_-87.65_2006_DPV_39MW_5_Min',
        'Actual_33.35_-86.05_2006_DPV_28MW_5_Min', 'Actual_33.35_-86.15_2006_DPV_28MW_5_Min',
        'Actual_33.35_-86.55_2006_DPV_35MW_5_Min', 'Actual_33.35_-86.65_2006_DPV_35MW_5_Min',
        'Actual_33.35_-86.75_2006_DPV_39MW_5_Min', 'Actual_33.35_-86.85_2006_DPV_39MW_5_Min',
        'Actual_33.35_-86.95_2006_DPV_39MW_5_Min', 'Actual_33.35_-87.05_2006_DPV_39MW_5_Min',
        'Actual_33.35_-87.45_2006_DPV_39MW_5_Min', 'Actual_33.35_-87.55_2006_DPV_39MW_5_Min',
        'Actual_33.45_-85.85_2006_UPV_50MW_5_Min', 'Actual_33.45_-85.95_2006_UPV_40MW_5_Min',
        'Actual_33.45_-86.15_2006_DPV_28MW_5_Min', 'Actual_33.45_-86.25_2006_UPV_40MW_5_Min',
        'Actual_33.45_-86.65_2006_DPV_39MW_5_Min', 'Actual_33.45_-86.75_2006_DPV_39MW_5_Min',
        'Actual_33.45_-86.85_2006_DPV_39MW_5_Min', 'Actual_33.45_-86.95_2006_DPV_39MW_5_Min',
        'Actual_33.45_-87.05_2006_DPV_39MW_5_Min', 'Actual_33.55_-85.55_2006_UPV_60MW_5_Min',
        'Actual_33.55_-86.65_2006_DPV_39MW_5_Min', 'Actual_33.55_-86.75_2006_DPV_39MW_5_Min',
        'Actual_33.55_-86.85_2006_DPV_39MW_5_Min', 'Actual_33.55_-86.95_2006_DPV_39MW_5_Min',
        'Actual_33.55_-87.05_2006_DPV_39MW_5_Min', 'Actual_33.65_-86.35_2006_UPV_80MW_5_Min',
        'Actual_33.65_-86.65_2006_DPV_39MW_5_Min', 'Actual_33.65_-86.75_2006_DPV_39MW_5_Min',
        'Actual_33.65_-86.85_2006_DPV_39MW_5_Min', 'Actual_33.65_-86.95_2006_DPV_39MW_5_Min',
        'Actual_33.75_-85.75_2006_DPV_39MW_5_Min', 'Actual_33.75_-85.85_2006_DPV_39MW_5_Min',
        'Actual_33.75_-86.25_2006_DPV_28MW_5_Min', 'Actual_33.75_-86.25_2006_UPV_20MW_5_Min',
        'Actual_33.75_-86.35_2006_DPV_28MW_5_Min', 'Actual_33.75_-86.65_2006_DPV_39MW_5_Min',
        'Actual_33.75_-86.75_2006_DPV_39MW_5_Min', 'Actual_33.75_-86.85_2006_DPV_39MW_5_Min',
        'Actual_33.85_-85.85_2006_DPV_39MW_5_Min', 'Actual_33.85_-86.35_2006_DPV_28MW_5_Min',
        'Actual_34.05_-85.95_2006_DPV_36MW_5_Min', 'Actual_34.05_-86.05_2006_DPV_36MW_5_Min',
        'Actual_34.15_-85.55_2006_UPV_70MW_5_Min', 'Actual_34.15_-86.05_2006_DPV_36MW_5_Min',
        'Actual_34.15_-86.75_2006_DPV_35MW_5_Min', 'Actual_34.15_-86.85_2006_DPV_35MW_5_Min',
        'Actual_34.25_-86.85_2006_DPV_35MW_5_Min', 'Actual_34.35_-86.25_2006_DPV_38MW_5_Min',
        'Actual_34.35_-86.35_2006_DPV_38MW_5_Min', 'Actual_34.35_-86.85_2006_DPV_37MW_5_Min',
        'Actual_34.45_-86.35_2006_DPV_38MW_5_Min', 'Actual_34.45_-86.75_2006_DPV_37MW_5_Min',
        'Actual_34.45_-86.85_2006_DPV_37MW_5_Min', 'Actual_34.55_-86.85_2006_DPV_37MW_5_Min',
        'Actual_34.65_-86.45_2006_DPV_38MW_5_Min', 'Actual_34.65_-86.55_2006_DPV_38MW_5_Min',
        'Actual_34.65_-86.65_2006_DPV_38MW_5_Min', 'Actual_34.75_-86.35_2006_DPV_38MW_5_Min',
        'Actual_34.75_-86.45_2006_DPV_38MW_5_Min', 'Actual_34.75_-86.55_2006_DPV_38MW_5_Min',
        'Actual_34.75_-86.65_2006_DPV_38MW_5_Min', 'Actual_34.85_-86.45_2006_DPV_38MW_5_Min',
        'Actual_34.85_-86.55_2006_DPV_38MW_5_Min', 'Actual_34.85_-86.65_2006_DPV_38MW_5_Min',
        'Actual_34.85_-86.85_2006_DPV_33MW_5_Min', 'Actual_34.85_-86.95_2006_DPV_33MW_5_Min',
        'Actual_34.95_-86.55_2006_DPV_38MW_5_Min', 'Actual_34.95_-86.95_2006_DPV_33MW_5_Min',
        'Actual_34.95_-87.55_2006_DPV_38MW_5_Min', 'Actual_34.95_-87.65_2006_DPV_38MW_5_Min',
        'Actual_35.05_-87.65_2006_DPV_38MW_5_Min'
    ]

    df_list = []

    for file in files:
        parts = file.split("_")
        df = pd.read_csv(os.path.join(data_folder, '{}.csv'.format(file)), index_col=0, sep=',')
        df_hr = df.iloc[0::12, :]
        df_sub = df_hr.copy()
        df_sub['latitude'] = parts[1]
        df_sub['longtitude'] = parts[2]
        df_sub['id'] = parts[1] + "_" + parts[2]
        df_sub['capacity'] = parts[5]
        df_list.append(df_sub)

    output = pd.concat(df_list, axis=0)
    output.index = pd.to_datetime(output.index)
    output.sort_index(inplace=True)
    earliest_time = output.index.min()
    date = output.index

    output['day_of_week'] = date.dayofweek
    output['hours_from_start'] = (date - earliest_time).seconds / 60 / 60 + (
            date - earliest_time).days * 24
    output['days_from_start'] = (date - earliest_time).days
    output['categorical_id'] = output['id']

    output.to_csv("solar.csv")

    print('Done.')


def download_electricity(args):
    """Downloads electricity dataset from UCI repository."""

    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00321/LD2011_2014.txt.zip'

    data_folder = args.data_folder
    csv_path = os.path.join(data_folder, 'LD2011_2014.txt')
    zip_path = csv_path + '.zip'

    download_and_unzip(url, zip_path, csv_path, data_folder)

    print('Aggregating to hourly data')

    df = pd.read_csv(csv_path, index_col=0, sep=';', decimal=',')
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)

    # used to determine the start and end dates of a series
    output = df.resample('1h').mean().replace(0., np.nan)

    earliest_time = output.index.min()

    df_list = []
    for label in output:
        print('Processing {}'.format(label))
        srs = output[label]

        start_date = min(srs.fillna(method='ffill').dropna().index)
        end_date = max(srs.fillna(method='bfill').dropna().index)

        active_range = (srs.index >= start_date) & (srs.index <= end_date)
        srs = srs[active_range].fillna(0.)

        tmp = pd.DataFrame({'power_usage': srs})
        date = tmp.index
        tmp['t'] = (date - earliest_time).seconds / 60 / 60 + (
                date - earliest_time).days * 24
        tmp['days_from_start'] = (date - earliest_time).days
        tmp['categorical_id'] = label
        tmp['date'] = date
        tmp['id'] = label
        tmp['hour'] = date.hour
        tmp['day'] = date.day
        tmp['day_of_week'] = date.dayofweek
        tmp['month'] = date.month

        df_list.append(tmp)

    output = pd.concat(df_list, axis=0, join='outer').reset_index(drop=True)

    output['categorical_id'] = output['id'].copy()
    output['hours_from_start'] = output['t']
    output['categorical_day_of_week'] = output['day_of_week'].copy()
    output['categorical_hour'] = output['hour'].copy()

    # Filter to match range used by other academic papers
    output = output[(output['days_from_start'] >= 1096)
                    & (output['days_from_start'] < 1346)].copy()

    output.to_csv("electricity.csv".format(args.data_folder))

    print('Done.')


def download_traffic(args):
    """Downloads traffic dataset from UCI repository."""

    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00204/PEMS-SF.zip'

    data_folder = args.data_folder
    csv_path = os.path.join(data_folder, 'PEMS_train')
    zip_path = os.path.join(data_folder, 'PEMS-SF.zip')

    download_and_unzip(url, zip_path, csv_path, data_folder)

    print('Aggregating to hourly data')

    def process_list(s, variable_type=int, delimiter=None):
        """Parses a line in the PEMS format to a list."""
        if delimiter is None:
          l = [
              variable_type(i) for i in s.replace('[', '').replace(']', '').split()
          ]
        else:
          l = [
              variable_type(i)
              for i in s.replace('[', '').replace(']', '').split(delimiter)
          ]

        return l

    def read_single_list(filename):
        """Returns single list from a file in the PEMS-custom format."""
        with open(os.path.join(data_folder, filename), 'r') as dat:
          l = process_list(dat.readlines()[0])
        return l

    def read_matrix(filename):
        """Returns a matrix from a file in the PEMS-custom format."""
        array_list = []
        with open(os.path.join(data_folder, filename), 'r') as dat:

          lines = dat.readlines()
          for i, line in enumerate(lines):
            if (i + 1) % 50 == 0:
              print('Completed {} of {} rows for {}'.format(i + 1, len(lines),
                                                            filename))

            array = [
                process_list(row_split, variable_type=float, delimiter=None)
                for row_split in process_list(
                    line, variable_type=str, delimiter=';')
            ]
            array_list.append(array)

        return array_list

    shuffle_order = np.array(read_single_list('randperm')) - 1  # index from 0
    train_dayofweek = read_single_list('PEMS_trainlabels')
    train_tensor = read_matrix('PEMS_train')
    test_dayofweek = read_single_list('PEMS_testlabels')
    test_tensor = read_matrix('PEMS_test')

    # Inverse permutate shuffle order
    print('Shuffling')
    inverse_mapping = {
      new_location: previous_location
      for previous_location, new_location in enumerate(shuffle_order)
    }
    reverse_shuffle_order = np.array([
      inverse_mapping[new_location]
      for new_location, _ in enumerate(shuffle_order)
    ])

    # Group and reoder based on permuation matrix
    print('Reodering')
    day_of_week = np.array(train_dayofweek + test_dayofweek)
    combined_tensor = np.array(train_tensor + test_tensor)

    day_of_week = day_of_week[reverse_shuffle_order]
    combined_tensor = combined_tensor[reverse_shuffle_order]

    # Put everything back into a dataframe
    print('Parsing as dataframe')
    labels = ['traj_{}'.format(i) for i in read_single_list('stations_list')]

    hourly_list = []
    for day, day_matrix in enumerate(combined_tensor):

        # Hourly data
        hourly = pd.DataFrame(day_matrix.T, columns=labels)
        hourly['hour_on_day'] = [int(i / 6) for i in hourly.index
                                ]  # sampled at 10 min intervals
        if hourly['hour_on_day'].max() > 23 or hourly['hour_on_day'].min() < 0:
          raise ValueError('Invalid hour! {}-{}'.format(
              hourly['hour_on_day'].min(), hourly['hour_on_day'].max()))

        hourly = hourly.groupby('hour_on_day', as_index=True).mean()[labels]
        hourly['sensor_day'] = day
        hourly['time_on_day'] = hourly.index
        hourly['day_of_week'] = day_of_week[day]

        hourly_list.append(hourly)

    hourly_frame = pd.concat(hourly_list, axis=0, ignore_index=True, sort=False)

    # Flatten such that each entitiy uses one row in dataframe
    store_columns = [c for c in hourly_frame.columns if 'traj' in c]
    other_columns = [c for c in hourly_frame.columns if 'traj' not in c]
    flat_df = pd.DataFrame(columns=['values', 'prev_values', 'next_values'] +
                         other_columns + ['id'])

    def format_index_string(x):
        """Returns formatted string for key."""

        if x < 10:
            return '00' + str(x)
        elif x < 100:
            return '0' + str(x)
        elif x < 1000:
            return str(x)

        raise ValueError('Invalid value of x {}'.format(x))

    for store in store_columns:
        print('Processing {}'.format(store))

        sliced = hourly_frame[[store] + other_columns].copy()
        sliced.columns = ['values'] + other_columns
        sliced['id'] = int(store.replace('traj_', ''))

        # Sort by Sensor-date-time
        key = sliced['id'].apply(str) \
          + sliced['sensor_day'].apply(lambda x: '_' + format_index_string(x)) \
            + sliced['time_on_day'].apply(lambda x: '_' + format_index_string(x))
        sliced = sliced.set_index(key).sort_index()

        sliced['values'] = sliced['values'].fillna(method='ffill')
        sliced['prev_values'] = sliced['values'].shift(1)
        sliced['next_values'] = sliced['values'].shift(-1)

        flat_df = flat_df.append(sliced.dropna(), ignore_index=True, sort=False)

    # Filter to match range used by other academic papers
    index = flat_df['sensor_day']
    flat_df = flat_df[index < 173].copy()

    # Creating columns fo categorical inputs
    flat_df['categorical_id'] = flat_df['id'].copy()
    flat_df['hours_from_start'] = flat_df['time_on_day'] \
      + flat_df['sensor_day']*24.
    flat_df['categorical_day_of_week'] = flat_df['day_of_week'].copy()
    flat_df['categorical_time_on_day'] = flat_df['time_on_day'].copy()

    flat_df.to_csv("traffic.csv")
    print('Done.')


def process_favorita(config):
    """Processes Favorita dataset.
    Makes use of the raw files should be manually downloaded from Kaggle @
    https://www.kaggle.com/c/favorita-grocery-sales-forecasting/data
    Args:
    config: Default experiment config for Favorita
    """

    url = 'https://www.kaggle.com/c/favorita-grocery-sales-forecasting/data'

    data_folder = config.data_folder

    # Save manual download to root folder to avoid deleting when re-processing.
    zip_file = os.path.join(data_folder,
                          'favorita-grocery-sales-forecasting.zip')

    if not os.path.exists(zip_file):
        raise ValueError(
            'Favorita zip file not found in {}!'.format(zip_file) +
            ' Please manually download data from Kaggle @ {}'.format(url))

    # Unpack main zip file
    outputs_file = os.path.join(data_folder, 'train.csv.7z')
    unzip(zip_file, outputs_file, data_folder)

    # Unpack individually zipped files
    for file in glob.glob(os.path.join(data_folder, '*.7z')):

        csv_file = file.replace('.7z', '')

        unzip(file, csv_file, data_folder)

    print('Unzipping complete, commencing data processing...')

    # Extract only a subset of data to save/process for efficiency
    start_date = pd.datetime(2015, 1, 1)
    end_date = pd.datetime(2016, 6, 1)

    print('Regenerating data...')

    # load temporal data
    temporal = pd.read_csv(os.path.join(data_folder, 'train.csv'), index_col=0)

    store_info = pd.read_csv(os.path.join(data_folder, 'stores.csv'), index_col=0)
    oil = pd.read_csv(
      os.path.join(data_folder, 'oil.csv'), index_col=0).iloc[:, 0]
    holidays = pd.read_csv(os.path.join(data_folder, 'holidays_events.csv'))
    items = pd.read_csv(os.path.join(data_folder, 'items.csv'), index_col=0)
    transactions = pd.read_csv(os.path.join(data_folder, 'transactions.csv'))

    # Take first 6 months of data
    temporal['date'] = pd.to_datetime(temporal['date'])

    # Filter dates to reduce storage space requirements
    if start_date is not None:
        temporal = temporal[(temporal['date'] >= start_date)]
    if end_date is not None:
        temporal = temporal[(temporal['date'] < end_date)]

    dates = temporal['date'].unique()

    # Add trajectory identifier
    temporal['traj_id'] = temporal['store_nbr'].apply(
      str) + '_' + temporal['item_nbr'].apply(str)
    temporal['unique_id'] = temporal['traj_id'] + '_' + temporal['date'].apply(
      str)

    # Remove all IDs with negative returns
    print('Removing returns data')
    min_returns = temporal['unit_sales'].groupby(temporal['traj_id']).min()
    valid_ids = set(min_returns[min_returns >= 0].index)
    selector = temporal['traj_id'].apply(lambda traj_id: traj_id in valid_ids)
    new_temporal = temporal[selector].copy()
    del temporal
    gc.collect()
    temporal = new_temporal
    temporal['open'] = 1

    # Resampling
    print('Resampling to regular grid')
    resampled_dfs = []
    for traj_id, raw_sub_df in temporal.groupby('traj_id'):
        print('Resampling', traj_id)
        sub_df = raw_sub_df.set_index('date', drop=True).copy()
        sub_df = sub_df.resample('1d').last()
        sub_df['date'] = sub_df.index
        sub_df[['store_nbr', 'item_nbr', 'onpromotion']] \
            = sub_df[['store_nbr', 'item_nbr', 'onpromotion']].fillna(method='ffill')
        sub_df['open'] = sub_df['open'].fillna(
            0)  # flag where sales data is unknown
        sub_df['log_sales'] = np.log(sub_df['unit_sales'])

        resampled_dfs.append(sub_df.reset_index(drop=True))

    new_temporal = pd.concat(resampled_dfs, axis=0)
    del temporal
    gc.collect()
    temporal = new_temporal

    print('Adding oil')
    oil.name = 'oil'
    oil.index = pd.to_datetime(oil.index)
    temporal = temporal.join(
      oil.loc[dates].fillna(method='ffill'), on='date', how='left')
    temporal['oil'] = temporal['oil'].fillna(-1)

    print('Adding store info')
    temporal = temporal.join(store_info, on='store_nbr', how='left')

    print('Adding item info')
    temporal = temporal.join(items, on='item_nbr', how='left')

    transactions['date'] = pd.to_datetime(transactions['date'])
    temporal = temporal.merge(
      transactions,
      left_on=['date', 'store_nbr'],
      right_on=['date', 'store_nbr'],
      how='left')
    temporal['transactions'] = temporal['transactions'].fillna(-1)

    # Additional date info
    temporal['day_of_week'] = pd.to_datetime(temporal['date'].values).dayofweek
    temporal['day_of_month'] = pd.to_datetime(temporal['date'].values).day
    temporal['month'] = pd.to_datetime(temporal['date'].values).month

    # Add holiday info
    print('Adding holidays')
    holiday_subset = holidays[holidays['transferred'].apply(
      lambda x: not x)].copy()
    holiday_subset.columns = [
      s if s != 'type' else 'holiday_type' for s in holiday_subset.columns
    ]
    holiday_subset['date'] = pd.to_datetime(holiday_subset['date'])
    local_holidays = holiday_subset[holiday_subset['locale'] == 'Local']
    regional_holidays = holiday_subset[holiday_subset['locale'] == 'Regional']
    national_holidays = holiday_subset[holiday_subset['locale'] == 'National']

    temporal['national_hol'] = temporal.merge(
      national_holidays, left_on=['date'], right_on=['date'],
      how='left')['description'].fillna('')
    temporal['regional_hol'] = temporal.merge(
      regional_holidays,
      left_on=['state', 'date'],
      right_on=['locale_name', 'date'],
      how='left')['description'].fillna('')
    temporal['local_hol'] = temporal.merge(
      local_holidays,
      left_on=['city', 'date'],
      right_on=['locale_name', 'date'],
      how='left')['description'].fillna('')

    temporal.sort_values('unique_id', inplace=True)

    print('Saving processed file to {}'.format(config.data_csv_path))
    temporal.to_csv("retail.py")


def main(expt_name, force_download, output_folder):

    print('#### Running download script ###')
    expt_config = ExperimentConfig(expt_name, output_folder)

    if os.path.exists(expt_config.data_csv_path) and not force_download:
        print('Data has been processed for {}. Skipping download...'.format(
            expt_name))
        sys.exit(0)
    else:
        print('Resetting data folder...')
        #shutil.rmtree(expt_config.data_csv_path)
        os.makedirs(expt_config.data_csv_path)

    # Default download functions
    download_functions = {
        'electricity': download_electricity,
        'traffic': download_traffic,
        'air_quality': download_air_quality,
        'favorita': process_favorita,
        'watershed': process_watershed,
        'solar': download_solar
    }

    if expt_name not in download_functions:
        raise ValueError('Unrecongised experiment! name={}'.format(expt_name))

    download_function = download_functions[expt_name]

    # Run data download
    print('Getting {} data...'.format(expt_name))
    download_function(expt_config)

    print('Download completed.')


if __name__ == '__main__':
    def get_args():
        """Returns settings from command line."""

        experiment_names = ExperimentConfig.default_experiments

        parser = argparse.ArgumentParser(description='Data download configs')
        parser.add_argument(
            '--expt_name',
            type=str,
            nargs='?',
            choices=experiment_names,
            help='Experiment Name. Default={}'.format(','.join(experiment_names)))
        parser.add_argument(
            '--output_folder',
            type=str,
            nargs='?',
            default='.',
            help='Path to folder for data download')
        parser.add_argument(
            '--force_download',
            type=str,
            nargs='?',
            choices=['yes', 'no'],
            default='yes',
            help='Whether to re-run data download')

        args = parser.parse_args()

        root_folder = None if args.output_folder == '.' else args.output_folder

        return args.expt_name, args.force_download == 'yes', root_folder


    name, force, folder = get_args()
    main(expt_name=name, force_download=force, output_folder=folder)
