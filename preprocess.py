import numpy as np
import pandas as pd
import openpyxl
import torch
import pickle
from sklearn.preprocessing import StandardScaler
import argparse
import matplotlib.pyplot as plt
import pywt


class Scaler:
    def __init__(self, site):
        self.site = site
        self.scalers = dict()

    def add_scaler(self, f_n, scaler):
        self.scalers[f_n] = scaler


class Data:
    def __init__(self, site_data, max_len, n_features, in_seq_len, out_seq_len, trn_per):

        self.scalers = list()
        self.sites_data = site_data
        self.ts = max_len
        self.n_seasons = 4
        self.moving_averages = [4, 8, 16, 32]
        self.n_moving_average = len(self.moving_averages)
        self.derivative = [4, 8, 16, 32]
        self.n_derivative = 0
        self.wavelets = ['db3', 'db5']
        self.n_wavelets = 0
        self.nf = n_features * (self.n_moving_average + self.n_wavelets + self.n_derivative)
        self.in_seq_len = in_seq_len
        self.out_seq_len = out_seq_len

        self.train_ts = int(self.ts * trn_per)
        self.valid_ts = int((self.ts - self.train_ts) * 0.5)
        self.test_ts = self.valid_ts

        self.train_ln = self.get_length(self.train_ts)
        self.valid_ln = self.get_length(self.valid_ts)
        self.test_ln = self.get_length(self.test_ts)

        self.train_x = torch.zeros(self.train_ln, self.in_seq_len, self.nf)
        self.train_y = torch.zeros(self.train_ln, self.out_seq_len, 1)
        self.valid_x = torch.zeros(self.valid_ln, self.in_seq_len, self.nf)
        self.valid_y = torch.zeros(self.valid_ln, self.out_seq_len, 1)
        self.test_x = torch.zeros(self.test_ln, self.in_seq_len, self.nf)
        self.test_y = torch.zeros(self.test_ln, self.out_seq_len, 1)

        for abr, df_site in self.sites_data.items():
            df_site = df_site.iloc[-self.ts:]
            self.train_x, self.train_y = \
                self.create_raster(df_site[:self.train_ts], self.train_ln, self.train_x, self.train_y)
            self.valid_x, self.valid_y = \
                self.create_raster(df_site[self.train_ts:self.train_ts+self.valid_ts], self.valid_ln, self.valid_x, self.valid_y)
            self.test_x, self.test_y = \
                self.create_raster(df_site[-self.test_ts:], self.test_ln, self.test_x, self.test_y)

        pickle.dump(self.train_x, open("train_x.p", "wb"))
        pickle.dump(self.train_y, open("train_y.p", "wb"))
        pickle.dump(self.valid_x, open("valid_x.p", "wb"))
        pickle.dump(self.valid_y, open("valid_y.p", "wb"))
        pickle.dump(self.test_x, open("test_x.p", "wb"))
        pickle.dump(self.test_y, open("test_y.p", "wb"))

    def get_length(self, len):
        ln = len - (self.in_seq_len + self.out_seq_len)
        ln = int(ln / (self.n_moving_average + self.n_wavelets + self.n_derivative))
        return ln

    def create_raster(self, data, ln, inputs, outputs):

        length = int(self.nf / (self.n_moving_average + self.n_wavelets + self.n_derivative))
        f_ind = 0
        ts = len(data)
        for f in range(length):

            dat = data.iloc[:, f + 1]
            dat = np.array(dat).reshape(-1, 1)
            dat = torch.from_numpy(np.array(dat).flatten())
            in_data, out_data = self.get_window_data(dat, ln, ts)
            inputs[:, :, f_ind:f_ind+self.n_moving_average+self.n_wavelets+self.n_derivative] = in_data
            f_ind = f_ind + self.n_moving_average
            if f == 1:
                outputs[:, :, 0] = out_data

        return inputs, outputs

    @staticmethod
    def moving_average(len, data, ts):

        data_mv = torch.zeros((ts, 1))
        for i in range(0, ts):
            if i < len:
                n = 1 if i == 0 else i
                data_mv[i, 0] = sum(data[0:i])/n
            else:
                data_mv[i, 0] = sum(data[i-len:i])/len
        return data_mv

    @staticmethod
    def get_derivative(k, data, ts):

        data_dv = torch.zeros((ts, 1))
        for i in range(0, ts):
            if i+k < ts:
                if i < k:
                    data_dv[i, 0] = data[i+k] - data[0]
                else:
                    data_dv[i, 0] = data[i+k] - data[i-k]
        return data_dv

    def create_wavelet(self, type, data):
        ca, cd = pywt.dwt(data.detach().numpy(), type)
        return torch.FloatTensor(ca)

    def get_window_data(self, data, ln, ts):

        data_2d_in = torch.zeros((ts, self.n_moving_average+self.n_wavelets+self.n_derivative))
        data_3d_in = torch.zeros((ln, self.in_seq_len,
                                  self.n_moving_average+self.n_wavelets+self.n_derivative))
        data_out = torch.zeros((ln, self.out_seq_len))

        for i, mv in enumerate(self.moving_averages):

            data_2d_in[:, i] = self.moving_average(mv, data, ts).squeeze(1)
            #data_2d_in[:, i+self.n_moving_average] = self.get_derivative(mv, data, ts).squeeze(1)

        j = 0
        for i in range(0, self.ts):
            if j < ln:
                data_3d_in[j, :, :] = data_2d_in[i:i+self.in_seq_len, :]
                data_out[j, :] = data[i+self.in_seq_len:i + self.in_seq_len + self.out_seq_len]
                j += 1
        return data_3d_in, data_out


class STData:
    def __init__(self, meta_path, site_path, params):
        self.meta_path = meta_path
        self.site_path = site_path
        self.I = 3
        self.J = 6
        self.n_features = 3
        self.sites_data = dict()
        site_dat = self.prep_data_per_site(params.site)
        self.sites_data[params.site] = site_dat
        self.raster = Data(self.sites_data, params.max_length, self.n_features,
                           params.in_seq_len, params.out_seq_len, params.train_percent)

    class Site:
        def __init__(self, name, abr, lat, long):

            self.name = name
            self.abr = abr
            self.lat = lat
            self.long = long

    def move_near(self, grid, key, x, y):

        if y+1 < self.J and grid[x, y+1] is None:
            grid[x, y + 1] = key
        elif x+1 < self.I and grid[x+1, y] is None:
            grid[x + 1, y] = key
        elif x-1 >= 0 and grid[x-1, y] is None:
            grid[x - 1, y] = key
        elif y-1 >= 0 and grid[x, y-1] is None:
            grid[x, y - 1] = key

    def create_grid(self):

        wb_site = openpyxl.load_workbook(self.meta_path, data_only=True)
        sheet_site = wb_site.get_sheet_by_name("Site Info")
        max_row = sheet_site.max_row

        site_list = list()

        for row in range(2, max_row + 1):
            abr = sheet_site.cell(row=row, column=2).value
            name = sheet_site.cell(row=row, column=1).value
            lat = sheet_site.cell(row=row, column=10).value
            long = sheet_site.cell(row=row, column=11).value
            site_list.append(self.Site(name, abr, lat, long))

        min_lat, min_long, max_lat, max_long = 100, 100, -100, -100

        for site in site_list:

            lat, long = site.lat, site.long

            if lat < min_lat:
                min_lat = lat
            if lat > max_lat:
                max_lat = lat
            if long < min_long:
                min_long = long
            if long > max_long:
                max_long = long

        grid = None
        for i in range(self.I):
            ls = list()
            for j in range(self.J):
                ls.append(None)
            if grid is None:
                grid = ls
            else:
                grid = np.vstack((grid, ls))

        grid = np.array(grid)

        lat_diff = (max_lat - min_lat) / (self.I - 1)
        long_diff = (max_long - min_long) / (self.J - 1)

        for site in site_list:

            lat, long = site.lat, site.long
            x = int((lat - min_lat) / lat_diff)
            y = int((long - min_long) / long_diff)

            if grid[x, y] is None:
                grid[x, y] = site.abr
            else:
                self.move_near(grid, site.abr, x, y)

        grid_site = dict()

        for i in range(self.I):
            for j in range(self.J):
                if grid[i, j] is not None:
                    grid_site[grid[i, j]] = [i, j]

        return grid_site

    def prep_data_per_site(self, abr):

        df = pd.read_csv("{}/{}_WQual_Level4.csv".format(self.site_path, abr))
        df["Date"] = pd.to_datetime(df["Date"])
        df["Date"] = df["Date"].dt.normalize()
        start_date = "2012-12-14"
        end_date = "2019-01-02"
        mask = (df["Date"] >= start_date) & (df["Date"] <= end_date)
        df = df[mask]
        df = df[["Date", "TempC", "SpConductivity", "Q"]]
        '''plt.plot(np.arange(0, 1500), df.SpConductivity.iloc[-1500:], color='k')
        plt.tick_params(axis="x", bottom=False, top=False)
        plt.tick_params(axis="y", left=False, right=False)
        #plt.axis('off')
        plt.show()'''
        df = df.ffill().bfill()
        return df


def main():

    parser = argparse.ArgumentParser(description="preprocess argument parser")
    parser.add_argument("--in_seq_len", type=int, default=128)
    parser.add_argument("--out_seq_len", type=int, default=64)
    parser.add_argument("--site", type=str, default="WHB")
    parser.add_argument("--train_percent", type=float, default=0.9)
    parser.add_argument("--max_length", type=int, default=2500)
    params = parser.parse_args()
    stdata = STData("data/metadata.xlsx", "data", params)


if __name__ == '__main__':
    main()