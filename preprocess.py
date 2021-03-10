import numpy as np
import pandas as pd
import openpyxl
import torch
import pickle
from sklearn.preprocessing import StandardScaler
import argparse
import matplotlib.pyplot as plt


class Scaler:
    def __init__(self, site):
        self.site = site
        self.scalers = dict()

    def add_scaler(self, f_n, scaler):
        self.scalers[f_n] = scaler


class Data:
    def __init__(self, site_data, ts, n_features, in_seq_len, out_seq_len):

        self.scalers = list()
        self.sites_data = site_data
        self.ts = ts
        self.n_seasons = 4
        self.hist = 4

        self.nf = n_features * self.hist
        '''self.I = I
        self.J = J'''
        self.in_seq_len = in_seq_len
        self.out_seq_len = out_seq_len
        self.ln = self.ts - (self.in_seq_len + self.out_seq_len)
        self.inputs = torch.zeros((self.ln, self.in_seq_len, self.nf))
        self.outputs = torch.zeros((self.ln, self.out_seq_len, 1))
        self.create_raster()
        '''self.outputs = torch.reshape(self.outputs, (self.outputs.shape[0], -1,
                                                    self.outputs.shape[2] * self.outputs.shape[3]))'''

        pickle.dump(self.inputs, open("inputs.p", "wb"))
        pickle.dump(self.outputs, open("outputs.p", "wb"))
        pickle.dump(self.scalers, open("scalers.pkl", "wb"))

    def create_raster(self):

        for abr, df_site in self.sites_data.items():

            scalers_per_site = Scaler(abr)
            self.scalers.append(scalers_per_site)

            len = int(self.nf / self.hist)
            f_ind = 0
            for f in range(len):

                stScaler = StandardScaler()
                dat = df_site.iloc[:, f + 1]
                dat = np.array(dat).reshape(-1, 1)
                stScaler.fit(dat)
                dat = stScaler.transform(dat)
                scalers_per_site.add_scaler(f, stScaler)
                dat = torch.from_numpy(np.array(dat).flatten())
                in_data, out_data = self.get_window_data(dat)
                self.inputs[:, :, f_ind:f_ind+self.hist] = in_data
                f_ind = f_ind + self.hist
                if f == 1:
                    self.outputs[:, :, 0] = out_data

    def create_one_hot(self, df):

        months = df["Date"].dt.month
        b = self.ts - self.in_seq_len*2
        one_hot = torch.zeros((b, self.in_seq_len, self.n_seasons))
        for i in range(b):
            for s in range(self.in_seq_len):
                m = months.iloc[i+s]
                j = 1 if m <= 3 else 2 if m > 3 & m <= 6 else 3 if m > 6 & m <= 9 else 4
                one_hot[i, s, j - 1] = 1
        return one_hot

    def get_window_data(self, data):

        data_2d_in = torch.zeros((self.ts, self.hist))
        data_3d_in = torch.zeros((self.ln, self.in_seq_len, self.hist))
        data_out = torch.zeros((self.ln, self.out_seq_len))

        for i in range(0, self.ts):

            if i < self.hist:
                data_2d_in[i, :self.hist - i] = torch.zeros(self.hist - i)
                data_2d_in[i, self.hist - i:] = data[0:i]

            else:
                data_2d_in[i, :] = data[i - self.hist:i]
        j = 0
        for i in range(0, self.ts):
            if j < self.ln:
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
        #self.site_abrs = ["BEF", "GOF", "DCF", 'MCQ', "WHB"]
        self.sites_data = dict()
        site_dat = self.prep_data_per_site(params.site)
        ln = len(site_dat)
        self.sites_data[params.site] = site_dat

        #self.grid = self.create_grid()
        '''for abr in self.site_abrs:
            self.sites_data[abr] = self.prep_data_per_site(abr)

        dates = self.sites_data["BEF"]["Date"].values
        for abr2, df2 in self.sites_data.items():
            df2 = df2[np.in1d(df2["Date"].values, dates)]
            self.sites_data[abr2] = df2

        self.min_len = len(list(self.sites_data.values())[0])

        for key in self.sites_data.keys():
            site_ln = len(self.sites_data[key])
            if site_ln < self.min_len:
                self.min_len = site_ln'''

        self.raster = Data(self.sites_data, ln, self.n_features,
                           params.in_seq_len, params.out_seq_len)

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
        plt.plot(np.arange(0, 1500), df.SpConductivity.iloc[-1500:])
        plt.tick_params(axis="x", bottom=False, top=False)
        plt.tick_params(axis="y", left=False, right=False)
        plt.show()
        df = df.ffill().bfill()
        return df


def main():

    parser = argparse.ArgumentParser(description="preprocess argument parser")
    parser.add_argument("--in_seq_len", type=int, default=128)
    parser.add_argument("--out_seq_len", type=int, default=28)
    parser.add_argument("--site", type=str, default="WHB")
    params = parser.parse_args()
    stdata = STData("data/metadata.xlsx", "data", params)


if __name__ == '__main__':
    main()