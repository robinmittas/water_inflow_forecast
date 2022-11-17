"""
this class is download the data for snow water equivalent(swe) and water level, the output of this function is a csv file which
include swe for selected station(define this in config file) and also water inflow m3/s selected station(define this in config file)
based on start time and end time.
"""
import requests
import pandas as pd
import yaml
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


class DataLoader:
    def __init__(self, config_path):
        """
        Constructor of DataLoader class.

        :param config_path:    path of the configuration file
        """
        with open(config_path, encoding='utf8') as file:
            self.config_file = yaml.load(file, Loader=yaml.FullLoader)

        self.station_name = self.config_file["station_name"]
        self.features = self.config_file["features"]
        self.start_time = self.config_file["start_time"]
        self.end_time = self.config_file["end_time"]
        self.years_range = range(self.start_time[-1], self.end_time[-1] + 1)
        self.stations_information = pd.read_csv("station_information.csv", encoding='utf-8')
        self.feature_link_dict = {"swe": "&Parameter=2003%20%20%20%20%20%20", "temperatur": "&Parameter=17%20%20%20%20%20%20", "relative humidity": "&Parameter=2%20%20%20%20%20%20",
                                  "wind direction and speed": "&Parameter=14%20%20%20%20%20%20", "precipitation": "&Parameter=0%20%20%20%20%20%20", "wind speed": "&Parameter=15%20%20%20%20%20%20",
                                  "snow depth": "&Parameter=2002%20%20%20%20%20%20", "ground water level": "&Parameter=5130%20%20%20%20%20%20", "stage": "&Parameter=1000%20%20%20%20%20%20",
                                  "reservoir volume": "&Parameter=1004%20%20%20%20%20%20", "discharge": "&Parameter=1001%20%20%20%20%20%20"}
        self.basic_link = "https://hydapi.nve.no/api/v1/Download?"
        self.time_resolution = self.config_file["time_resolution"][0]
        self.data_frame = pd.DataFrame()
        self.headers = {
            'User-Agent': "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) ""Version/15.0 Safari/605.1.15"}
        self.save_path = os.path.join(self.config_file["output_path"], self.config_file["save_name"])
        self.save_path_prepared = os.path.join(self.config_file["output_path"], "prepared_"+self.config_file["save_name"])
        self.run_preparation = self.config_file["run_preparation"]
        self.drop_cols = self.config_file["drop_cols"]
        self.consecutive_days_with_nulls = self.config_file["consecutive_days_with_nulls"]
        self.threshold_linear_inter = self.config_file["threshold_linear_inter"]
        self.interpolation_method = self.config_file["interpolation_method"]
        self.order_interpolation = self.config_file["order_interpolation"]
        self.fill_nan_method = self.config_file["fill_nan_method"]
        self.plot_columns = self.config_file["plot_columns"]

    def is_valid_start_date(self, station_start_date):
        """
        Check if the start date is valid or not.

        :param station_start_date: start date of the station

        :return: True if the date is valid or False if the date is not valid
        """
        if self.start_time[2] > int(station_start_date[2]):
            return True
        elif self.start_time[2] == int(station_start_date[2]) and self.start_time[1] > int(station_start_date[1]):
            return True
        elif self.start_time[2] == int(station_start_date[2]) and self.start_time[1] == int(station_start_date[1]) \
                and self.start_time[0] >= int(station_start_date[0]):
            return True

        return False

    def download_data(self, download_link, feature, station_name):
        """
        Donwload the data based on the link
        :param download_link:                       the link for download data
        :param feature:                             feature name
        :param station_name:                        station name

        """
        page_bytes = requests.get(url=download_link, headers=self.headers).content  # do the request for download data
        page_json = page_bytes.decode('utf8').replace("'", '"')
        page_list = page_json.split("\r\n")
        page_list_len = len(page_list)
        for index, data_temp in enumerate(page_list):
            if index == 0:
                pass
            elif index == 1:
                pass
            elif index == page_list_len - 1:
                break
            else:
                time_stamp = data_temp.split(";")[0][:-1].split("-")
                time_stamp = time_stamp[2][0:2] + "-" + time_stamp[1] + "-" + time_stamp[0] + time_stamp[2][2:]
                str_time = data_temp.split(";")[1]
                if str_time != "":
                    self.data_frame.loc[time_stamp, station_name + "/" + " " + feature] = float(data_temp.split(";")[1])

    def create_download_link(self, station_id, station_start_time, station_end_time, feature_link, time_resolution):
        """
        Creat the link for downloading data
        :param station_id:               every station corresponding a station id, dowmload the data according its id e.g. -> 213.7.0
        :param station_start_time:       start download time -> 03102019
        :param station_end_time:         end download time -> 03102020
        :param feature_link:             specfic link for feature
        :return:                         link for download
        """
        if time_resolution == "day":
            time_resolution = 1440
        elif time_resolution == "hours":
            time_resolution = 60
        else:
            time_resolution = 0
        start_time = str(station_start_time[2]) + '-' + str(station_start_time[1]) + '-' + str(station_start_time[0]) + '/'
        end_time = str(station_end_time[2]) + '-' + str(station_end_time[1]) + '-' + str(station_end_time[0])
        download_link = self.basic_link + "StationId=" + station_id + "%20%20%20%20%20%20" + feature_link + "&VersionNo=0%20%20%20%20%20%20" + "&ResolutionTime=" + str(
            time_resolution) + "%20%20%20%20%20%20&ReferenceTime=" + start_time + end_time + "%20%20%20%20%20%20&Percentiles=false%20%20%20%20%20%20&Flood=false%20%20%20%20%20%20&Language=en"
        return download_link

    def download(self):
        for station_name in self.station_name:
            for feature in self.features:
                station_index = list(self.stations_information["station name"].values).index(station_name)
                availiable = self.stations_information.loc[station_index, feature]
                addition_information = self.stations_information.loc[station_index, "additional information"]
                if availiable == "yes" and addition_information is not None:
                    print("downloading the %s for %s" % (feature, station_name))
                    station_start_date = self.stations_information.loc[station_index, "serie startet:" + str(feature)].split(".")
                    download_start_time = self.start_time
                    if not self.is_valid_start_date(station_start_date):
                        download_start_time = station_start_date
                    station_id = self.stations_information.loc[station_index, "station ID"]
                    download_link = self.create_download_link(station_id, download_start_time, self.end_time, self.feature_link_dict[feature], self.time_resolution)
                    self.download_data(download_link, feature, station_name)
                else:
                    # TODO: have to deal with the additioanl information, additional information means precipitation in some station not availiable
                    continue

        # Set index to correct datetime
        self.data_frame.index = pd.to_datetime(self.data_frame.index, dayfirst=True)
        # sort the data based on date
        self.data_frame = self.data_frame.sort_index()
        # Write out Downloaded data
        self.data_frame.to_csv(self.save_path)


    def prepare_data(self):
        """
        This function handles the missing values by interpolation/ filling by mean of station over other years (by day, month)
        :return: after_interpolated_df, before_interpolation_df
        """
        ##########################################################################################
        # Prepare the data
        ##########################################################################################
        # For simplicity set df = self.data_frame
        df = self.data_frame

        # Make deep copy as we want to plot before and after interpolation later
        df_before_interpolation = df.copy(deep=True)

        # Throw Columns which habe more then specified NA Values
        not_na_threshold = df.shape[0] * self.drop_cols
        for column in df.columns:
            na_counter = df[column].isnull().sum()
            if na_counter > not_na_threshold:
                df = df.drop(columns=column)
                print(f"Column {column} is dropped as col has more then {self.drop_cols * 100}% total NA-Values")
            else:
                for year in self.years_range:
                    na_counter_year = df[df.index.year == year][column].isnull().sum().sum()
                    na_share = na_counter_year / df[df.index.year == year].shape[0]

                    # Count consecutive NA Values. If there are less then specified days in a row with Null values --> interpolate
                    na_groups = df[df.index.year == year][column].notna().cumsum()[df[df.index.year == year][column].isna()]
                    lengths_consecutive_na = na_groups.groupby(na_groups).agg(len).max()


                    # If the na Share for this year is bigger then the threshold for linear interpolation, take means
                    if na_share > self.threshold_linear_inter or lengths_consecutive_na > self.consecutive_days_with_nulls:
                        if self.fill_nan_method == "fill_nan_by_daily_avg":
                            df = self.fill_nan_by_daily_avg(df=df, column=column, year=year)

                        elif self.fill_nan_method == "fill_nan_by_same_features_avg":
                            df = self.fill_nan_by_same_features_avg(df=df, column=column, year=year)
                            # after filling NA values by same features avg, there might still be na values --> for these NAs take avg of feature of other years
                            na_counter_after = df[df.index.year == year][column].isnull().sum().sum()
                            if na_counter_after > 0:
                                df = self.fill_nan_by_daily_avg(df=df, column=column, year=year)

                    # else do interpolation
                    else:
                        df.loc[df.index.year == year, column] = df.loc[df.index.year == year, column].interpolate(method=self.interpolation_method, limit_direction="both", order=self.order_interpolation)
                        print(f"NA Values for column {column} in year {year} are interpolated")

        # Write out prepared data
        df.to_csv(self.save_path_prepared)
        return df, df_before_interpolation

    def fill_nan_by_same_features_avg(self, df: pd.DataFrame, column: str, year: int):
        """
        This function will fill na values by the average of same features (e.g. all swe/ precipitation or temperature) columns of same day
        :param df: dataframe, downloaoded data
        :param column: over which column we fill na
        :param year: over which year we fill na value
        :return: df with filled na values of this year for specified column
        """
        feature = column.split("/")[1]
        same_feature_column = []
        for col in df:
            if feature in col:
                same_feature_column.append(col)
            else: continue
        avg_feature = df[df.index.year==year][same_feature_column].mean(axis=1)
        df.loc[df.index.year==year, column] = df.loc[df.index.year==year, column].fillna(avg_feature)
        print(f"NA Values for column {column} in year {year} are filled with mean values same feature of same day")
        return df

    def fill_nan_by_daily_avg(self, df: pd.DataFrame, column: str, year: int):
        """
        This function will fill na values by the average of specified column over all other available years
        :param df: dataframe, downloaoded data
        :param column: over which column we fill na
        :param year: over which year we fill na value
        :return: df with filled na values of this year for specified column
        """
        # Calculate series with average of this column feature/ station over all years, grouped by month, day
        avg_feature = pd.DataFrame(df[column].groupby([df.index.month, df.index.day]).mean()).reset_index()[column]
        # some years have 366, others 365
        avg_feature = avg_feature.iloc[0:df[df.index.year == year].shape[0]]
        avg_feature.index = df[df.index.year == year].index
        df.loc[df.index.year == year, column] = df.loc[df.index.year == year, column].fillna(avg_feature)
        print(f"NA Values for column {column} in year {year} are filled with mean values of other years")
        return df

    def plot_original_data(self, df:pd.DataFrame):
        """
        this function will plot original dataframe
        @param df: the original data frame
        @return: None
        """
        for col_feature in self.plot_columns:
            fig, axs = plt.subplots(int(len(self.years_range) / 5), 5, figsize=(10, 7))
            fig.suptitle('Station/Feature: %s' % col_feature, fontsize=16)
            for year, ax in zip(self.years_range, axs.flat):
                ax.plot(df[df.index.year == year][col_feature])
                miss_num = df.loc[df.index.year == year, col_feature].isnull().sum()
                miss_rate = round(miss_num / df[df.index.year == year].shape[0], 2)
                # TODO: Remove Jan of the next year on the x-axes of the current year
                ax.set_title("year = %s  missing rate = %s" % (year, str(round(miss_rate * 100, 1)) + "%"), fontsize=7)
                #ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=range(1, 13)))
                ax.xaxis.set_minor_locator(plt.MaxNLocator(12))
                ax.xaxis.set_major_locator(plt.MaxNLocator(6))
                ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]))
                ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 3, 5, 7, 9, 11]))
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
                ax.tick_params(axis='x', which='major', labelsize=6, labelrotation=45)
            fig.tight_layout()
            plt.show()

    def plot_interpolated_data(self, df_interpolated: pd.DataFrame, df_before_interpolation: pd.DataFrame):
        """
        plot the original and interpolated data
        @param df_interpolated: the interpolated data frame
        @param df_before_interpolation: the original data frame which include the nan value
        @return: None
        """
        for col_feature in self.plot_columns:
            fig, axs = plt.subplots(int(len(self.years_range) / 5), 5, figsize=(10, 7))
            fig.suptitle('Station/Feature: %s' % col_feature, fontsize=16)
            for year, ax in zip(self.years_range, axs.flat):
                ax.plot(df_before_interpolation[df_before_interpolation.index.year == year][col_feature], linewidth=7.0,
                        label='original')
                ax.plot(df_interpolated[df_interpolated.index.year == year][col_feature],
                        label="interpolation + original")
                ax.set_title("year = %s" % (year), fontsize=7)
                # TODO: Remove Jan of the next year on the x-axes of the current year
                # ax.xaxis.set_minor_locator(mdates.MonthLocator())
                ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]))
                ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 3, 5, 7, 9, 11]))
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
                ax.tick_params(axis='x', which='major', labelsize=6, labelrotation=45)
                handles, labels = ax.get_legend_handles_labels()
            fig.tight_layout()
            fig.legend(handles, labels, loc='upper left')
            plt.show()




if __name__ == "__main__":
    config_path = "config.yaml"
    data = DataLoader(config_path)
    data.download()

    if data.run_preparation:
        df_interpolated, df_before_interpolation = data.prepare_data()
        data.plot_original_data(df_before_interpolation)
        data.plot_interpolated_data(df_interpolated, df_before_interpolation)