from climata.snotel import StationDailyDataIO
from climata.snotel import StationIO
import pandas as pd
from functools import reduce
import datetime
from tqdm import trange


def getNRCS(station_id, param_id, nyears, frequency):
    """
        Download individual time series data from NRCS API.

        Parameters
        ----------
        station_id (string): name of the station for which we download the data
        param_id (string): filter data wih these parameters
        nyears (int): number of years to download
        frequency (string): The offset string or object representing target conversion for ``pandas.DataFrame.resample``

        Returns
        -------
        success (bool): True for success or False for failure
        df (Union[pandas.DataFrame, string]): dataframe containing the data or string containing the error message

    """
    ndays, yesterday = 365 * nyears, datetime.date.today()
    datelist = pd.date_range(end=yesterday, periods=ndays).tolist()

    data = StationDailyDataIO(
        start_date=datelist[0],
        end_date=datelist[-1],
        station=station_id,
        parameter=param_id,
    )
    if len(data.data) == 0:
        return False, 'The data source is empty for station ' + station_id
    temp = pd.DataFrame(data.data[0]['data'].data)
    df = pd.DataFrame(temp['value'], columns=['value'])
    df.index = pd.to_datetime(temp['date'])
    df.index.name = 'Date'
    if df.index[-1].year != datetime.date.today().year:
        return False, 'Either today is new years, or the gap in data is too large for station ' + station_id
    if df.index[0].year > 1990:
        return False, 'The starting year in the time series is too recent for station ' + station_id
    df.columns = [station_id]
    missing_data = len(df) - len(df.dropna())
    if missing_data > 100:
        return False, 'Today is definitely not new years, but the gap in the data is still too large for station ' + station_id
    if missing_data > 0:
        new = pd.DataFrame()
        cols = df.columns
        interp_df = [new.assign(Interpolant=df[i].interpolate(method='time')) for i in cols][0]
        interp_df.columns = [station_id]
        resampled_df = interp_df.resample(frequency).mean()
        return True, resampled_df
    else:
        resampled_df = df.resample(frequency).mean()
        return True, resampled_df


def bulk_download(parameter, years, frequency):
    """
        Download multiple years from NRCS API based on a datatype.

        Parameters
        ----------
        parameter (string): parameter for StationIO and getNRCS (datatype for time series files)
        years (int): number of years to download
        frequency (string): The offset string or object representing target conversion for pandas.DataFrame.resample

        Returns
        -------
        time_series_df (pandas.DataFrame): The dataframe with the timeseries
        metadata (pandas.DataFrame): The dataframe with the metadata (name, lat, lng) for the stations

    """
    # Download snow water equivalent data across central and northern Utah
    stations = StationIO(state='UT', parameter=parameter, min_latitude=40.3, max_latitude=40.8, min_longitude=-111.90,
                         max_longitude=-110.45)
    station_ids = [stations.data[i]['stationTriplet'] for i in range(len(stations))]
    station_names = [stations.data[i]['name'] for i in range(len(stations))]
    dflist = []
    stations_trange = trange(len(stations))
    for i in stations_trange:
        success, ncrs = getNRCS(station_ids[i], parameter, years, frequency)
        if success:
            dflist.append(ncrs)
        else:
            stations_trange.set_postfix_str(ncrs)

    time_series_df = reduce(lambda x, y: pd.merge(x, y, on='Date'), dflist)
    # Populate metadata
    metadata = pd.DataFrame(station_ids)
    metadata.columns = ['id']
    metadata['name'] = station_names
    metadata['lat'] = [stations.data[i]['latitude'] for i in range(len(stations))]
    metadata['lng'] = [stations.data[i]['longitude'] for i in range(len(stations))]
    metadata = metadata[metadata['id'].isin(time_series_df.columns)]
    time_series_df.columns = metadata['name']
    return time_series_df, metadata


def main():
    # Define object for Upper Stillwater Reservoir storage volume (ac-ft) time series data
    print("Downloading data for Upper Stillwater")
    success, sv = getNRCS('09278000:UT:BOR', 'RESC', 31, 'W')
    sv.columns = ['Upper Stillwater']
    # Keep only values from 1990 to the present date
    sv = sv[sv.index.year >= 1990]
    # Define object for all SWE monitoring stations surrounding Upper Stillwater Reservoir
    print("Downloading data with bulk_download")
    swe, swe_metadata = bulk_download('WTEQ', 31, 'W')
    # Combine SV and SWE together into single dataframe
    data = reduce(lambda x, y: pd.merge(x, y, left_index=True, right_index=True), [sv, swe])
    # Save data as project_data.csv
    data.to_csv('project_data.csv')
    # How to read the data from project_data.csv
    # data = pd.read_csv('project_data.csv', index_col="Date")


if __name__ == "__main__":
    main()