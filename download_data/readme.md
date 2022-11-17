# Data download function
## This class will create the link for download the Snow-water equivalent data and Water inflow data for selected station
### Requeirments:
- pip install request
### How to use it:
config_path = "config.yaml"

data = DataLoader(config_path)

file_swe_path = "data_swe.csv"
#### first check the given path is available, if the data not in the local, donwnload it from website
data_swe = data.load_snow_water_equivalent(file_swe_path) 

file_water_level_path = "data_water_level.csv"

data_water_level = data.load_water_inflow(file_water_level_path)
#### donwload water inflow and swe data together
data_water_level_swe = data.water_inflow_swe(file_path)


