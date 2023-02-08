# Digital snow melt - Automated forecasting from snow parameters: TUM DI-LAB

Further Information can be found here: https://www.mdsi.tum.de/di-lab/vergangene-projekte/ws2021-thinkoutside-the-digital-snow-scanner-automated-extraction-of-snow-parameters-from-radar-data/

Final Report: https://www.mdsi.tum.de/fileadmin/w00cet/di-lab/pdf/ThinkOutside_WS2021_FinalReport.pdf

Final Presentation: https://www.mdsi.tum.de/fileadmin/w00cet/di-lab/pdf/ThinkOutside_WS2021_FinalPresentation.pdf

All Work has been done together with Fabienne Greier, Florian Donhauser,  Md. Forhad Hossain and Wudamu.

## Time Series Forecasting
The Model under ResCNN_LSTM can be used to forecast any kind of Time Series with seasonal patterns. Within this project we forecasted the Volume of Wwaterinflow into Reservoir Lakes. The Model combines Residual Connections, Convolutional and LSTM Layers. The input data is passed through the network by sliding window inputs. You will need to adjust some functions in ResCNN_LSTM_fit.py to your needs.

## Abstract
The forecast of seasonal snowmelt and inflow into water reservoirs based on automatic extraction of snow parameters plays a crucial role in the hydropower industry to plan and deliver energy more efficiently.
This project aims to research appropriate Machine Learning (ML) methods for predicting the water inflow of specific reservoir lakes, implement a model based on an opensource ML model, train it with Norwegian data, and optimize the model prediction performance.
To retrieve the Norwegian data, a DataLoader was created that downloads the weather station data from a website.
In the downloaded data, many measurements were missing. Therefore, several techniques were developed to fill in the missing values.
The target data of inflow into a water reservoir, provided by customers of the Norwegian startup ThinkOutside, had large fluctuations, so the temporal resolution had to be changed.
After cleaning up and restructuring the existing code that served as a basis for the model, hyperparameter tuning was performed using Optuna to find the best configurations.
Different prediction time frames were tested to evaluate the performance of the ML model. Overall, the accuracy of the predictions increased with tuned hyperparameters. However, some stations with noise in the year prior to the forecast showed poor performance. For these stations, it is recommended to use coarser methods for resampling the data.

## Dependencies
We have created a requirements.txt file which contains all python-packages needed to run the code.
To install these packages, we suggest creating a virtual environment to run all scripts.

## How to trigger the different scripts
- First of all make sure, to have some suiting input data. We have used historic 10-15 years weekly data.
- As a next step we can trigger the script /ResCNN_LSTM/hyperparameter_tuning.py which will find the best possible hyperparameters for the specified station, date setting in /ResCNN_LSTM/config.yaml
- Once this has run successfully we can have a look into /ResCNN_LSTM/hypertune_configs/configs_evaluation.txt and find the best config setting
- Now we just have to modify the path to this specific setting in /ResCNN_LSTM/ResCNN_LSTM_fit.py in the very last line of the code
