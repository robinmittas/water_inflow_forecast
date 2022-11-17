import datetime
from sklearn.preprocessing import MinMaxScaler
import time
from datetime import timedelta
import pandas as pd
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
import tensorflow as tf
from ResCNN_LSTM_Model import ResCNN_LSTM_Model
import yaml
from functools import reduce
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
import matplotlib.pyplot as plt
import os

seed = 19
tf.random.set_seed(seed)
config = None



def sliding_window_input(df, input_length, idx):
    """
    Parameters:
    -------------
    df (pandas.df)                  -- pandas df
    input_length (int)              -- length of window, e.g. 20 weeks
    idx (int)                       -- index; basically starting at the beinning of df and take first 20 weeks, afterwards increase idx by 1
    :return:                        -- numpy.ndarray output of waterlevel in reservoir lake
    -------------
    """
    window = df.values[idx:input_length + idx, :]
    return window


def sliding_window_output(df, input_length, output_length, idx):
    """
    Parameters:
    -------------
    df (pandas.df)                  -- pandas df
    input_length (int)              -- length of window, e.g. 20 weeks
    output_length (int)             -- how many weeks we want to predict (e.g. 15 weeks): take the 15 weeks which follow to 20 input weeks
    idx (int)                       -- index; basically starting at the beinning of df and take first 20 weeks, afterwards increase idx by 1
    :return:                        -- numpy.ndarray output of waterlevel in reservoir lake
    -------------
    """
    window = df.values[input_length + idx:input_length + output_length + idx]
    return window


def prepare_data_norwegian(filepath_input: str, filepath_inflow: str, resample_target: str, resample_input: str,
                           interpolate_target:str, order_interpolation:int,
                           n_timesteps: int, n_outputs: int, training_input_end_date: datetime.date,
                           target_name: list, keep_other_inflow_columns: bool, split_y_test: bool):
    """
        Parameters:
        -------------
        filepath_input (str)                      -- the path of downloaded input data (SWE)
        filepath_inflow (str)                     -- the path to csv file of inflow data (customer data)
        resample_target (str)                     -- how we want to aggregate target data (to make it smooth and to be able to make predictions, e.g. 3W)
        resample_input (str)                      -- how we want to aggregate input data (to make it smooth and to be able to make predictions, e.g. 3W)
        interpolate_target (str)                  -- how to interpolate the coarser target data (if target is in e.g. 3W frequency and input in 1W, we need to fill the remaining target data)
        order_interpolation (int)                 -- if interpolation in ["polynomial", "spline"], we need to specify order of interpolation in range [1,5]
        n_timesteps (int)                         -- length of window, e.g. 20 weeks
        n_outputs (int)                           -- how many weeks we want to predict (e.g. 15 weeks): take the 15 weeks which follow to 20 input weeks
        training_input_end_date (datetime.date)   -- All dates < specified are used for training
        target_name (list)                        -- name of the columns of the target station (if ["all"] --> all stations are summed up)
        keep_other_inflow_columns (bool)          -- Whether inflow columns (e.g. from inflow-glomma) should be kept to train model
        split_y_test (boop)                       -- Wheter we have a test set or not (if set to false, training_input_end_date will be set to last available date)
        :return:                                  -- scaled training data, x_test and y_test
        -------------
        """

    #############################################################################################################
    # 1. Data Cleansing: Prepare Data, handle NA Values, write out scaled data as csv
    #############################################################################################################
    # Read the data
    input = pd.read_csv(filepath_input, index_col=config["index_col"], encoding='utf-8')
    # We will keep all columns (all reservoir inflow information and use all of them also as input of model)
    target = pd.read_excel(filepath_inflow, index_col=config["index_col"], skiprows=config["skiprows"])

    # First row just contains dummy data (which unit: m^3/s)
    target = target.iloc[1:, :]

    # Convert the index cols into datetime (and normalize, to set the hours, if date contains hour format, set to 0:00)
    target.index = pd.to_datetime(target.index, format="%Y-%m-%d").normalize().tz_localize(tz=None)
    input.index = pd.to_datetime(input.index, format="%Y-%m-%d").normalize().tz_localize(tz=None)

    # If user wants to sum up different lakes
    if len(target_name) > 1:
        target["Inflow Sum Over Region"] = target[target_name].sum(axis=1)
        # Update name for plots
        config["target_name"] = "Inflow Sum Over Region"
        target_name = "Inflow Sum Over Region"

    elif len(target_name) == 1 and target_name[0] == "all":
        target["Inflow Sum Over Region"] = target.sum(axis=1)
        config["target_name"] = "Inflow Sum Over Region"
        target_name = "Inflow Sum Over Region"

    elif len(target_name) == 1:
        target_name = target_name[0]
        config["target_name"] = config["target_name"][0]

    # If set to true just keep the target column
    if not keep_other_inflow_columns:
        target = target[[target_name]]

    # Sort Data by date
    target = target.sort_index()
    input = input.sort_index()

    # Make sure that Columns are all numeric
    target[target.columns] = target[target.columns].apply(pd.to_numeric)
    input[input.columns] = input[input.columns].apply(pd.to_numeric)

    # Resample input and target with given time resolution
    target = target.resample(resample_target).mean()
    input = input.resample(resample_input).mean()

    # Merge input and target
    # NOTE: input and target might have different time resolution, use now again interpolation to fill missing target values
    data = input.join(target)
    data = data.interpolate(method=interpolate_target, order=order_interpolation)
    # After interpolation it might be that the first 2 entries of a column of dataframe target are NA --> Just fill it by the latest actual observation
    for col in target.columns:
        start_na_target_value = target[target.index <= min(data.index)][col].values[-1]
        data[col] = data[col].fillna(value=start_na_target_value)

    # Write csv file of prepared Data
    data.to_csv("../data/norwegian_data_swe_inflow.csv")


    #############################################################################################################
    # 2. Split Data into Training and y, x test values, and scale date
    #############################################################################################################
    # Now scale data (scale per column as we have different features within columns: temperature/ swe/ precipitation)
    for col in data.columns:
        if col != target_name:
            scaler = MinMaxScaler(feature_range=(0, 1))
            data[col] = scaler.fit_transform(data[col].values.reshape(-1, 1))
        # For target column we create a global scaler to descale it later again
        elif col == target_name:
            global inflow_scaler
            inflow_scaler = MinMaxScaler(feature_range=(0, 1))
            data[target_name] = inflow_scaler.fit_transform(data[target_name].values.reshape(-1, 1))

    # transform date object to datetime object (for comparison in df and to create prediction dates)
    training_input_end_date = datetime.datetime(training_input_end_date.year, training_input_end_date.month, training_input_end_date.day)

    if split_y_test:

        # Assign range of values for model to be tested on (and take first "n_output" weeks)
        y_test = data[data.index > training_input_end_date][[target_name]][:n_outputs]
        # Data Check (if for ex. training_input_end_date=2021-12-31 (but dataset ends at 2022-01-31) and n_outputs = 49 --> We dont have 49 data_points for y_test in our data
        if y_test.shape[0] != n_outputs:
            raise Exception(f"Specified Combination of n_output, training_input_end_date and split_y_test does not match. There are not {n_outputs} datapoints after {training_input_end_date}")

        # Assign range of values for model input during testing (and take upcoming "n_timestep" weeks as input)
        test_input_start_date = y_test.index[0] - timedelta(weeks=n_timesteps)
        test_input = data[data.index >= test_input_start_date][:n_timesteps]
        test_input = test_input.values.reshape(1, test_input.shape[0], test_input.shape[1])

        # Training data is exactly until the start date of test input
        training = data[data.index < test_input_start_date]
        y_test[target_name] = inflow_scaler.inverse_transform(y_test.values.reshape(-1, 1))
        return training, test_input, y_test

    else:
        training = data.iloc[:-n_timesteps]
        test_input = data.iloc[-n_timesteps:]

        # Define the dates of our prediction which
        real_end_date_training = test_input.index[-1]
        penultimate_date = test_input.index[-2]
        date_diff = (real_end_date_training - penultimate_date).days
        prediction_dates = [real_end_date_training + timedelta(days = date_diff * i) for i in range(1, n_outputs+1)]
        # We will return empty dataframe with index set to prediction dates (easier for plot function to handle)
        df_prediction_dates = pd.DataFrame(index=prediction_dates)
        # Reshape data
        test_input = test_input.values.reshape(1, test_input.shape[0], test_input.shape[1])
        return training, test_input, df_prediction_dates


def ape(true, test):
    """
    Parameters:
    -------------
    true (float)                        -- target value
    test (float)                        -- predicted value
    :return:                            -- error
    """
    value = (abs(true - test)/true)*100
    return np.array(value)

def get_stats(y_test, y_pred):
    """
    Parameters:
    -------------
    y_pred (np.array)               -- y_pred for one run/ filtered avg of all runs
    y_test (np.array)               -- array with test values
    :return:                        -- no return - function just prints information
    -------------
    """
    print('Mean Absolute Error:', round((mean_absolute_error(y_test, y_pred)/np.mean(y_test))*100,3), '%')
    print('Root Mean Squared Error:', round(((mean_squared_error(y_test, y_pred)**0.5)/np.mean(y_test))*100,3), '%')
    print('Median Absolute Error:', round((median_absolute_error(y_test, y_pred)/np.mean(y_test))*100,3), '%')
    print('Explained Variance:', round(explained_variance_score(y_test, y_pred),3), '\n')
    return

def get_stats_df(y_test, y_pred):
    """
    Parameters:
    -------------
    y_pred (np.array)               -- y_pred for one run/ filtered avg of all runs
    y_test (np.array)               -- array with test values
    :return:                        -- statistics as list
    -------------
    """
    meanae = str(round(abs((mean_absolute_error(y_test, y_pred)/np.mean(y_test))*100),3))
    rmse = str(round(abs(((mean_squared_error(y_test, y_pred)**0.5)/np.mean(y_test))*100),3))
    medae = str(round(abs((median_absolute_error(y_test, y_pred)/np.mean(y_test))*100),3))
    expvar = str(round(explained_variance_score(y_test, y_pred),3))
    stats = [[meanae], [rmse], [medae], [expvar]]
    index = ['Mean Absolute Error (%)', 'Root Mean Squared Error (%)', 'Median Absolute Error (%)', 'Explained Variance']
    return stats, index




def make_plots(history, y_pred, y_test):
    """
    The function creates 3 Plots:
    - Model Learning Curve
    - Prediction vs Target plot (including boxplots and 95% confidence interval)
    - Erros plot
    Parameters:
    -------------
    history (list)                  -- list containing all keras.callbacks.History values of all runs
    y_pred (list)                   -- list containing all y_pred of all runs
    y_test (pd.Series)              -- pandas Series of test values
    :return:                        -- Creating 3 plots, no return
    -------------
    """

    # Define number of repeated runs of NN
    repeats = len(history)

    ##################################################################################################################
    # Model Learning Curve
    ##################################################################################################################
    loss = [pd.DataFrame(history[i].history['loss']) for i in range(len(history))]
    val_loss = [pd.DataFrame(history[i].history['val_loss']) for i in range(len(history))]
    max_epochs = max([len(loss[i]) for i in range(len(loss))])
    loss = [loss[i].values for i in range(len(loss))]
    val_loss = [val_loss[i].values for i in range(len(val_loss))]
    for i in range(len(loss)):
        if len(loss[i]) == max_epochs:
            loss[i] = loss[i]
        else:
            while len(loss[i]) < max_epochs:
                loss[i] = np.append(loss[i], np.nan)
                val_loss[i] = np.append(val_loss[i], np.nan)
        loss[i] = pd.DataFrame(loss[i])
        val_loss[i] = pd.DataFrame(val_loss[i])
    loss_df = reduce(lambda x, y: pd.merge(x, y, left_index=True, right_index=True), loss)
    val_loss_df = reduce(lambda x, y: pd.merge(x, y, left_index=True, right_index=True), val_loss)

    loss_avg = loss_df.mean(axis=1)
    loss_avg.name = 'loss_avg'
    val_loss_std = val_loss_df.std(axis=1)
    val_loss_std.name = 'val_loss_std'
    val_loss_avg = val_loss_df.mean(axis=1)
    val_loss_avg.name = 'val_loss_avg'
    upper_val = pd.Series(val_loss_avg.values + 1.96 * val_loss_std.values / (repeats ** 0.5))
    upper_val.name = 'upper_val'
    lower_val = pd.Series(val_loss_avg.values - 1.96 * val_loss_std.values / (repeats ** 0.5))
    lower_val.name = 'lower_val'
    history = reduce(lambda x, y: pd.merge(x, y, left_index=True, right_index=True), [loss_avg, val_loss_avg, upper_val, lower_val])
    history.columns = ['loss_avg', 'val_loss_avg', 'upper_val', 'lower_val']


    plt.figure(figsize=(20,10))
    plt.plot(history.index, history.loss_avg.values, color='Blue', linewidth=2, linestyle='solid', label="Training Loss (80%)")
    plt.plot(history.index, history.val_loss_avg.values, color='Orange', linewidth=2, linestyle='solid', label="Validation Loss (20%)")
    plt.fill_between(history.index, history.upper_val, history.lower_val, color='Orange', alpha=.15, label='95% Confidence Interval')
    plt.title(str(y_test.index[0].year)+" Model Learning Curve", fontsize=18)
    plt.ylabel("Loss: Mean Squared Error", fontsize=14)
    plt.ylim([0, 0.02])
    plt.xlabel("Experiences", fontsize=14)
    plt.legend(fontsize=14, loc = 'upper left')
    plt.savefig(fname=config["path_lc"], bbox_inches='tight')


    ##################################################################################################################
    # Prediction vs Target plot (including boxplots and 95% confidence interval)
    ##################################################################################################################
    future_idx = range(1, len(y_test) + 1)
    plt.figure(figsize=(20, 10))
    x_labels = y_test.index
    x_labels = x_labels.strftime('%Y-%m-%d')
    df = pd.DataFrame(np.concatenate(y_pred, axis=1), index=x_labels)
    #[get_stats(y_test.values, y_pred[i]) for i in range(len(y_pred))]


    if repeats > 1:
        q1, q3 = df.quantile(q=0.25, axis=1), df.quantile(q=0.75, axis=1)
        iqr = q3 - q1
        upper_bnd = q3 + 1.5*iqr
        lower_bnd = q1 - 1.5*iqr

        filtered_avg = np.stack([df.iloc[i][(df.iloc[i] < upper_bnd[i]) & (df.iloc[i] > lower_bnd[i])].mean() for i in range(len(df))])
        filtered_std = np.stack([df.iloc[i][(df.iloc[i] < upper_bnd[i]) & (df.iloc[i] > lower_bnd[i])].std() for i in range(len(df))])
        upper = filtered_avg + 1.96*filtered_std/(repeats ** 0.5)
        lower = filtered_avg - 1.96*filtered_std/(repeats ** 0.5)
        plt.boxplot(df.transpose())

        plt.fill_between(future_idx, upper, lower, color='k', alpha=.15, label='95% Confidence Interval')

    else:
        filtered_avg = df.values

    if config["split_y_test"]:
        cellText, rows = get_stats_df(y_test.values, filtered_avg.reshape(-1, ))
        plt.plot(future_idx, y_test.values, color='Blue', linewidth=2, linestyle='solid', label="Actual")
        plt.table(cellText=cellText, rowLabels=rows, colLabels=['Stats'], bbox=[0.2,0.4,0.1,0.5])

    plt.plot(future_idx, filtered_avg, color='Orange', linewidth=2, linestyle='solid', label="Forecasted")
    plt.title(f"Forecast for {config['target_name']}", fontsize=18)
    plt.ylabel("Inflow in m^3/s", fontsize=14)
    plt.xlabel("Time", fontsize=14)
    plt.xticks([i for i in future_idx], x_labels, rotation=90)
    plt.legend(fontsize=14, loc='upper left')
    plt.savefig(fname=config["path_fc"], bbox_inches='tight')

    ##################################################################################################################
    # Erros plot
    ##################################################################################################################
    # Just if we have a y_test set
    if config["split_y_test"]:
        errors = np.stack(np.array([ape(y_test.values[i], df.mean(axis=1).values[i]) for i in range(len(df))]))
        upper = np.stack(np.array([ape(y_test.values[i], df.mean(axis=1).values[i] + (df.std(axis=1).values[i]*1.96)/(repeats**0.5)) for i in range(len(df))]))
        lower = np.stack(np.array([ape(y_test.values[i], df.mean(axis=1).values[i] - (df.std(axis=1).values[i]*1.96)/(repeats**0.5)) for i in range(len(df))]))
        errors_df = pd.DataFrame(lower, columns=['Lower Limit'])
        errors_df['Average'] = errors
        errors_df['Upper Limit'] = upper
        errors_df.index = df.index
        errors_df.plot(kind='bar', figsize=[20,10], title='Absolute Percent Error (%)')
    return


def main(config_filename):
    """
    Parameters taken from config:
    -------------
    n_timesteps (int)               -- length of window, e.g. 20 weeks
    n_outputs (int)                 -- how many weeks we want to predict (e.g. 15 weeks): take the 15 weeks which follow to 20 input weeks
    epochs (int)                    -- how many epochs
    batch_size (int)                -- batch size for model
    year (int)                      -- which year we want to investigate
    months (Sequence[int])          -- which months we want to use (e.g. [3, 4] for March and April)
    filepath (str)                  -- filepath for the training data csv file
    target_name (str)               -- name of the column of the target station
    monitor (str)                   -- validation loss / or loss: Stop training when the monitored metric has stopped improving
    patience (int)                  -- Number of epochs with no improvement after which training will be stopped.
    loss (str)                      -- loss function
    optimizer (str)                 -- optimizer for model
    n_nodes1 (int)                  -- the dimensionality of the output space (i.e. the number of output filters in the convolution) for the first layers block
    n_nodes2(int)                   -- the dimensionality of the output space (i.e. the number of output filters in the convolution) for the second layers block
    filter1 (int)                   -- the dimensionality of the output space (i.e. the number of output filters in the convolution) for the third layers block
    filter2 (int)                   -- the dimensionality of the output space of first Dense Layer
    kernel_size (int)               -- specifies the length of the 1D convolution window
    weight_regularizer (int)        -- every coefficient in the weight matrix of the layer will add weight_regularizer * weight_coefficient_value**2 to the total loss of the network
    :return:                        -- y_pred (np.array), history (keras.callbacks.History)
    -------------
    """

    # load config file
    global config
    with open(config_filename, encoding='utf8') as conf:
        config = yaml.load(conf, Loader=yaml.FullLoader)

    # Option to turn off GPU
    if config["run_on_gpu"] is False:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    num_gpus = len(tf.config.list_physical_devices('GPU'))
    print("Num GPUs Available: ", num_gpus)
    if num_gpus == 0:
        print(print(f"\033[93mWarning: No GPU found!\033[0m"))

    training, X_test, y_test = prepare_data_norwegian(filepath_input=config["filepath_input"], filepath_inflow=config["filepath_inflow"],
                                                      resample_target=config["resample_method_inflow"], resample_input=config["resample_method_input"],
                                                      interpolate_target=config["interpolate_target"], order_interpolation=config["order_interpolation"],
                                                      n_timesteps=config["n_timesteps"], n_outputs=config["n_outputs"], training_input_end_date=config["training_input_end_date"],
                                                      target_name=config["target_name"], keep_other_inflow_columns=config["keep_other_inflow_columns"], split_y_test=config["split_y_test"])


    # Transform data into window inputs/ outputs
    X_train = np.stack([sliding_window_input(training, config["n_timesteps"], i) for i in range(len(training) - config["n_timesteps"])])[
              :-config["n_outputs"]]
    y_train = np.stack([sliding_window_output(training[config["target_name"]], config["n_timesteps"], config["n_outputs"], i) for i in
                        range(len(training) - (config["n_timesteps"] + config["n_outputs"]))])


    # Create List to store y_pred and history of different runs
    y_pred = []
    history = []
    ## Create df to store predicted values of each run
    predictions = pd.DataFrame(index=y_test.index)

    for i in range(config["repeats"]):
        run = i+1
        print(f"Run {run} out of {config['repeats']} runs")
        model = ResCNN_LSTM_Model(n_nodes1=config["n_nodes1"], n_nodes2=config["n_nodes2"], filter1=config["filter1"],
                                  filter2=config["filter2"], kernel_size=config["kernel_size"], n_outputs=config["n_outputs"],
                                  weight_regularizer=config["weight_regularizer"], use_cnn_layers=config["rescnn"], use_lstm_layers=config["lstm"], use_batch_normalization=config["use_batch_normalization"])

        # Make use of Tensorboard, to see losses of tensorboard go to terminal, cd into repo and run "tensorboard --logdir=logs/" (in particular: a logs folder is created and train and validation data is stored here after running NN)
        tensorboard = TensorBoard(log_dir=f"logs/{time.time()}")

        # Compile Model with Mean Squared Error loss function and adam optimizer
        if config["optimizer"] == "adam":
            optimizer = tf.keras.optimizers.Adam(learning_rate=config["learning_rate"])
        else:
            print("Unsupported optimizer! Only Adam is supported now!")

        model.compile(loss=config["loss"], optimizer=optimizer)
        # fit network
        es = EarlyStopping(monitor=config["monitor"], mode='auto', patience=config["patience"])
        history_run = model.fit(X_train, y_train, epochs=config["epochs"], batch_size=config["batch_size"], verbose=2, validation_split=0.2, callbacks=[es, tensorboard])
        history.append(history_run)

        # test model against hold-out set
        y_pred_run = model.predict(X_test)

        # Descale/ Scale back y_pred
        y_pred_run = inflow_scaler.inverse_transform(y_pred_run[0, :].reshape(-1, 1))

        # Append current run to y_pred list
        y_pred.append(y_pred_run)
        # Add column to df
        predictions[f"run_{run}_predictions"] = y_pred_run

    # Calculate the average over all runs/ all columns
    predictions["prediction_avg"] = predictions.mean(axis=1)
    # Add y_test values, if available
    if config["split_y_test"]:
        predictions["y_test"] = y_test

    # Store predictions in specified path
    predictions.to_csv(config["path_predictions"])

    # Create Plots if set to True
    if config["run_plot"]:
        make_plots(history, y_pred, y_test)

    return history, y_pred, y_test


if __name__ == "__main__":
    history, y_pred, y_test = main("config.yaml")