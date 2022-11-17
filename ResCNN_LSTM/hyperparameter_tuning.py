import csv
import optuna
import ResCNN_LSTM_fit
import yaml
from statistics import mean


# Define an objective function to be minimized
def objective(trial):
    config = {}
    config["n_timesteps"] = trial.suggest_int('n_timesteps', 8, 102)
    config["batch_size"] = trial.suggest_categorical('batch_size', [64, 128])
    config["epochs"] = trial.suggest_int("epochs", 25, 100)
    config["n_nodes1"] = trial.suggest_int("n_nodes1", 4, 128)
    config["n_nodes2"] = trial.suggest_int("n_nodes2", 4, 64)
    config["filter1"] = trial.suggest_int("filter1", 2, 128)
    config["filter2"] = trial.suggest_int("filter2", 2, 64)
    config["kernel_size"] = trial.suggest_int("kernel_size", 3, 9)
    config["learning_rate"] = trial.suggest_loguniform('learning_rate', 1e-5, 1e-3)
    config["dropout_probability"] = trial.suggest_float('dropout_probability', 0, 0.5)
    config["use_batch_normalization"] = trial.suggest_categorical('use_batch_normalization', [True, False])
    config["patience"] = trial.suggest_int("patience", 5, 50)
    config["weight_regularizer"] = trial.suggest_loguniform("weight_regularizer", 1e-9, 1e-5)
    # config["rescnn"] = trial.suggest_categorical('rescnn', [[False, False, True], [False, True, True], [True, True, True]])
    # config["lstm"] = trial.suggest_categorical('lstm', [[False, False, False, True], [False, False, True, True], [False, True, True, True], [True, True, True, True]])

    # resample_method_inflow = trial.suggest_categorical("resample_method_inflow", ["1W", "2W", "3W])
    # config["resample_method_inflow"] = resample_method_inflow
    # config["resample_method_input"] = resample_method_inflow

    # loss = trial.suggest_categorical("loss", ["mae", "mse"])
    # config["loss"] = loss
    config["run_plot"] = False
    # config["path_fc"] = "../plots/forecast_plot_" + "%03d" % trial.number + ".pdf"
    config_filename = "hypertune_configs/config_optuna_" + "%03d" % trial.number + ".yaml"
    create_yaml(
        config_filename,
        config
    )
    history, y_pred, y_test = ResCNN_LSTM_fit.main(config_filename)
    # TODO If we want to compare mae and mse loss function, we need a metric based on the validation set (mae for both? mse for both? something else?)
    mean_val_loss = mean([history[i].history["val_loss"][-1] for i in range(len(history))])

    with open("hypertune_configs/configs_evaluation.txt", "a") as f:
        f.write(f"\nTrial {trial.number}: validation loss = {mean_val_loss}")
        f.close()

    row = [trial.number, config['n_timesteps'], config['batch_size'], config['epochs'], config['n_nodes1'], config['n_nodes2'],
              config['filter1'], config['filter2'], config['kernel_size'], config['learning_rate'], config['dropout_probability'],
              config['use_batch_normalization'], config['patience'], config['weight_regularizer'], mean_val_loss]

    with open("hypertune_configs/configs_evaluation.csv", "a", encoding='UTF8', newline='') as csv_f:
        csv_writer = csv.writer(csv_f)
        csv_writer.writerow(row)
        csv_f.close()

    return mean_val_loss


def create_yaml(
        filename,
        config
):
    with open("config.yaml", encoding='utf8') as f:
        config_template = yaml.safe_load(f)

    for key in config:
        config_template[key] = config[key]

    with open(filename, "w") as f:
        yaml.dump(config_template, f)


def main():
    # Create a study object and minimize the objective function.
    with open("hypertune_configs/configs_evaluation.txt", "w") as f:
        f.write("######### Hyperparameter Config Validations #############")
        f.close()
    header = ['trial', 'n_timesteps', 'batch_size', 'epochs', 'n_nodes1', 'n_nodes2',
              'filter1', 'filter2', 'kernel_size', 'learning_rate', 'dropout_probability',
              'use_batch_normalization', 'patience', 'weight_regularizer', 'val_loss']

    with open("hypertune_configs/configs_evaluation.csv", "w", encoding='UTF8', newline='') as csv_f:
        csv_writer = csv.writer(csv_f)
        csv_writer.writerow(header)
        csv_f.close()

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=20)


if __name__ == "__main__":
    main()
