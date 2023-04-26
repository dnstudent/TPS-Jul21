import seaborn as sns
import pandas as pd


def plot_predictions(data, estimator, **kwargs):
    """Given a climate dataset and a pollution model, plots the predictions of the model on the dataset.
    """
    y_pred = estimator.forecast(data)
    if isinstance(data, (list, tuple)):
        y_true = pd.concat(data)[estimator.targets]
    else:
        y_true = data[estimator.targets]
    sns_df = pd.concat([y_true, y_pred], axis=1,
                       keys=["Osservazioni", "Predizioni"]).melt(var_name=["Sorgente dati", "Inquinante"], ignore_index=False)
    return sns.relplot(data=sns_df.reset_index(), x="date_time", y="value",
                       hue="Sorgente dati", style="Sorgente dati", row="Inquinante", kind="line", errorbar=None, **kwargs)
