import seaborn as sns
import pandas as pd


def plot_predictions(data, model, **kwargs):
    """Given a climate dataset and a pollution model, plots the predictions of the model on the dataset.
    """
    y_pred = model.forecast(data)
    y_true = data[model.targets]
    sns_df = pd.concat([y_true, y_pred], axis=1,
                       keys=["observed", "predicted"]).melt(var_name=["series", "target"], ignore_index=False)
    return sns.relplot(data=sns_df.reset_index(), x="date_time", y="value",
                       hue="series", row="target", kind="line", errorbar=None, **kwargs)
