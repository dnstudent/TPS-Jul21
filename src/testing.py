# %%
from keras.regularizers import l2
from lib.data.utils import train_data, targets, features, split_dataframe
# from lib.data.plot import plot_predictions
from lib.models.estimators import PollutionEstimator
from lib.models.on_the_fly import OnTheFlyLSTM
# from lib.models import DirectRnn
# from statsmodels.tsa.deterministic import DeterministicProcess, CalendarFourier


import tensorflow as tf
import keras
from keras.callbacks import TensorBoard as TB, EarlyStopping as ES  # , ReduceLROnPlateau as RLRP
# from keras.optimizers import Adam, SGD
import keras_tuner as kt
import seaborn as sns
sns.set_theme("notebook", style="whitegrid", rc={"figure.dpi": 100})

data_dir = "../data/"

df = train_data(data_dir, delta=False)
# in_day_features = CalendarFourier("D", 8)
# time_features = DeterministicProcess(
#     df.to_period("H").index,
#     constant=False,
#     additional_terms=[in_day_features],
#     drop=True
# ).in_sample().to_timestamp()

# df = df.join(time_features, how="left")
all_features = features  # + time_features.columns.to_list()

train_df, valid_df = split_dataframe(df, 0.2, 0.0)
(X_train, X_valid), (y_train, y_valid) = ([td[all_features] for td in train_df], valid_df[all_features]), ([
    td[targets] for td in train_df], valid_df[targets])

regularization = l2(0.01)
rnn_kwargs = {"dropout": 0.1, "recurrent_dropout": 0.1,
              "kernel_regularizer": regularization, "recurrent_regularizer": regularization, "bias_regularizer": regularization, "activity_regularizer": regularization}


def model_builder(hp):
    double_rnn = hp.Boolean("double_rnn")
    input_days = hp.Int("input_days", min_value=1, max_value=7, step=1)
    n_recursive_units = hp.Int("n_units", min_value=32, max_value=1024, step=2, sampling="log")
    if double_rnn:
        second_layer = [3]
    else:
        second_layer = []
    model = OnTheFlyLSTM(len(all_features), len(targets), n_recursive_units=(
        [n_recursive_units] + second_layer), rnn_kwargs=rnn_kwargs, input_days=input_days)
    estimator = PollutionEstimator(all_features, targets, model, deltas=False)
    lr = hp.Float("learning_rate", min_value=1e-4, max_value=5e-2, sampling="log")
    la = hp.Choice("learning_algorithm", values=["adam", "sgd", "rmsprop"])
    estimator.adapt_training(train_df)
    estimator.compile(optimizer=keras.optimizers.deserialize(
        {"class_name": la, "config": {"learning_rate": lr, "clipnorm": 1_000.0}}), loss="mse")
    return estimator


# %%
tuner = kt.Hyperband(model_builder, objective="val_loss", max_epochs=500,
                     directory="tuning_test", project_name="on_the_fly_lstm_test", executions_per_trial=3, distribution_strategy=tf.distribute.MirroredStrategy(["/GPU:0", "/CPU:0"]))
tuner.search(X=X_train, y=y_train, validation_data=valid_df, shift_hours=1, batch_size=256, callbacks=[ES(patience=100)], verbose=0)


# %%
# with tf.device("/CPU:0"):
#     n_units = [64]
#     itd = [2]
#     models = [OnTheFlyLSTM(len(all_features), len(targets), n_recursive_units=[nu, 3], rnn_kwargs=rnn_kwargs, input_days=d, name=f"n{nu}/id{d}") for nu in n_units for d in itd]
#     estimators = [PollutionEstimator(all_features, targets, model, deltas=False, name=model.name) for model in models]
#     for e in estimators:
#         name = f"otf_dblrnn_regul/adam/{e.name}"
#         e.adapt_training(train_df)
#         e.compile(optimizer="adam", loss="mse")
#         e.train(train_df, epochs=2000, validation_data=valid_df, shift_hours=1, batch_size=32, callbacks=[
#             TB(f"../logs/{name}"), ES(patience=100, restore_best_weights=False)])


# %%
# fg = plot_predictions(valid_df, estimators[0], aspect=3, facet_kws={
#                       "sharey": False})  # , "xlim": (pd.Timestamp("2010-12-15"), pd.Timestamp("2010-12-25"))})
