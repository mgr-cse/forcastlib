import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow_probability import sts

class SeasonTrend:
    def __init__(self, seasonval, nu_prior=None, mu_prior=None, guide=None) -> None:
        self.seasonval = seasonval
        self.nu_prior = nu_prior
        self.mu_prior = mu_prior
        self.guide = guide

    def fit(self, X_train, Y_train, elbo_itr=200):
        observed_time_series = Y_train
        global_trend = sts.LocalLinearTrend(observed_time_series=observed_time_series)
        seasonal_trend = tfp.sts.Seasonal(num_seasons=self.seasonval, observed_time_series=observed_time_series)
        self.model = sts.Sum([global_trend, seasonal_trend], observed_time_series=observed_time_series)

        if not self.guide:
            self.guide = tfp.sts.build_factored_surrogate_posterior(model=self.model)

        # train model
        elbo = tfp.vi.fit_surrogate_posterior(
            target_log_prob_fn=self.model.joint_distribution(
            observed_time_series=observed_time_series).log_prob,
            surrogate_posterior=self.guide,
            optimizer=tf.optimizers.Adam(learning_rate=0.1),
            num_steps=elbo_itr,
            jit_compile=True)

        self.q_sample = self.guide.sample(50)
        self.train_data = Y_train

    def predict(self, X_test):
        num_forecast_steps = len(X_test)
        forcast_dist = tfp.sts.forecast(
            self.model,
            observed_time_series=self.train_data,
            parameter_samples=self.q_sample,
            num_steps_forecast=num_forecast_steps)

        mean = forcast_dist.mean().numpy()[..., 0]
        std = forcast_dist.stddev().numpy()[..., 0]

        return (mean, std)

