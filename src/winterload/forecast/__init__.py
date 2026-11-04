"""The domain layer: forecasting logic without I/O. Each subpackage imports only the ones listed
below it:

- lifecycle: decisions about a model over time (champion, promotion, retraining, the test-year guard).
- evaluation: offline measurement (splits, the tuning target, backtests, season reports, feature studies).
- model: the forecasters (the LightGBM model and the seasonal-naive baseline).
- features: what goes into a model (the training frame, Swiss holidays, population weights).

metrics.py sits at the root because every subpackage scores forecasts with it.
"""
