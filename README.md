# Winterload

see `docs/proposal.pdf`

# Infrastructure
First, copy `/.env.example` to `/.env` and `/.env.docker.example` to `/.env.docker`
Then adjust the values in it according to the documentation inside.

## Local
The infrastructure is managed using [docker compose](https://docs.docker.com/desktop/). Spin it up using [make](https://www.gnu.org/software/make/):

```shell
make setup
```

after the first setup, you can start using the docker containers from now on with this comman. 

```shell
make up
```

_or alternatively copy the command from the @docker line below_ `up:` _in the file `./Makefile`_

Running all containers might take a while during first setup.

Stopping all containers without removing the data can be done using

```shell
make down
# or
docker compose down
```

If you want to clear everything, you can run

```shell
make stop
```

### Object Storage
This project uses [SeaweedFS](https://s3fs.readthedocs.io/en/latest/api.html) for object storage.

The following backends are made available:

- [S3 API](http://localhost:8333/)
- [Filer UI](http://localhost:8889/)
- [Master UI](http://localhost:9333)
- [Admin UI](http://localhost:23646)

Use it in your script like:

```python
from s3fs import S3FileSystem
# Autoconfigured using FSSPEC_S3_ENDPOINT_URL, FSSPEC_S3_KEY, FSSPEC_S3_SECRET
fs = S3FileSystem()
```

## Cloud

### BigQuery
This project uses [BigQuery](https://cloud.google.com/solutions/data-analytics-and-ai) as cloud storage. To set it up 
properly, you need to install the [gcloud CLI](https://cloud.google.com/sdk/docs/install). The Google project needs to have billing and APIs enabled, and must 
be able to create service accounts and provision roles so the terraform setup can be run properly.

To setup the gcloud CLI for this project, please use the commands below.

```shell
gcloud auth application-default login
gcloud auth application-default set-quota-project <your-project-id>
```

After setting up the basics, the service account needs to be provided by the following command. The output of the 
service account will be written to `/secrets` (ignored folder) which then will be automatically mounted by docker compose 
into the `/app/secrets` folder (configured using `GOOGLE_APPLICATION_CREDENTIALS=/app/secrets/gcp-service-account.json` 
in `.env.docker`).

```shell
gcloud iam service-accounts keys create secrets/gcp-service-account.json \
  --iam-account="$(terraform -chdir=infra output -raw service_account_email)"
```

### Terraform

[Terraform](https://developer.hashicorp.com/terraform/install) is used to automatically setup the cloud infrastructure 
for this project. To run it, the Google account has to be allowed to enable APIs, create service accounts and grant roles.

Terraform automatically loads configuration from the .env file.
`GCP_PROJECT_ID`, `BIGQUERY_LOCATION` and `BIGQUERY_DATASET` need to be set accordingly.

The three commands setting up the cloud infrastructure are:

```shell
make tf-init
make tf-plan
make tf-apply
```

The setup script will also automatically setup the cloud infrastructure using `tf-init` and `tf-apply`. 

## Feature Storage

This project uses [Feast](https://feast.dev/) as feature store. It provides online and offline capabilities. For now, 
offline capabilities suffice.

The feature store definition can be found in `src/winterload/feature_repo` which will be mounted to 
`/app/src/winterload/feature_repo` using docker (configured using `FEAST_FS_YAML_FILE_PATH` in .env.docker).

### Offline

The features can be stored either in a local S3 object storage using SeaweedFS or a remove Google Cloud BigQuery database.
The target of the data pipeline can be configured using the `OFFLINE_STORAGE` .env variable and it can be set to either 
`s3` or `bigquery`.

To use BigQuery as a database, you also have Google Cloud setup (see infrastructure) and you have to configure 
`GCP_PROJECT_ID`, `BIGQUERY_LOCATION` and `BIGQUERY_DATASET` accordingly.

## Experiment tracking and model registry

This project uses [MLflow](https://mlflow.org/) for experiment tracking and as model registry. The docker compose
service `mlflow` runs the server, and its UI is at [localhost:5001](http://localhost:5001). Start it with:

```shell
docker compose --env-file .env --env-file .env.docker up -d --wait mlflow
```

The training pipeline needs this server, so start it before `uv run winterload`. `MLFLOW_TRACKING_URI` selects the
server (default: `http://localhost:5001`). Runs, models and aliases live in the Docker volume `mlflow_data`, and
`make stop` deletes that volume.

The registered model `winterload-load-forecast` has two aliases. `champion` is the winner refit on all data, for
inference. `champion_eval` stops at the cutoff of the run that promoted it, so later runs can compare against it on
hours it has not seen.

# Pipeline

Run the pipeline using

```shell
uv run winterload
```

`-p` picks one stage. The monitoring stage stays out of the default run, because it runs on its own daily schedule:

```shell
uv run winterload -p monitoring
```

It has two steps. `retrain_monitor` scores the model and the seasonal-naive baseline on the last 7 days and runs
champion/challenger when the model's MAPE exceeds 0.9 x the baseline's. `publication_probe` appends how late the load and
weather sources publish to the result store.

# Data access and results

Only `src/winterload/repository/` reads stored data, and `tests/test_architecture.py` fails if anything else does. Notebooks read
the feature store through repositories and never call an external API.

`weather_measurements` holds two models in its `model` column: `ecmwf_ifs` from the Historical Forecast API and `era5`, the
reanalysis that scores forecasts. Feast only sees `ecmwf_ifs`. ERA5 exists for the days the notebooks score. Add days with a
forced backfill, because the backfill otherwise skips days it already holds:

```shell
uv run winterload -p features.weather.backfill -s 2025-01-31 -e 2025-02-14 -f
```

Notebooks and monitoring keep generated results as CSV objects on SeaweedFS under `RESULTS_PATH` (default
`s3://winterload/results`), and never in git. Start the store with `make up` before `make lab` or `-p monitoring`, and browse it in
the [Filer UI](http://localhost:8889/). The notebooks keep their executed outputs in git, so readers without the store can see the
numbers. `make stop` deletes the SeaweedFS volume with these results, and the publication delays can't be measured again.

# Forecast inputs

## Weather forecasts

Each target day D uses one ECMWF IFS run, chosen so that it was published before the forecast is issued at 08:00 Zurich time
on D-1. Open-Meteo published the 00 UTC run at 06:51 UTC on 2026-10-31, which misses 08:00 in summer (06:00 UTC), so the store
holds the 18 UTC run of D-2. The rule lives in `weather_run` in `src/winterload/forecast/features/dataset.py`, and
`notebooks/benchmark/weather_run_choice.ipynb` scores both runs against ERA5: the safer run costs 0.14 C of temperature error in
February and gains 0.06 C in July.

The Single Runs archive sets the limits of the history:

- It starts on 2024-03-14, so weather starts on 2024-03-16, and data from before that can't train the model.
- It adds the 18 UTC cycle on 2024-08-06. Earlier days use the 12 UTC run of D-2.
- It has no 18 UTC run for the target days 2025-08-06 and 2025-08-09, and single variables are empty on 2025-08-07,
  2025-08-08, 2025-08-10, 2025-08-11 and 2026-06-24. Hours in those gaps drop out of the training frame.

The four cities combine with an unweighted mean. Population weights scored worse on six chronological folds
(`notebooks/training/feature_study.ipynb`).

## Holidays

The `holidays` package lists public holidays per canton, and only New Year, Ascension, 1 August and Christmas hold in all 26.
The model trains on `holiday_share_ch`, the share of Swiss residents whose canton has a public holiday on the local date.
`is_holiday_ch` marks the dates where that share reaches 50%: those four days plus Good Friday, Easter Monday, Whit Monday and
26 December. See `src/winterload/forecast/features/holidays_ch.py`, and `notebooks/training/feature_study.ipynb` for the errors on
holiday hours.

## Model settings

`PreliminaryModel` keeps LightGBM's defaults, written out in `src/winterload/forecast/model/preliminary.py`, because the tuning
target measures against it:

| Setting | Value | Controls |
|---|---|---|
| `n_estimators` | 100 | Number of trees |
| `learning_rate` | 0.1 | Weight of each tree |
| `num_leaves` | 31 | Tree complexity |
| `max_depth` | -1 (no limit) | Tree complexity |
| `min_child_samples` | 20 | Tree complexity: rows per leaf |
| Early stopping | Off | Scored worse on the folds (`notebooks/training/feature_study.ipynb`) |

`notebooks/training/feature_study.ipynb` compares feature groups, weather weights, settings and seasonal models on six 8-week
chronological folds before the test year, and logs each comparison to MLflow.

| Feature set | Mean MAPE | Mean MAE (MW) |
|---|---|---|
| Load + calendar + weather | 6.75% | 466 |
| Load + calendar | 7.73% | 541 |
| Load | 7.78% | 535 |
| Load + weather | 7.93% | 551 |
| Seasonal-naive baseline | 8.25% | 559 |
| Calendar + weather | 9.14% | 648 |
| Weather | 10.26% | 725 |
| Calendar | 10.30% | 737 |

Shuffling the order of the cities within an hour gives 6.76% to 6.79% for the first row, so a smaller gap says nothing.
`WeatherRepository` sorts the cities, which keeps results reproducible.

# Originality check

Lists checked on 2026-10-31. The table names the closest projects, and none of them forecasts a country's hourly grid load.

| List | Closest projects | What they predict |
|---|---|---|
| [mlops-lab.ch FS26](https://mlops-lab.ch/semesters/fs26.html), 15 projects | gizstrom | Daily PV output of one private rooftop installation |
| [mlops-lab.ch HS26](https://mlops-lab.ch/semesters/hs26.html), 19 projects in progress | Negative Hours; Swiss Control Area Balance Forecast | German-Luxembourg hours with negative prices; Swiss system imbalance at 15-minute resolution |
| [KTH ID2223 2022/23](https://id2223kth.github.io/assignments/project/ServerlessMLProjectsID22232023.html) | Predict Electricity Demand in NYC; Electricity Price Prediction for Sweden | [Daily demand in New York](https://github.com/aykhazanchi/id2223-scalable-ml/tree/master/proj), trained on observed NOAA weather; SE3 prices |
| [KTH ID2223 2024](https://id2223kth.github.io/assignments/2024/ID2223Projects2024.html) | None on energy | |
| [KTH ID2223 2025](https://id2223kth.github.io/assignments/2025/ID2223Projects2025.html) | Cross Border Energy Flow Predictor; three on Swedish prices | Flows and prices |
| [KTH ID2223 2026](https://id2223kth.github.io/assignments/2026/ID2223Projects2026.html) | Green Energy Prediction; SE3 Electricity Price Prediction | [Hourly wind and solar generation in Sweden](https://github.com/cimun/id2223_project), whose README does not say which weather trains it; SE3 prices |

Two things set this project apart:

- **Winter drift.** It runs from October to January, into the heating season, so monitoring and retraining meet real drift.
  The retrain trigger compares against the baseline, so a seasonal rise in error that also hits the baseline does not fire it.
- **Archived forecasts.** It trains on forecast runs that were published before each 08:00 issue time, the input it sees in
  production.

# LICENSE
proprietary (for now)