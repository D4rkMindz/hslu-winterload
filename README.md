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


# LICENSE
proprietary (for now)