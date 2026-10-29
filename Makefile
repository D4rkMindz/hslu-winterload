########################################################################################################################
# Docker
########################################################################################################################
COMPOSE := docker compose --env-file .env --env-file .env.docker

up:
	@$(COMPOSE) up -d

down:
	@$(COMPOSE) down

stop:
	@$(COMPOSE) down -v --rmi all --remove-orphans


########################################################################################################################
# Feast feature store
########################################################################################################################
# --env-file: Feast picks its config from FEAST_FS_YAML_FILE_PATH (default: feature_store.yaml) and
# expands ${GCP_PROJECT_ID} etc. in it, all from .env.
FEAST := uv run --env-file .env feast --chdir src/winterload/feature_repo/

apply:
	@$(FEAST) apply

########################################################################################################################
# Terraform
########################################################################################################################
# Reads a value from .env, so Terraform targets the same project and dataset as the pipelines.
env = $(shell sed -n 's/^$(1)=//p' .env 2>/dev/null)

OFFLINE_STORAGE := $(or $(call env,OFFLINE_STORAGE),s3)
export TF_VAR_project_id := $(call env,GCP_PROJECT_ID)
export TF_VAR_region := $(call env,BIGQUERY_LOCATION)
export TF_VAR_dataset_id := $(call env,BIGQUERY_DATASET)
TF_STATE_BUCKET := $(TF_VAR_project_id)-tfstate
TERRAFORM := terraform -chdir=infra

.PHONY: tf-init tf-plan tf-apply
# Terraform can't create the bucket it keeps its state in, so this does it once (infra/state.tf then
# imports and tracks it). Idempotent.
tf-init:
	@gcloud storage buckets describe gs://$(TF_STATE_BUCKET) --project=$(TF_VAR_project_id) >/dev/null 2>&1 || ( \
		gcloud storage buckets create gs://$(TF_STATE_BUCKET) --project=$(TF_VAR_project_id) \
			--location=$(TF_VAR_region) --uniform-bucket-level-access --public-access-prevention && \
		gcloud storage buckets update gs://$(TF_STATE_BUCKET) --versioning )
	@$(TERRAFORM) init -backend-config="bucket=$(TF_STATE_BUCKET)"

tf-plan:
	@$(TERRAFORM) plan

# Shows the plan once more and asks for confirmation before changing anything.
tf-apply:
	@$(TERRAFORM) apply

########################################################################################################################
# Jupyter Lab
########################################################################################################################
JUPYTER_PORT := 8989
# password for local: XEfyLLhBxrK6LGGUMnDboT2*MFKbbfJe8_QqkgiQaNouuUtxPDt@m46j7@uFQHnG8-GP@ohsvZNp26qHrKxGh3MA6jbdgka9o
# for hashing your own: run
#  $ uv run python -c "from jupyter_server.auth import passwd; print(passwd('<your password>'))"
#  > argon2:$argon2id$v=19$m=10240,t=10,p=8$yqO/ISF45iJk38QaWUVOuQ$ZIxTrQ0Ct/RWe7TyEybtE9hAa4wsEnNdkbMmyEer16s
# copy the hash and replace $ with $$
JUPYTER_HASH := argon2:$$argon2id$$v=19$$m=10240,t=10,p=8$$yqO/ISF45iJk38QaWUVOuQ$$ZIxTrQ0Ct/RWe7TyEybtE9hAa4wsEnNdkbMmyEer16s

lab:
	@uv run jupyter lab --ServerApp.port=$(JUPYTER_PORT) --ServerApp.port_retries=0 --PasswordIdentityProvider.hashed_password='$(JUPYTER_HASH)'


########################################################################################################################
# Setup winterload
########################################################################################################################
.PHONY: setup lab
setup:
	# @make stop
	@$(COMPOSE) up -d --wait
# BigQuery: provision the project first -- storage.setup() creates the tables inside Terraform's dataset.
ifeq ($(OFFLINE_STORAGE),bigquery)
	@make tf-init
	@make tf-apply
endif
	@uv run python -c "from winterload.storage import setup; setup()"
	@make apply