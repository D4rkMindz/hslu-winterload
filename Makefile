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
FEAST := uv run feast --chdir src/winterload/feature_repo/

apply:
	@$(FEAST) apply

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
	@make stop
	@$(COMPOSE) up -d --wait
	@uv run python -c "from winterload.storage import setup; setup()"
	@make apply