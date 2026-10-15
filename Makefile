JUPYTER_PORT := 8973
# password for local: XEfyLLhBxrK6LGGUMnDboT2*MFKbbfJe8_QqkgiQaNouuUtxPDt@m46j7@uFQHnG8-GP@ohsvZNp26qHrKxGh3MA6jbdgka9o
# for hashing your own: run
#  $ uv run python -c "from jupyter_server.auth import passwd; print(passwd('<your password>'))"
#  > argon2:$argon2id$v=19$m=10240,t=10,p=8$yqO/ISF45iJk38QaWUVOuQ$ZIxTrQ0Ct/RWe7TyEybtE9hAa4wsEnNdkbMmyEer16s
# copy the hash and replace $ with $$
# e.g. JUPYTER_HASH := argon2:$$argon2id$$v=19$$m=10240,t=10,p=8$$jzgEK6rOSQAsIhmXxjquiw$$9r/lNO2A4sEg100hY8i9eRYkwsmBMDmYreVAQunIG9I
JUPYTER_HASH := argon2:$$argon2id$$v=19$$m=10240,t=10,p=8$$yqO/ISF45iJk38QaWUVOuQ$$ZIxTrQ0Ct/RWe7TyEybtE9hAa4wsEnNdkbMmyEer16s

.PHONY: lab
lab:
	jupyter lab --ServerApp.port=$(JUPYTER_PORT) --ServerApp.port_retries=0 --PasswordIdentityProvider.hashed_password='$(JUPYTER_HASH)'