# Winterload

see `docs/proposal.pdf`

# Infrastructure
First, copy ``/.env.example`` to ``/.env`` and ``/.env.docker.example`` to ``/.env.docker``
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

# LICENSE
proprietary (for now)