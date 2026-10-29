import logging
from datetime import datetime

import requests


class FrauenhoferISE(object):
    BASE_URL = "https://api.energy-charts.info"

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def get_public_power(self, start: datetime = datetime.combine(datetime.today(), datetime.min.time()),
                         end: datetime = datetime.now()):
        # raise NotImplementedError("Frauenhofer ISE API (https://api.energy-charts.info/) is currently down since 2026-09-29, see https://github.com/evcc-io/evcc/issues/34437")
        self.logger.info("Getting public power from %s to %s", start.isoformat(), end.isoformat())
        r = requests.get(f"{self.BASE_URL}/v2/public_power?country=ch&start={start}&end={end}", headers={
            "Accept": "application/json",
        })
        self.logger.debug("%s %s\n%s\n%s", r.status_code, r.apparent_encoding, r.headers, r.text)
        body = r.json()

        return self._parse_load(body["data"])

    def _parse_load(self, data):
        self.logger.debug("Parsing %s rows", len(data))
        return [{"timestamp": row["timestamp"], **row["values"]} for row in data]
