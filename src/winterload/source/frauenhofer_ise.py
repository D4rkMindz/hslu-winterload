from datetime import datetime

import requests


class FrauenhoferISE(object):
    BASE_URL = "https://api.energy-charts.info"

    def __init__(self):
        pass

    def get_public_power(self, start: datetime = datetime.combine(datetime.today(), datetime.min.time()),
                         end: datetime = datetime.now()):
        raise NotImplementedError("Frauenhofer ISE API (https://api.energy-charts.info/) is currently down since 2026-09-29, see https://github.com/evcc-io/evcc/issues/34437")
        r = requests.get(f"{self.BASE_URL}/public_power?country=ch&start={start}&end={end}")
        body = r.json()

        return body