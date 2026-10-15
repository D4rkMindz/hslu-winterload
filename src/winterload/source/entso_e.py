import datetime as dt
from xml.etree import ElementTree as ET

import requests

from winterload.config import settings


class EntsoE(object):
    BASE_URL = "https://web-api.tp.entsoe.eu/api"
    NS = {"ns": "urn:iec62325.351:tc57wg16:451-6:generationloaddocument:3:0"}
    RESOLUTION_TO_TIMEDELTA = {
        "PT15M": dt.timedelta(minutes=15),
        "PT30M": dt.timedelta(minutes=30),
        "PT60M": dt.timedelta(hours=1),
        "P1D": dt.timedelta(days=1),
    }

    def __init__(self):
        self.api_key = settings.entsoe_api_key

    def get_load(self, start_date=dt.datetime(2015, 1, 1, 0, 0, 0), end_date=dt.datetime(2015, 12, 31, 23, 59, 59)):
        # curl --location 'https://web-api.tp.entsoe.eu/api?documentType=A65&processType=A16&outBiddingZone_Domain=10YCZ-CEPS-----N&periodStart=202303030000&periodEnd=202303060000' \
        # --header 'SECURITY_TOKEN: •••••••'
        r = requests.get(f"{self.BASE_URL}", params={
            # https://transparencyplatform.zendesk.com/hc/en-us/articles/15885757676308-Area-List-with-Energy-Identification-Code-EIC
            "outBiddingZone_Domain": "10YCH-SWISSGRIDZ",
            "documentType": "A65",  # System load
            "processType": "A16",  # Realised
            "curveType": "A03",  # include all measurements
            "periodStart": start_date.strftime("%Y%m%d%H%M"),
            "periodEnd": end_date.strftime("%Y%m%d%H%M"),
        }, headers={"SECURITY_TOKEN": self.api_key})
        return self._parse_load(r.content)

    def _parse_load(self, xml: bytes | str) -> list:
        root = ET.fromstring(xml)
        rows = []
        for time_series in root.findall("ns:TimeSeries", self.NS):
            bidding_zone = time_series.findtext("ns:outBiddingZone_Domain.mRID", namespaces=self.NS)
            unit = time_series.findtext("ns:quantity_Measure_Unit.name", namespaces=self.NS)
            for period in time_series.findall("ns:Period", self.NS):
                period_start = dt.datetime.fromisoformat(
                    period.findtext("ns:timeInterval/ns:start", namespaces=self.NS))
                resolution = period.findtext("ns:resolution", namespaces=self.NS)
                step = self.RESOLUTION_TO_TIMEDELTA[resolution]
                for point in period.findall("ns:Point", self.NS):
                    position = int(point.findtext("ns:position", namespaces=self.NS))
                    rows.append({
                        "timestamp": period_start + (position - 1) * step,
                        "quantity": float(point.findtext("ns:quantity", namespaces=self.NS)),
                        "bidding_zone": bidding_zone,
                        "unit": unit,
                    })

        return rows
