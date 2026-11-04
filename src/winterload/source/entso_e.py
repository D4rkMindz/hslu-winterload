import datetime as dt
import logging
from typing import TypedDict
from xml.etree import ElementTree as ET

import requests

from winterload import DEFAULT_END_DATE, DEFAULT_START_DATE
from winterload.config import settings


class EnergyLoadRecord(TypedDict):
    timestamp: dt.datetime
    quantity: float
    bidding_zone: str
    unit: str


class EntsoE(object):
    BASE_URL = "https://web-api.tp.entsoe.eu/api"
    NS = {"ns": "urn:iec62325.351:tc57wg16:451-6:generationloaddocument:3:0"}
    RESOLUTION_TO_TIMEDELTA: dict[str, dt.timedelta] = {
        "PT15M": dt.timedelta(minutes=15),
        "PT30M": dt.timedelta(minutes=30),
        "PT60M": dt.timedelta(hours=1),
        "P1D": dt.timedelta(days=1),
    }

    def __init__(self):
        self.api_key = settings.entsoe_api_key
        self.logger = logging.getLogger(self.__class__.__name__)

    def get_load(self, start_date: dt.datetime = DEFAULT_START_DATE,
                 end_date: dt.datetime = DEFAULT_END_DATE) -> list[EnergyLoadRecord]:
        # curl --location 'https://web-api.tp.entsoe.eu/api?documentType=A65&processType=A16&outBiddingZone_Domain=10YCZ-CEPS-----N&periodStart=202303030000&periodEnd=202303060000' \
        # --header 'SECURITY_TOKEN: •••••••'
        self.logger.info("Getting energy load data from ENTSO-E API between %s and %s", start_date, end_date)
        r = requests.get(f"{self.BASE_URL}", params={
            # https://transparencyplatform.zendesk.com/hc/en-us/articles/15885757676308-Area-List-with-Energy-Identification-Code-EIC
            "outBiddingZone_Domain": "10YCH-SWISSGRIDZ",
            "documentType": "A65",  # System load
            "processType": "A16",  # Realised
            "curveType": "A03",  # include all measurements
            "periodStart": start_date.strftime("%Y%m%d%H%M"),
            "periodEnd": end_date.strftime("%Y%m%d%H%M"),
        }, headers={"SECURITY_TOKEN": self.api_key})
        return self._parse_energy_load(r.content)

    def _parse_energy_load(self, xml: bytes | str) -> list[EnergyLoadRecord]:
        root = ET.fromstring(xml)
        energy_load_rows: list[EnergyLoadRecord] = []
        for time_series in root.findall("ns:TimeSeries", self.NS):
            bidding_zone = time_series.findtext("ns:outBiddingZone_Domain.mRID", namespaces=self.NS)
            unit = time_series.findtext("ns:quantity_Measure_Unit.name", namespaces=self.NS)
            for period in time_series.findall("ns:Period", self.NS):
                period_start_text = str(period.findtext("ns:timeInterval/ns:start", namespaces=self.NS))
                period_start = dt.datetime.fromisoformat(period_start_text)
                resolution = period.findtext("ns:resolution", namespaces=self.NS)
                step = self.RESOLUTION_TO_TIMEDELTA[str(resolution)]
                for point in period.findall("ns:Point", self.NS):
                    position = int(str(point.findtext("ns:position", namespaces=self.NS)))
                    energy_load: EnergyLoadRecord = {
                        "timestamp": period_start + (position - 1) * step,
                        "quantity": float(str(point.findtext("ns:quantity", namespaces=self.NS))),
                        "bidding_zone": str(bidding_zone),
                        "unit": str(unit),
                    }
                    energy_load_rows.append(energy_load)

        self.logger.debug("Parsed %i rows", len(energy_load_rows))
        return energy_load_rows
