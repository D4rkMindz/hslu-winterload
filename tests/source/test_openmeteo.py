from winterload.source.openmeteo import FORECAST_MODEL, MODELS, REANALYSIS_MODEL, OpenMeteo, OpenMeteoResult


def test_models_are_the_result_dict_keys_not_the_model_result_dict_keys():
    # MODELS used to be built from OpenMeteoModelResult.__annotations__ (its only key is "hourly"),
    # not OpenMeteoResult.__annotations__ (its keys are the model names, e.g. "ecmwf_ifs"). Every
    # API call passing models=MODELS sent models=["hourly"], which Open-Meteo rejects outright.
    assert MODELS == list(OpenMeteoResult.__annotations__)
    assert MODELS == ["ecmwf_ifs"]


def test_get_run_metadata_reads_the_static_meta_file(monkeypatch):
    calls = []

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"last_run_availability_time": 123}

    def fake_get(url, **kwargs):
        calls.append(url)
        return FakeResponse()

    monkeypatch.setattr("winterload.source.openmeteo.requests.get", fake_get)

    result = OpenMeteo.get_run_metadata("https://example.test")

    assert calls == ["https://example.test/data/ecmwf_ifs/static/meta.json"]
    assert result == {"last_run_availability_time": 123}


def test_model_names():
    assert FORECAST_MODEL == "ecmwf_ifs"
    assert REANALYSIS_MODEL == "era5"


def test_the_cache_keeps_octet_stream_answers_and_skips_json_errors():
    from types import SimpleNamespace
    keep = OpenMeteo().client._session.settings.filter_fn
    assert keep(SimpleNamespace(headers={"Content-Type": "application/octet-stream"}))
    assert not keep(SimpleNamespace(headers={"Content-Type": "application/json; charset=utf-8"}))
    assert not keep(SimpleNamespace(headers={}))
