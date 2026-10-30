from winterload.source.openmeteo import MODELS, OpenMeteoResult


def test_models_are_the_result_dict_keys_not_the_model_result_dict_keys():
    # MODELS used to be built from OpenMeteoModelResult.__annotations__ (its only key is "hourly"),
    # not OpenMeteoResult.__annotations__ (its keys are the model names, e.g. "ecmwf_ifs"). Every
    # API call passing models=MODELS sent models=["hourly"], which Open-Meteo rejects outright.
    assert MODELS == list(OpenMeteoResult.__annotations__)
    assert MODELS == ["ecmwf_ifs"]
