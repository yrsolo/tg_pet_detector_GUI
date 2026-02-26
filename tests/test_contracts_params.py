from contracts.contracts import ShadowParams


def test_shadow_params_to_dict_casts_to_strings():
    p = ShadowParams(rot=15, max_objects=3, return_debug=True)
    d = p.to_dict()

    assert d["rot"] == "15"
    assert d["max_objects"] == "3"
    assert d["return_debug"] == "1"


def test_shadow_params_from_dict_parses_types_and_defaults():
    form = {"rot": "20", "max_objects": "5", "return_debug": "1"}
    p = ShadowParams.from_dict(form)

    assert p.rot == 20
    assert p.max_objects == 5
    assert p.return_debug is True

    # дефолты, если мусор/нет ключей
    p2 = ShadowParams.from_dict({"rot": "not_int"})
    assert p2.rot == 0
    assert p2.max_objects == 4
    assert p2.return_debug is False
