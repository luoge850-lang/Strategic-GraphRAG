import importlib.util
import math
from pathlib import Path

spec = importlib.util.spec_from_file_location("live_acceptance", Path(__file__).with_name("live_acceptance.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_numeric_status_unit_and_value_all_required():
    case = module.CASES[0]
    good = {"status": "PASS", "value": case["expected"], "unit": case["unit"]}
    assert module.score(case, {"calculation": good})["status"] == "PASS"
    for change in ({"status": "INSUFFICIENT_EVIDENCE"}, {"unit": "EUR millions"},
                   {"value": None}, {"value": math.inf}, {"value": math.nan}, {"value": True}):
        assert module.score(case, {"calculation": {**good, **change}})["status"] == "FAIL"


def test_unanswerable_rejects_unrelated_citation():
    case = next(case for case in module.CASES if case["expected"] is None)
    response = {"calculation": {"status": "INSUFFICIENT_EVIDENCE"}, "citations": []}
    assert module.score(case, response)["status"] == "PASS"
    response["citations"] = [{"page": 80}]
    assert module.score(case, response)["status"] == "FAIL"
