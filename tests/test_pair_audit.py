import copy

import pytest

from rssr.pair_audit import audit_pair


def fixture():
    image = dict(source_version="synthetic fixture", license="fixture only",
                 crs="EPSG:32631", units="reflectance", mask_evidence="fixture mask",
                 sha256="a"*64, north_up=True, width=13, height=7,
                 bounds=[0, 0, 130, 70], bands=["red", "green", "blue"],
                 acquired_at="2026-01-01T00:00:00Z")
    pair = dict(pair_id="fixture", region_id="fixture-region", purpose="development_inspection",
                band_mapping_evidence="synthetic equal responses", radiometry_evidence="synthetic identity",
                lr=image, hr=copy.deepcopy(image),
                registration=dict(residual_hr_pixels=0, evidence="synthetic known grid"))
    pair["hr"].update(width=26, height=14)
    policy = dict(protocol_id="TEST ONLY, not sensor acceptance", scale=2,
                  extent_tolerance=0, max_time_delta_seconds=0, max_registration_hr_pixels=0)
    return pair, policy


def test_rectangular_odd_pair_is_declarations_only():
    pair, policy = fixture()
    result = audit_pair(pair, policy)
    assert result["status"] == "declarations_compatible"
    assert result["independently_verified"] is False


@pytest.mark.parametrize("field,value", [
    ("width", 25), ("crs", "EPSG:4326"), ("bounds", [1, 0, 131, 70]),
    ("bands", ["blue", "green", "red"]), ("units", "DN"),
    ("acquired_at", "2026-01-01T00:00:01Z"), ("acquired_at", "2026-01-01T00:00:00"),
    ("mask_evidence", ""), ("sha256", "unknown"), ("north_up", False),
    ("bounds", [0, 0, float("nan"), 70]), ("height", 0),
    ("bands", "red"), ("bounds", [False, 0, 130, 70]),
])
def test_incompatible_pair_blocked(field, value):
    pair, policy = fixture()
    pair["hr"][field] = value
    assert audit_pair(pair, policy)["status"] == "blocked"


def test_missing_registration_and_test_purpose_blocked():
    pair, policy = fixture()
    del pair["registration"]
    pair["purpose"] = "test"
    assert len(audit_pair(pair, policy)["reasons"]) >= 2


@pytest.mark.parametrize("value", [float("nan"), -1, 1, True])
def test_registration_residual_rejected(value):
    pair, policy = fixture()
    pair["registration"]["residual_hr_pixels"] = value
    assert audit_pair(pair, policy)["status"] == "blocked"


def test_policy_missing_is_not_implicitly_accepted():
    pair, _ = fixture()
    assert audit_pair(pair, {})["status"] == "blocked"
