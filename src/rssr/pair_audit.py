"""Metadata-only admission checks for a proposed cross-sensor pair.

No downloads, raster reads, calibration fitting or accuracy claims. Passing means
the supplied declarations are internally compatible, not independently verified.
Tolerance values must come from a separately frozen task-specific protocol.
"""
import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import re


def audit_pair(pair, policy):
    """Return blocking reasons; missing/malformed metadata fails closed.

    Grids use width/height in pixels and bounds [left,bottom,right,top] in the
    declared CRS. Only common-CRS, north-up, same-extent grids are admitted.
    Bands are ordered semantic names after an explicitly documented mapping.
    """
    errors = []
    try:
        scale = policy["scale"]
        if type(scale) is not int or scale <= 0:
            raise ValueError("policy scale must be a positive integer")
        for key in ("extent_tolerance", "max_time_delta_seconds", "max_registration_hr_pixels"):
            value = policy[key]
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError(f"invalid policy {key}")
        if not policy["protocol_id"].strip():
            raise ValueError("missing protocol ID")
        if pair["purpose"] != "development_inspection":
            errors.append("Only development inspection is supported; no test admission")
        for key in ("pair_id", "region_id", "band_mapping_evidence", "radiometry_evidence"):
            if not isinstance(pair[key], str) or not pair[key].strip():
                errors.append(f"Missing {key}")
        lr, hr = pair["lr"], pair["hr"]
        for side, item in (("lr", lr), ("hr", hr)):
            for key in ("source_version", "license", "crs", "units", "mask_evidence"):
                if not isinstance(item[key], str) or not item[key].strip():
                    errors.append(f"{side}: missing {key}")
            if not re.fullmatch(r"[0-9a-f]{64}", item["sha256"]):
                errors.append(f"{side}: invalid SHA256")
            if item["north_up"] is not True:
                errors.append(f"{side}: unsupported rotated grid")
            for dim in ("width", "height"):
                if type(item[dim]) is not int or item[dim] <= 0:
                    raise ValueError(f"{side}: invalid {dim}")
            bounds = item["bounds"]
            if not isinstance(bounds, list) or len(bounds) != 4 or not all(
                not isinstance(x, bool) and math.isfinite(x) for x in bounds
            ):
                raise ValueError(f"{side}: invalid bounds")
            if bounds[0] >= bounds[2] or bounds[1] >= bounds[3]:
                raise ValueError(f"{side}: empty/inverted bounds")
            if not isinstance(item["bands"], list) or not item["bands"] or any(
                not isinstance(b, str) or not b.strip() for b in item["bands"]
            ):
                raise ValueError(f"{side}: invalid bands")
            if len(set(item["bands"])) != len(item["bands"]):
                errors.append(f"{side}: duplicate bands")
        if hr["width"] != scale * lr["width"] or hr["height"] != scale * lr["height"]:
            errors.append("Dimensions do not match the declared native scale")
        if lr["crs"] != hr["crs"]:
            errors.append("CRS differs; explicit grid policy required")
        elif any(abs(a-b) > policy["extent_tolerance"] for a, b in zip(lr["bounds"], hr["bounds"])):
            errors.append("Grid extents differ")
        if lr["bands"] != hr["bands"]:
            errors.append("Ordered mapped bands differ")
        if lr["units"] != hr["units"]:
            errors.append("Radiometric units differ")
        times = [datetime.fromisoformat(x["acquired_at"].replace("Z", "+00:00")) for x in (lr, hr)]
        if any(t.utcoffset() is None for t in times):
            errors.append("Acquisition timestamp lacks timezone")
        elif abs((times[0]-times[1]).total_seconds()) > policy["max_time_delta_seconds"]:
            errors.append("Acquisition time separation exceeds policy")
        registration = pair["registration"]
        residual = registration["residual_hr_pixels"]
        if isinstance(residual, bool) or not math.isfinite(residual) or residual < 0:
            raise ValueError("invalid registration residual")
        if not registration["evidence"].strip():
            errors.append("Missing registration evidence")
        if residual > policy["max_registration_hr_pixels"]:
            errors.append("Registration residual exceeds policy")
    except (KeyError, TypeError, ValueError, AttributeError, OverflowError) as exc:
        errors.append(f"Invalid or missing metadata: {exc}")
    return {"status": "blocked" if errors else "declarations_compatible",
            "reasons": errors, "independently_verified": False,
            "scope": "Metadata declarations only; not scientific or test-set acceptance"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pair", type=Path)
    parser.add_argument("policy", type=Path)
    args = parser.parse_args()
    report = audit_pair(json.loads(args.pair.read_text(encoding="utf-8")),
                        json.loads(args.policy.read_text(encoding="utf-8")))
    print(json.dumps(report, indent=2, allow_nan=False))
    raise SystemExit(2 if report["status"] == "blocked" else 0)


if __name__ == "__main__":
    main()
