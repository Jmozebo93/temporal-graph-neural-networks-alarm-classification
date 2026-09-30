"""Balanced synthetic alarm generator.

This is a sibling to main.py that keeps the same site model and event logic,
but generates a more balanced dataset for temporal GNN training.

Key differences from main.py:
  - intrusion events are added to the training split
  - false alarm targets are reduced and spread more evenly
  - split sizes are shorter so the class mix is less dominated by benign windows

Outputs are written next to the script using a balanced prefix.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path
from collections import Counter
import random

from main import (
    build_site_from_config,
    generate_benign,
    generate_false_positives,
    inject_intrusions,
    load_config,
    write_csv,
)


def make_balanced_config(cfg: dict) -> dict:
    balanced = deepcopy(cfg)

    # Reduce extreme nuisance skew so intrusion windows are not drowned out.
    balanced["false_alarms"]["targets_per_day"] = {
        "perimeter": 3.0,
        "protected_area": 1.0,
        "skin": 0.6,
        "interior": 1.0,
        "asset": 0.15,
    }

    # Slightly lower per-sensor randomization so the distribution stays stable.
    balanced["false_alarms"]["per_sensor_jitter"] = [0.9, 1.1]

    # Use a denser attack schedule so the training split contains positives.
    balanced["attacks"]["count_train"] = 6
    balanced["attacks"]["count_scenario"] = 8
    balanced["attacks"]["count_validation"] = 6

    # Keep each split shorter so the resulting graph windows are less skewed.
    balanced["balanced_split_days"] = {
        "train": 7,
        "scenario": 5,
        "validation": 5,
    }

    # After generation, each split is rebalanced to this many events per class.
    # Increase this if you want larger equal-sized files.
    balanced["balanced_generation"] = {
        "target_per_class": 500,
    }

    return balanced


def rebalance_events(events, target_per_class: int, seed: int):
    rng = random.Random(seed)
    buckets = {}
    for event in events:
        buckets.setdefault(event.ground_truth, []).append(event)

    labels = ["benign", "false_positive", "intrusion"]
    balanced = []
    for label in labels:
        bucket = buckets.get(label, [])
        if not bucket:
            continue
        if len(bucket) >= target_per_class:
            balanced.extend(rng.sample(bucket, target_per_class))
        else:
            balanced.extend(bucket)
            balanced.extend(rng.choices(bucket, k=target_per_class - len(bucket)))

    rng.shuffle(balanced)
    return balanced


def generate_balanced_all(
    config_path: str,
    out_prefix: str,
    seed: int,
    start: datetime,
    days_train: int,
    days_scenario: int,
    days_val: int,
):
    import random

    random.seed(seed)
    cfg = make_balanced_config(load_config(config_path))
    sensors, edges, zone_sensors, key = build_site_from_config(cfg)
    target_per_class = int(cfg.get("balanced_generation", {}).get("target_per_class", 500))

    asset_zone = key["ASSET_ZONE"]
    turnstile_zone = key["TURNSTILE_ZONE"]

    # TRAIN: benign + nuisance + a smaller number of intrusions.
    train_start = start
    train_end = start + timedelta(days=days_train)
    ev_train, eid = generate_false_positives(cfg, sensors, train_start, train_end, "balanced_train", 0)
    ev_ben, eid = generate_benign(
        cfg,
        sensors,
        zone_sensors,
        edges,
        train_start,
        train_end,
        turnstile_zone,
        eid,
        "balanced_train",
    )
    ev_train.extend(ev_ben)
    ev_atk_train, eid = inject_intrusions(
        cfg,
        sensors,
        zone_sensors,
        edges,
        asset_zone,
        train_start,
        train_end,
        int(cfg["attacks"]["count_train"]),
        eid,
        "balanced_train",
    )
    ev_train.extend(ev_atk_train)
    ev_train = rebalance_events(ev_train, target_per_class=target_per_class, seed=seed)
    ev_train.sort(key=lambda e: e.timestamp)
    write_csv(f"{out_prefix}_train.csv", ev_train)

    # SCENARIO: benign + nuisance + more intrusion coverage.
    scen_start = train_end
    scen_end = scen_start + timedelta(days=days_scenario)
    ev_scen, eid2 = generate_false_positives(cfg, sensors, scen_start, scen_end, "balanced_scenario", 0)
    ev_ben2, eid2 = generate_benign(
        cfg,
        sensors,
        zone_sensors,
        edges,
        scen_start,
        scen_end,
        turnstile_zone,
        eid2,
        "balanced_scenario",
    )
    ev_scen.extend(ev_ben2)
    ev_atk_scen, eid2 = inject_intrusions(
        cfg,
        sensors,
        zone_sensors,
        edges,
        asset_zone,
        scen_start,
        scen_end,
        int(cfg["attacks"]["count_scenario"]),
        eid2,
        "balanced_scenario",
    )
    ev_scen.extend(ev_atk_scen)
    ev_scen = rebalance_events(ev_scen, target_per_class=target_per_class, seed=seed + 1)
    ev_scen.sort(key=lambda e: e.timestamp)
    write_csv(f"{out_prefix}_scenario.csv", ev_scen)

    # VALIDATION: keep a fresh split but still include positives.
    random.seed(seed + 101)
    val_start = scen_end
    val_end = val_start + timedelta(days=days_val)
    ev_val, eid3 = generate_false_positives(cfg, sensors, val_start, val_end, "balanced_validation", 0)
    ev_ben3, eid3 = generate_benign(
        cfg,
        sensors,
        zone_sensors,
        edges,
        val_start,
        val_end,
        turnstile_zone,
        eid3,
        "balanced_validation",
    )
    ev_val.extend(ev_ben3)
    ev_atk_val, eid3 = inject_intrusions(
        cfg,
        sensors,
        zone_sensors,
        edges,
        asset_zone,
        val_start,
        val_end,
        int(cfg["attacks"]["count_validation"]),
        eid3,
        "balanced_validation",
    )
    ev_val.extend(ev_atk_val)
    ev_val = rebalance_events(ev_val, target_per_class=target_per_class, seed=seed + 2)
    ev_val.sort(key=lambda e: e.timestamp)
    write_csv(f"{out_prefix}_validation.csv", ev_val)


if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    balanced_prefix = str(script_dir / "balanced_synthetic_data")
    generate_balanced_all(
        config_path=str(script_dir / "site_config.yaml"),
        out_prefix=balanced_prefix,
        seed=21,
        start=datetime(2026, 1, 1, 0, 0, 0),
        days_train=7,
        days_scenario=5,
        days_val=5,
    )