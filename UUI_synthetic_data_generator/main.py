import math
import random
import csv
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

import yaml  # pip install pyyaml


# ============================
# Data model
# ============================

@dataclass
class Sensor:
    sensor_id: str
    sensor_type: str
    layer: str
    sector: str
    zone: str
    x: float
    y: float
    p_detect: float
    false_rate_per_hr: float

@dataclass
class Edge:
    src: str
    dst: str
    edge_type: str
    transit_mean_s: float
    transit_sigma_s: float
    requires_breach: bool
    breach_mean_s: float
    breach_sigma_s: float

@dataclass
class Event:
    timestamp: str
    event_id: str
    sensor_id: str
    sensor_type: str
    layer: str
    sector: str
    zone: str
    event_type: str             # "alarm"
    assessment: str             # "valid" | "invalid" | "intrusion"
    scenario_label: str         # "normal" | "false_positive" | "intrusion:..."
    ground_truth: str           # "benign" | "false_positive" | "intrusion"
    assessment_latency_s: float
    actor: str                  # "baseline" | "adversary" | "env"
    meta: str


# ============================
# Helpers
# ============================

def iso(ts: datetime) -> str:
    return ts.isoformat(timespec="seconds")

def normal_pos(mean: float, sigma: float) -> float:
    return max(0.0, random.gauss(mean, sigma))

def poisson_count(lmbda: float) -> int:
    L = math.exp(-lmbda)
    k, p = 0, 1.0
    while p > L:
        k += 1
        p *= random.random()
    return k - 1

def parse_hhmm(hhmm: str) -> Tuple[int, int]:
    h, m = hhmm.split(":")
    return int(h), int(m)

def random_time_near(base: datetime, sigma_minutes: float) -> datetime:
    return base + timedelta(minutes=random.gauss(0.0, sigma_minutes))

def build_edge_map(edges: List[Edge]) -> Dict[str, List[Edge]]:
    m: Dict[str, List[Edge]] = {}
    for e in edges:
        m.setdefault(e.src, []).append(e)
    return m

def expected_edge_time(e: Edge) -> float:
    return (e.breach_mean_s if e.requires_breach else 0.0) + e.transit_mean_s

def estimate_remaining_time_to_asset(edges: List[Edge], asset_zone: str) -> Dict[str, float]:
    zones = set()
    for e in edges:
        zones.add(e.src); zones.add(e.dst)

    rev: Dict[str, List[Tuple[str, float]]] = {}
    for e in edges:
        rev.setdefault(e.dst, []).append((e.src, expected_edge_time(e)))

    dist = {z: float("inf") for z in zones}
    dist[asset_zone] = 0.0
    visited = set()

    while True:
        cur, curd = None, float("inf")
        for z, d in dist.items():
            if z not in visited and d < curd:
                cur, curd = z, d
        if cur is None:
            break
        visited.add(cur)
        for prev, w in rev.get(cur, []):
            if dist[prev] > dist[cur] + w:
                dist[prev] = dist[cur] + w
    return dist


# ============================
# Config load
# ============================

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ============================
# Build site from config
# ============================

def sector_xy(perim_w: float, perim_h: float, sector: dict) -> Tuple[float, float]:
    side = sector["side"]
    frac = float(sector["frac"])
    if side == "N":
        return perim_w * frac, perim_h
    if side == "S":
        return perim_w * frac, 0.0
    if side == "E":
        return perim_w, perim_h * frac
    return 0.0, perim_h * frac

def allocate_false_rates(cfg: dict, sensors: Dict[str, Sensor]) -> None:
    """
    Allocate per-sensor false_rate_per_hr to hit category daily targets approximately.
    Categories are inferred from sensor.layer and sensor_type.
    """
    targets = cfg["false_alarms"]["targets_per_day"]
    jitter_lo, jitter_hi = cfg["false_alarms"]["per_sensor_jitter"]

    # Categorize sensors into buckets
    buckets: Dict[str, List[Sensor]] = {k: [] for k in targets.keys()}
    for s in sensors.values():
        if s.layer == "PERIMETER" and s.sensor_type != "ACCESS_CTRL":
            buckets["perimeter"].append(s)
        elif s.layer == "PROTECTED_AREA":
            buckets["protected_area"].append(s)
        elif s.layer == "SKIN":
            # wall breakwire should be nearly false-free; still assign but tiny
            buckets["skin"].append(s)
        elif s.layer == "INTERIOR":
            buckets["interior"].append(s)
        elif s.layer == "ASSET":
            buckets["asset"].append(s)

    # Assign rates
    for cat, sensor_list in buckets.items():
        if not sensor_list:
            continue
        per_day = float(targets.get(cat, 0.0))
        base_per_sensor_per_day = per_day / len(sensor_list)
        base_per_sensor_per_hr = base_per_sensor_per_day / 24.0

        for s in sensor_list:
            # Keep wall breakwire extremely low regardless
            if s.sensor_type == "BREAKWIRE":
                s.false_rate_per_hr = min(base_per_sensor_per_hr, 0.05 / 24.0)
            elif s.sensor_type == "ACCESS_CTRL":
                s.false_rate_per_hr = min(base_per_sensor_per_hr, 0.1 / 24.0)
            else:
                s.false_rate_per_hr = base_per_sensor_per_hr * random.uniform(jitter_lo, jitter_hi)

def build_edges(cfg: dict, perim_sectors: List[dict]) -> List[Edge]:
    dly = cfg["delays"]

    # Helper to map edge_type -> delay model
    def edge_params(edge_type: str) -> Tuple[bool, float, float, float, float]:
        if edge_type == "ACCESS_TRANSIT":
            return (False, 0, 0, 10, 3)
        if edge_type == "FENCE":
            b = dly["fence_breach_s"]; t = dly["pa_transit_s"]
            return (True, b["mean"], b["sigma"], 20, 6)  # short transit to PA radar node
        if edge_type == "PA_TRANSIT":
            t = dly["pa_transit_s"]
            return (False, 0, 0, t["mean"], t["sigma"])
        if edge_type == "SKIN_DOOR":
            b = dly["skin_breach_s"]["DOOR"]
            return (True, b["mean"], b["sigma"], 10, 4)
        if edge_type == "SKIN_WIN":
            b = dly["skin_breach_s"]["WIN"]
            return (True, b["mean"], b["sigma"], 12, 5)
        if edge_type == "SKIN_WALL":
            b = dly["skin_breach_s"]["WALL"]
            return (True, b["mean"], b["sigma"], 15, 6)
        if edge_type == "ASSET_DOOR":
            b = dly["asset_door_breach_s"]
            return (True, b["mean"], b["sigma"], 30, 10)
        # INT_TRANSIT default
        t = dly["interior_transit_s"]
        return (False, 0, 0, t["mean"], t["sigma"])

    edges: List[Edge] = []

    # Perimeter -> PA radar edges (for adversary paths); exclude access sector
    access_sector_id = None
    for s in perim_sectors:
        if s.get("access_control", False):
            access_sector_id = s["id"]

    for s in perim_sectors:
        sid = s["id"]
        if sid == access_sector_id:
            continue
        quadrant = sid[0]  # "N","E","S","W"
        src = f"PERIM_{sid}"
        dst = f"PA_RADAR_{quadrant}"
        requires, bmean, bsig, tmean, tsig = edge_params("FENCE")
        edges.append(Edge(src, dst, "FENCE", tmean, tsig, requires, bmean, bsig))

    # Explicit edges from config
    for e in cfg["connectivity"]["edges"]:
        et = e["edge_type"]
        requires, bmean, bsig, tmean, tsig = edge_params(et)
        edges.append(Edge(
            src=e["src"],
            dst=e["dst"],
            edge_type=et,
            transit_mean_s=tmean,
            transit_sigma_s=tsig,
            requires_breach=requires,
            breach_mean_s=bmean,
            breach_sigma_s=bsig
        ))

    return edges

def build_site_from_config(cfg: dict) -> Tuple[Dict[str, Sensor], List[Edge], Dict[str, List[str]], Dict[str, str]]:
    sensors: Dict[str, Sensor] = {}
    zone_sensors: Dict[str, List[str]] = {}

    def add_sensor(s: Sensor):
        sensors[s.sensor_id] = s
        zone_sensors.setdefault(s.zone, []).append(s.sensor_id)

    perim = cfg["site"]["perimeter"]
    bldg = cfg["site"]["building"]
    perim_w = float(perim["width"])
    perim_h = float(perim["height"])

    # Perimeter sectors & sensors (2 per sector)
    sectors = perim["sectors"]
    for sec in sectors:
        sid = sec["id"]
        x, y = sector_xy(perim_w, perim_h, sec)
        zone = f"PERIM_{sid}"
        for spec in perim["sensors_per_sector"]:
            stype = spec["type"]
            s = Sensor(
                sensor_id=f"{stype}_{sid}",
                sensor_type=stype,
                layer="PERIMETER",
                sector=sid,
                zone=zone,
                x=x, y=y,
                p_detect=float(spec["p_detect"]),
                false_rate_per_hr=0.0,  # allocated later
            )
            add_sensor(s)

        # Access control
        if sec.get("access_control", False):
            turnstile_zone = f"TURNSTILE_{sid}"
            add_sensor(Sensor(
                sensor_id=f"ACS_{sid}",
                sensor_type="ACCESS_CTRL",
                layer="PERIMETER",
                sector=sid,
                zone=turnstile_zone,
                x=x, y=y,
                p_detect=0.999,
                false_rate_per_hr=0.0,
            ))

    # Protected area quadrant radars
    for r in cfg["protected_area"]["quadrant_radars"]:
        add_sensor(Sensor(
            sensor_id=r["id"],
            sensor_type=r["type"],
            layer="PROTECTED_AREA",
            sector=r["quadrant"],
            zone=r["id"],  # zone = radar node
            x=0.0, y=0.0,  # optional; can be computed if desired
            p_detect=float(r["p_detect"]),
            false_rate_per_hr=0.0,
        ))

    # Building skin points + sensors
    for sp in bldg["skin_points"]:
        zone = sp["zone"]
        kind = sp["kind"]
        for ss in bldg["skin_sensors"][kind]:
            sid = ss.get("sensor_id", f"{ss['type']}_{zone}")
            add_sensor(Sensor(
                sensor_id=sid,
                sensor_type=ss["type"],
                layer="SKIN",
                sector="BLDG",
                zone=zone,
                x=0.0, y=0.0,
                p_detect=float(ss["p_detect"]),
                false_rate_per_hr=0.0,
            ))

    # Interior sensors
    for spec in bldg["interior_sensors"]:
        zone = spec["zone"]
        sid = spec.get("sensor_id", f"{spec['type']}_{zone}")
        layer = next(z["layer"] for z in bldg["zones"] if z["id"] == zone)
        add_sensor(Sensor(
            sensor_id=sid,
            sensor_type=spec["type"],
            layer=layer,
            sector="BLDG",
            zone=zone,
            x=0.0, y=0.0,
            p_detect=float(spec["p_detect"]),
            false_rate_per_hr=0.0,
        ))

    # Allocate false alarm rates to meet per-day targets
    allocate_false_rates(cfg, sensors)

    # Build edges
    edges = build_edges(cfg, sectors)

    # Key zones
    access_sector_id = next(s["id"] for s in sectors if s.get("access_control", False))
    key = {
        "ASSET_ZONE": "ASSET_ROOM",
        "TURNSTILE_ZONE": f"TURNSTILE_{access_sector_id}",
        "ACCESS_SECTOR": access_sector_id,
    }
    return sensors, edges, zone_sensors, key


# ============================
# Assessment sampling
# ============================

def sample_from_weights(weights: Dict[str, float]) -> str:
    items = list(weights.items())
    labels = [k for k, _ in items]
    w = [float(v) for _, v in items]
    return random.choices(labels, weights=w, k=1)[0]


# ============================
# Generators
# ============================

def generate_false_positives(cfg: dict, sensors: Dict[str, Sensor], start: datetime, end: datetime,
                            dataset_label: str, event_id_start: int = 0) -> Tuple[List[Event], int]:
    events: List[Event] = []
    eid = event_id_start
    hours = (end - start).total_seconds() / 3600.0
    weights = cfg["behavior"]["assessments"]["nuisance"]["weights"]

    for s in sensors.values():
        k = poisson_count(s.false_rate_per_hr * hours)
        for _ in range(k):
            t = start + timedelta(seconds=random.random() * (end - start).total_seconds())
            assessment = sample_from_weights(weights)
            events.append(Event(
                timestamp=iso(t),
                event_id=f"E{eid:09d}",
                sensor_id=s.sensor_id,
                sensor_type=s.sensor_type,
                layer=s.layer,
                sector=s.sector,
                zone=s.zone,
                event_type="alarm",
                assessment=assessment,
                scenario_label="false_positive",
                ground_truth="false_positive",
                assessment_latency_s=normal_pos(20, 8),
                actor="env",
                meta=dataset_label
            ))
            eid += 1

    events.sort(key=lambda e: e.timestamp)
    return events, eid

def generate_benign(cfg: dict, sensors: Dict[str, Sensor], zone_sensors: Dict[str, List[str]],
                    edges: List[Edge], start: datetime, end: datetime,
                    turnstile_zone: str, event_id_start: int, dataset_label: str) -> Tuple[List[Event], int]:
    events: List[Event] = []
    eid = event_id_start
    edge_map = build_edge_map(edges)

    mis_cfg = cfg["behavior"]["assessments"]["benign_misassessment"]
    p_err = float(mis_cfg["p_error"])
    when_err = mis_cfg["when_error_weights"]

    sched = cfg["behavior"]["benign_schedule"]

    def log_zone(ts: datetime, zone: str, meta: str, force_valid: bool = False):
        nonlocal eid
        sids = zone_sensors.get(zone, [])
        random.shuffle(sids)
        for sid in sids[:2]:
            s = sensors[sid]
            if random.random() <= s.p_detect:
                assessment = "valid" if force_valid else "valid"
                if (not force_valid) and random.random() < p_err:
                    assessment = sample_from_weights(when_err)
                events.append(Event(
                    timestamp=iso(ts),
                    event_id=f"E{eid:09d}",
                    sensor_id=s.sensor_id,
                    sensor_type=s.sensor_type,
                    layer=s.layer,
                    sector=s.sector,
                    zone=s.zone,
                    event_type="alarm",
                    assessment=assessment,
                    scenario_label="normal",
                    ground_truth="benign",
                    assessment_latency_s=normal_pos(15, 6),
                    actor="baseline",
                    meta=f"{dataset_label}:{meta}",
                ))
                eid += 1

    def transit(prev: str, cur: str) -> float:
        for e in edge_map.get(prev, []):
            if e.dst == cur:
                return normal_pos(e.transit_mean_s, e.transit_sigma_s)
        return normal_pos(25, 10)

    def choose_dest(p_go_asset: float) -> str:
        if random.random() < p_go_asset:
            return "ASSET_ROOM"
        return random.choice(["ROOM_ADMIN", "ROOM_SECURITY"])

    def run_trip(t0: datetime, dest: str, tag: str):
        # Trusted entry at turnstile is always valid
        log_zone(t0, turnstile_zone, f"{tag}:turnstile_entry", force_valid=True)

        route_in = [turnstile_zone, "PA_RADAR_S", "SKIN_S_DOOR", "ROOM_S_ENTRY", "HALL_MAIN", dest]
        route_out = [dest, "HALL_MAIN", "ROOM_S_ENTRY", "SKIN_S_DOOR", "PA_RADAR_S", turnstile_zone]

        t = t0
        for i in range(1, len(route_in)):
            t += timedelta(seconds=transit(route_in[i-1], route_in[i]))
            log_zone(t, route_in[i], f"{tag}:move_to:{dest}")

        t += timedelta(minutes=normal_pos(20, 10))

        for i in range(1, len(route_out)):
            t += timedelta(seconds=transit(route_out[i-1], route_out[i]))
            log_zone(t, route_out[i], f"{tag}:exit_from:{dest}")

    # iterate each day
    day = datetime(start.year, start.month, start.day)
    while day < end:
        for key, spec in sched.items():
            if key == "stragglers":
                bh = spec["business_hours"]
                h0, m0 = parse_hhmm(bh["start"])
                h1, m1 = parse_hhmm(bh["end"])
                count = int(spec["count"])
                sigma = float(spec["sigma_minutes"])
                p_go_asset = float(spec["p_go_asset"])
                for _ in range(count):
                    hour = random.randint(h0, h1-1)
                    minute = random.randint(0, 59)
                    t0 = random_time_near(datetime(day.year, day.month, day.day, hour, minute, 0), sigma)
                    if start <= t0 <= end:
                        run_trip(t0, choose_dest(p_go_asset), "straggler")
                continue

            hh, mm = parse_hhmm(spec["time"])
            base = datetime(day.year, day.month, day.day, hh, mm, 0)
            sigma = float(spec["sigma_minutes"])
            count = int(spec["count"])
            p_go_asset = float(spec["p_go_asset"])
            for _ in range(count):
                t0 = random_time_near(base, sigma)
                if start <= t0 <= end:
                    run_trip(t0, choose_dest(p_go_asset), key)

        day += timedelta(days=1)

    events.sort(key=lambda e: e.timestamp)
    return events, eid

def inject_intrusions(cfg: dict, sensors: Dict[str, Sensor], zone_sensors: Dict[str, List[str]],
                     edges: List[Edge], asset_zone: str,
                     start: datetime, end: datetime,
                     n_attacks: int, event_id_start: int, dataset_label: str) -> Tuple[List[Event], int]:
    events: List[Event] = []
    eid = event_id_start
    edge_map = build_edge_map(edges)
    dist_to_asset = estimate_remaining_time_to_asset(edges, asset_zone)

    p_bypass = float(cfg["attacks"]["p_bypass_perimeter"])
    op = cfg["behavior"]["assessments"]["intrusion_operator"]
    p_correct = float(op["p_correct"])
    when_wrong = op["when_wrong_weights"]
    max_steps = int(cfg["attacks"]["max_steps"])

    # adversary starts: any non-access perimeter sector
    access_sector = cfg["site"]["perimeter"]["sectors"]
    access_id = next(s["id"] for s in access_sector if s.get("access_control", False))
    perim_ids = [s["id"] for s in cfg["site"]["perimeter"]["sectors"] if s["id"] != access_id]
    start_zones = [f"PERIM_{sid}" for sid in perim_ids]

    def trigger_zone(ts: datetime, zone: str, step_label: str):
        nonlocal eid
        only_types = {"BREAKWIRE"} if (zone.startswith("SKIN_") and zone.endswith("_WALL")) else None
        sids = zone_sensors.get(zone, [])
        random.shuffle(sids)

        fired = 0
        for sid in sids:
            s = sensors[sid]
            if only_types is not None and s.sensor_type not in only_types:
                continue
            if random.random() <= s.p_detect:
                assessment = "intrusion" if random.random() < p_correct else sample_from_weights(when_wrong)
                events.append(Event(
                    timestamp=iso(ts),
                    event_id=f"E{eid:09d}",
                    sensor_id=s.sensor_id,
                    sensor_type=s.sensor_type,
                    layer=s.layer,
                    sector=s.sector,
                    zone=s.zone,
                    event_type="alarm",
                    assessment=assessment,
                    scenario_label=step_label,
                    ground_truth="intrusion",
                    assessment_latency_s=normal_pos(18, 7),
                    actor="adversary",
                    meta=f"{dataset_label}:attack",
                ))
                eid += 1
                fired += 1
            if fired >= 2:
                break

    for k in range(n_attacks):
        t = start + timedelta(hours=random.uniform(0, (end - start).total_seconds() / 3600))
        zone = random.choice(start_zones)
        attack_id = f"attack_{k:02d}"

        for step in range(max_steps):
            label = f"intrusion:{attack_id}:step_{step:02d}:{zone}"

            if zone.startswith("PERIM_") and random.random() < p_bypass:
                pass
            else:
                trigger_zone(t, zone, label)

            if zone == asset_zone:
                break

            candidates = edge_map.get(zone, [])
            if not candidates:
                break

            candidates = sorted(candidates, key=lambda e: expected_edge_time(e) + dist_to_asset.get(e.dst, float("inf")))
            chosen = candidates[0]

            if chosen.requires_breach:
                t += timedelta(seconds=normal_pos(chosen.breach_mean_s, chosen.breach_sigma_s))
            t += timedelta(seconds=normal_pos(chosen.transit_mean_s, chosen.transit_sigma_s))
            zone = chosen.dst

    events.sort(key=lambda e: e.timestamp)
    return events, eid


# ============================
# Output
# ============================

def write_csv(path: str, events: List[Event]):
    if not events:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(events[0]).keys()))
        w.writeheader()
        for e in events:
            w.writerow(asdict(e))


# ============================
# Driver
# ============================

def generate_all(config_path: str, out_prefix: str,
                 seed: int,
                 start: datetime,
                 days_train: int, days_scenario: int, days_val: int):
    random.seed(seed)
    cfg = load_config(config_path)
    sensors, edges, zone_sensors, key = build_site_from_config(cfg)

    asset_zone = key["ASSET_ZONE"]
    turnstile_zone = key["TURNSTILE_ZONE"]

    # TRAIN: nuisance + benign
    train_start = start
    train_end = start + timedelta(days=days_train)
    ev_train, eid = generate_false_positives(cfg, sensors, train_start, train_end, "train", 0)
    ev_ben, eid = generate_benign(cfg, sensors, zone_sensors, edges, train_start, train_end, turnstile_zone, eid, "train")
    ev_train.extend(ev_ben)
    ev_train.sort(key=lambda e: e.timestamp)
    write_csv(f"{out_prefix}_train.csv", ev_train)

    # SCENARIO: nuisance + benign + intrusions
    scen_start = train_end
    scen_end = scen_start + timedelta(days=days_scenario)
    ev_scen, eid2 = generate_false_positives(cfg, sensors, scen_start, scen_end, "scenario", 0)
    ev_ben2, eid2 = generate_benign(cfg, sensors, zone_sensors, edges, scen_start, scen_end, turnstile_zone, eid2, "scenario")
    ev_scen.extend(ev_ben2)

    n_attacks = int(cfg["attacks"]["count_scenario"])
    ev_atk, eid2 = inject_intrusions(cfg, sensors, zone_sensors, edges, asset_zone, scen_start, scen_end, n_attacks, eid2, "scenario")
    ev_scen.extend(ev_atk)
    ev_scen.sort(key=lambda e: e.timestamp)
    write_csv(f"{out_prefix}_scenario.csv", ev_scen)

    # VALIDATION: same config but different seed could be applied; you can also add "validation overrides" in YAML if desired
    random.seed(seed + 101)
    val_start = scen_end
    val_end = val_start + timedelta(days=days_val)
    ev_val, eid3 = generate_false_positives(cfg, sensors, val_start, val_end, "validation", 0)
    ev_ben3, eid3 = generate_benign(cfg, sensors, zone_sensors, edges, val_start, val_end, turnstile_zone, eid3, "validation")
    ev_val.extend(ev_ben3)

    n_attacks_val = int(cfg["attacks"]["count_validation"])
    ev_atk2, eid3 = inject_intrusions(cfg, sensors, zone_sensors, edges, asset_zone, val_start, val_end, n_attacks_val, eid3, "validation")
    ev_val.extend(ev_atk2)
    ev_val.sort(key=lambda e: e.timestamp)
    write_csv(f"{out_prefix}_validation.csv", ev_val)


if __name__ == "__main__":
    generate_all(
        config_path="synthetic_data/site_config.yaml",
        out_prefix="synthetic_data/sythetic_data",
        seed=21,
        start=datetime(2026, 1, 1, 0, 0, 0),
        days_train=14,
        days_scenario=7,
        days_val=7,
    )