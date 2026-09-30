from __future__ import annotations

# Data-loading and graph-construction utilities for the Temporal GNN workflow.
# This module parses alarm records, assigns labels, and converts sliding windows
# of events into graph-structured inputs for the model.

import csv
import math
from datetime import datetime
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch_geometric.data import Data

# Standard label sets used for the two supported labeling schemes.
GROUND_TRUTH_CLASSES = ["benign", "false_positive", "intrusion"]
ASSESSMENT_CLASSES = ["valid", "invalid", "intrusion"]


def parse_timestamp(value: str) -> datetime:
    # Parse ISO-style timestamps so records can be ordered chronologically.
    text = str(value).strip()
    if not text:
        raise ValueError("Empty timestamp")
    return datetime.fromisoformat(text)


def load_csv_records(path: str) -> List[dict]:
    # Load a CSV file of alarms and order rows by timestamp.
    with open(path, "r", newline="") as handle:
        records = list(csv.DictReader(handle))
    return sorted(records, key=lambda record: parse_timestamp(record["timestamp"]))


def label_schema(label_field: str) -> Tuple[List[str], Callable[[dict], str]]:
    # Resolve the active label configuration and return the valid class names
    # along with the function that extracts the label from a single record.
    field = label_field.strip().lower()
    if field == "ground_truth":
        return GROUND_TRUTH_CLASSES, lambda record: str(record.get("ground_truth", "")).strip().lower()
    if field == "assessment":
        return ASSESSMENT_CLASSES, lambda record: str(record.get("assessment", "")).strip().lower()
    raise ValueError("label-field must be 'ground_truth' or 'assessment'")


class TemporalGraphBuilder:
    # Build sliding-window temporal graphs from a chronologically ordered record list.
    # Each graph contains node features, edges that reflect temporal proximity, and
    # a window-level label used for supervised training.
    def __init__(self, label_field: str, window_size: int, stride: int, max_time_gap_s: int = 3600):
        self.label_field = label_field.strip().lower()
        self.label_names, self.label_fn = label_schema(label_field)
        self.label_to_idx = {name: idx for idx, name in enumerate(self.label_names)}
        self.window_size = window_size
        self.stride = stride
        self.max_time_gap_s = max_time_gap_s
        self.vectorizer = DictVectorizer(sparse=False)

    def _window_label(self, window: Sequence[dict]) -> str:
        # Aggregate the event-level labels within a time window into one final
        # window label. This keeps the graph supervised at the window level.
        if self.label_field == "ground_truth":
            labels = [str(record.get("ground_truth", "")).strip().lower() for record in window]
            if "intrusion" in labels:
                return "intrusion"
            if "false_positive" in labels:
                return "false_positive"
            return "benign"

        if self.label_field == "assessment":
            labels = [str(record.get("assessment", "")).strip().lower() for record in window]
            if "intrusion" in labels:
                return "intrusion"
            if "invalid" in labels:
                return "invalid"
            return "valid"

        label = self.label_fn(window[-1])
        if label not in self.label_to_idx:
            raise ValueError(f"Unexpected label value: {label}")
        return label

    def _record_features(
        self,
        record: dict,
        history: Sequence[dict],
        position: int,
        window_length: int,
    ) -> Dict[str, float | str]:
        # Convert a single alarm event into a feature dictionary that captures
        # temporal context, sensor context, and cyclic time-of-day patterns.
        timestamp = parse_timestamp(record["timestamp"])
        previous = history[-1] if history else None
        prev_same_sensor = None
        for candidate in reversed(history):
            if candidate.get("sensor_id") == record.get("sensor_id"):
                prev_same_sensor = candidate
                break

        seconds_since_previous = 0.0
        seconds_since_same_sensor = 0.0
        if previous is not None:
            seconds_since_previous = max(0.0, (timestamp - parse_timestamp(previous["timestamp"])).total_seconds())
        if prev_same_sensor is not None:
            seconds_since_same_sensor = max(
                0.0,
                (timestamp - parse_timestamp(prev_same_sensor["timestamp"])).total_seconds(),
            )

        seconds_in_day = timestamp.hour * 3600 + timestamp.minute * 60 + timestamp.second
        day_fraction = seconds_in_day / 86400.0

        sensor_id = str(record.get("sensor_id", "UNKNOWN")).strip()
        sensor_type = str(record.get("sensor_type", "UNKNOWN")).strip().upper()
        layer = str(record.get("layer", "UNKNOWN")).strip().upper()
        sector = str(record.get("sector", "UNKNOWN")).strip().upper()
        zone = str(record.get("zone", "UNKNOWN")).strip().upper()
        actor = str(record.get("actor", "UNKNOWN")).strip().upper()
        recent_same_sensor = 0.0
        recent_same_zone = 0.0
        recent_sensor_density = 0.0
        recent_zone_density = 0.0
        if history:
            recent_same_sensor = sum(
                1 for prior in history if str(prior.get("sensor_id", "")).strip() == sensor_id
            ) / len(history)
            recent_same_zone = sum(
                1 for prior in history if str(prior.get("zone", "")).strip().upper() == zone
            ) / len(history)
            recent_sensor_density = sum(
                1 for prior in history if str(prior.get("sensor_id", "")).strip() == sensor_id
            ) / max(len(history), 1)
            recent_zone_density = sum(
                1 for prior in history if str(prior.get("zone", "")).strip().upper() == zone
            ) / max(len(history), 1)

        features: Dict[str, float | str] = {
            "sensor_id": sensor_id,
            "sensor_type": sensor_type,
            "layer": layer,
            "sector": sector,
            "zone": zone,
            "actor": actor,
            "hour_sin": math.sin(2.0 * math.pi * day_fraction),
            "hour_cos": math.cos(2.0 * math.pi * day_fraction),
            "weekday_sin": math.sin(2.0 * math.pi * (timestamp.weekday() / 7.0)),
            "weekday_cos": math.cos(2.0 * math.pi * (timestamp.weekday() / 7.0)),
            "is_weekend": 1.0 if timestamp.weekday() >= 5 else 0.0,
            "seconds_since_previous": min(seconds_since_previous, 7200.0) / 7200.0,
            "seconds_since_same_sensor": min(seconds_since_same_sensor, 7200.0) / 7200.0,
            "window_position": position / max(window_length - 1, 1),
            "window_position_sin": math.sin(2.0 * math.pi * position / max(window_length - 1, 1)),
            "window_position_cos": math.cos(2.0 * math.pi * position / max(window_length - 1, 1)),
            "same_sensor_as_prev": 1.0 if previous is not None and previous.get("sensor_id") == sensor_id else 0.0,
            "same_zone_as_prev": 1.0 if previous is not None and previous.get("zone") == zone else 0.0,
            "recent_same_sensor": recent_same_sensor,
            "recent_same_zone": recent_same_zone,
            "recent_sensor_density": recent_sensor_density,
            "recent_zone_density": recent_zone_density,
        }
        return features

    def _build_edge_index(self, window: Sequence[dict]) -> torch.Tensor:
        # Connect nearby events in time and also link events from the same sensor
        # or zone, providing graph structure that reflects temporal and spatial
        # relationships in the alarm stream.
        edges: List[Tuple[int, int]] = []
        timestamps = [parse_timestamp(record["timestamp"]) for record in window]

        for idx in range(len(window) - 1):
            delta_s = abs((timestamps[idx + 1] - timestamps[idx]).total_seconds())
            if delta_s <= self.max_time_gap_s:
                edges.append((idx, idx + 1))
                edges.append((idx + 1, idx))

        for idx in range(len(window)):
            left_bound = max(0, idx - self.window_size)
            current_sensor = str(window[idx].get("sensor_id", ""))
            current_zone = str(window[idx].get("zone", ""))
            for other in range(left_bound, idx):
                delta_s = abs((timestamps[idx] - timestamps[other]).total_seconds())
                if delta_s > self.max_time_gap_s:
                    continue
                if (
                    str(window[other].get("sensor_id", "")) == current_sensor
                    or str(window[other].get("zone", "")) == current_zone
                ):
                    edges.append((idx, other))
                    edges.append((other, idx))

        if not edges:
            return torch.arange(len(window), dtype=torch.long).unsqueeze(0).repeat(2, 1)

        edge_array = np.unique(np.asarray(edges, dtype=np.int64), axis=0)
        return torch.tensor(edge_array, dtype=torch.long).t().contiguous()

    def _window_to_graph(self, window: Sequence[dict]) -> Data:
        # Turn one time window into a PyG graph with node features, edge index,
        # and a supervised target label.
        node_dicts = []
        for idx, record in enumerate(window):
            history = window[max(0, idx - 8):idx]
            node_dicts.append(self._record_features(record, history, idx, len(window)))

        x = torch.tensor(self.vectorizer.transform(node_dicts), dtype=torch.float32)
        edge_index = self._build_edge_index(window)
        label_text = self._window_label(window)
        if label_text not in self.label_to_idx:
            raise ValueError(f"Unexpected label value: {label_text}")

        positions = torch.arange(len(window), dtype=torch.float32)
        if len(window) > 1:
            positions = positions / float(len(window) - 1)
        positions = positions.unsqueeze(1)

        data = Data(
            x=x,
            edge_index=edge_index,
            y=torch.tensor([self.label_to_idx[label_text]], dtype=torch.long),
        )
        data.position = positions
        data.label_text = label_text
        data.window_start = window[0]["timestamp"]
        data.window_end = window[-1]["timestamp"]
        data.window_label = label_text
        return data

    def build_graphs(self, records: Sequence[dict], fit_vectorizer: bool = False) -> List[Data]:
        # Slide across the time-ordered records and build one graph per window.
        records = list(records)
        if len(records) < self.window_size:
            return []

        windows = [
            records[start:start + self.window_size]
            for start in range(0, len(records) - self.window_size + 1, self.stride)
        ]

        feature_dicts: List[Dict[str, float | str]] = []
        for window in windows:
            for idx, record in enumerate(window):
                history = window[max(0, idx - 8):idx]
                feature_dicts.append(self._record_features(record, history, idx, len(window)))

        if fit_vectorizer:
            self.vectorizer.fit(feature_dicts)

        graphs: List[Data] = []
        for window in windows:
            try:
                graphs.append(self._window_to_graph(window))
            except Exception as exc:
                print(f"Warning: skipped window {window[0].get('timestamp')} -> {window[-1].get('timestamp')}: {exc}")
        return graphs


try:
    from sklearn.feature_extraction import DictVectorizer
except Exception:  # pragma: no cover
    class DictVectorizer:  # type: ignore[override]
        def __init__(self, sparse=False):
            self.sparse = sparse
            self.feature_names_ = []

        def fit(self, feature_dicts):
            keys = []
            for feature_dict in feature_dicts:
                for key in feature_dict:
                    if key not in keys:
                        keys.append(key)
            self.feature_names_ = keys
            return self

        def transform(self, feature_dicts):
            rows = []
            for feature_dict in feature_dicts:
                row = []
                for key in self.feature_names_:
                    value = feature_dict.get(key, 0.0)
                    if isinstance(value, (int, float)):
                        row.append(float(value))
                    else:
                        row.append(float(str(value) != ""))
                rows.append(row)
            return np.asarray(rows, dtype=float)
