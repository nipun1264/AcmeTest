import copy
import json

import pytest

from pitch_engine.config import AppConfig, ConfigError, load_config

VALID = {
    "video": {"path": "synthetic_pitch_feed.mp4", "target_fps": 30},
    "detector": {
        "kind": "sam_mask_v1",
        "sport": "football",
        "min_area": 1000,
        "confidence_threshold": 0.05,
    },
    "sampling": {"frame_stride": 1},
    "crop_search": {"aspect_ratio": "16:9", "padding_px": 20},
    "reporting": {"base_url": "http://localhost:5000", "enabled": True},
    "debug_mode": True,
}


def write_config(tmp_path, data):
    path = tmp_path / "config.json"
    path.write_text(data if isinstance(data, str) else json.dumps(data), encoding="utf-8")
    return path


def mutated(section, key, value):
    data = copy.deepcopy(VALID)
    if value is ...:
        del data[section][key]
    else:
        data[section][key] = value
    return data


def test_valid_config_loads(tmp_path):
    config = load_config(write_config(tmp_path, VALID))
    assert isinstance(config, AppConfig)
    assert config.detector.min_area == 1000
    assert config.sampling.frame_stride == 1


@pytest.mark.parametrize(
    "data, expected_location",
    [
        (mutated("video", "path", ...), "video.path"),
        (mutated("video", "target_fps", 0), "video.target_fps"),
        (mutated("video", "target_fps", "30"), "video.target_fps"),
        (mutated("detector", "kind", "unknown_model"), "detector.kind"),
        (mutated("detector", "min_area", -5), "detector.min_area"),
        (mutated("detector", "confidence_threshold", 1.5), "detector.confidence_threshold"),
        (mutated("sampling", "frame_stride", 0), "sampling.frame_stride"),
        (mutated("crop_search", "aspect_ratio", "widescreen"), "crop_search.aspect_ratio"),
        (mutated("crop_search", "padding_px", -1), "crop_search.padding_px"),
        (mutated("reporting", "base_url", ...), "reporting.base_url"),
        (mutated("reporting", "base_url", ""), "reporting.base_url"),
    ],
)
def test_bad_config_fails_at_load_time_and_names_the_field(tmp_path, data, expected_location):
    with pytest.raises(ConfigError) as excinfo:
        load_config(write_config(tmp_path, data))
    assert expected_location in str(excinfo.value)


def test_wrong_type_top_level_field_is_rejected(tmp_path):
    data = copy.deepcopy(VALID)
    data["debug_mode"] = "yes"
    with pytest.raises(ConfigError, match="debug_mode"):
        load_config(write_config(tmp_path, data))


def test_unknown_key_is_rejected(tmp_path):
    data = copy.deepcopy(VALID)
    data["detector"]["typ"] = "sam_mask_v1"  # typo of "kind" left dangling
    with pytest.raises(ConfigError, match="detector.typ"):
        load_config(write_config(tmp_path, data))


def test_unparsable_json_is_rejected(tmp_path):
    with pytest.raises(ConfigError, match="not valid JSON"):
        load_config(write_config(tmp_path, "{ not json"))


def test_missing_file_is_rejected(tmp_path):
    with pytest.raises(ConfigError, match="Cannot read config file"):
        load_config(tmp_path / "does_not_exist.json")


def test_all_problems_are_reported_together(tmp_path):
    data = mutated("detector", "min_area", -5)
    data["sampling"]["frame_stride"] = 0
    with pytest.raises(ConfigError) as excinfo:
        load_config(write_config(tmp_path, data))
    message = str(excinfo.value)
    assert "detector.min_area" in message and "sampling.frame_stride" in message
