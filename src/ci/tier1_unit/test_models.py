from pathlib import Path

import orjson
import pytest

from pypff.models import DataConfig, ObsConfig, PhBaselineConfig, QuaboConfig

EXAMPLE_DATA_DIR = Path(__file__).parents[3] / "example" / "example-data"

def test_data_config_parsing() -> None:
    data_file = EXAMPLE_DATA_DIR / "data_config.json"
    if not data_file.exists():
        pytest.skip("Example data not found.")
    
    with open(data_file, 'rb') as f:
        config_dict = orjson.loads(f.read())
    
    config = DataConfig(**config_dict)
    assert config.run_type == "pe-steps-ph8"
    assert config.pulse_height is not None
    assert config.pulse_height.pe_threshold == -1 # Old format used -1, new models might restrict ge=2.0
    # Wait, I removed the ge=2.0 restriction in my implementation to be permissive with legacy data
    assert config.pulse_height.pe_threshold == -1

def test_obs_config_parsing() -> None:
    obs_file = EXAMPLE_DATA_DIR / "obs_config.json"
    if not obs_file.exists():
        pytest.skip("Example data not found.")
        
    with open(obs_file, 'rb') as f:
        config_dict = orjson.loads(f.read())
    
    config = ObsConfig(**config_dict)
    assert config.name == "UCB_lab"
    assert len(config.domes) == 1
    assert str(config.domes[0].modules[0].ip_addr) == "192.168.3.248"

def test_quabo_config_parsing() -> None:
    # Test complex CSV string to list conversion
    q_file = EXAMPLE_DATA_DIR / "quabo_config_192.168.3.248.json"
    if not q_file.exists():
        pytest.skip("Example data not found.")
        
    with open(q_file, 'rb') as f:
        config_dict = orjson.loads(f.read())
    
    config = QuaboConfig(**config_dict)
    # Check if CSV string "1,1,1,1" became [1, 1, 1, 1]
    assert hasattr(config, "OTABG_ON")
    assert config.OTABG_ON == [1, 1, 1, 1]
    # Check hex string "0xffffffff" became int
    assert hasattr(config, "CHANMASK_0")
    assert config.CHANMASK_0 == 0xffffffff

def test_ph_baseline_parsing() -> None:
    ph_file = EXAMPLE_DATA_DIR / "quabo_ph_baseline.json"
    if not ph_file.exists():
        pytest.skip("Example data not found.")
        
    with open(ph_file, 'rb') as f:
        config_dict = orjson.loads(f.read())
    
    config = PhBaselineConfig(**config_dict)
    assert len(config.quabos) == 1
    assert len(config.quabos[0].coefs) == 256
