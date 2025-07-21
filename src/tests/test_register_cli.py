import json
from pathlib import Path

from src.fingerprint.presentation.cli.register_person import load_minutiae_from_json
from src.fingerprint.domain.entities.minutiae import Minutiae


def test_load_minutiae_from_json(tmp_path: Path) -> None:
    data = [{"type": "termination", "position": [1, 2], "orientation": 0.1}]
    json_path = tmp_path / "minutiae.json"
    json_path.write_text(json.dumps(data))

    result = load_minutiae_from_json(str(json_path))
    assert result == [Minutiae("termination", (1, 2), 0.1)]

