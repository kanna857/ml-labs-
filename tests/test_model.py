from pathlib import Path
import subprocess
import sys

def test_serialization_script():
    result = subprocess.run(
        [sys.executable, "experiments/exp03_serialization.py"],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0
    assert "Predicted price: 80.0 lakhs" in result.stdout
