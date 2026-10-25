import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))  # for fake_openai import
sys.path.insert(0, str(Path(__file__).parent.parent))
