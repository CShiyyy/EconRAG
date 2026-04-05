import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
DB_PATH: Path = Path(os.getenv("ECONRAG_DB_PATH", str(DATA_DIR / "econrag.db")))

VALID_UNIVERSES: tuple[str, ...] = ("sp500", "nasdaq100", "djia30")

UNIVERSE_URLS: dict[str, str] = {
    "sp500": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
    "nasdaq100": "https://en.wikipedia.org/wiki/Nasdaq-100",
    "djia30": "https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average",
}

# constraint_name -> (default_value, description)
DEFAULT_CONSTRAINTS: dict[str, tuple[float, str]] = {
    "cash_floor": (0.05, "Minimum portfolio fraction held in cash at all times."),
    "max_single_position": (0.15, "Maximum weight for any single ticker."),
    "max_sector_concentration": (0.35, "Maximum combined weight for all tickers in one sector."),
    "min_position_size": (0.02, "Below this weight, a position is not worth opening."),
}
