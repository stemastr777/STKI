import csv
from typing import Dict, Any


def query_mountain_location(mountain: str) -> None | Dict[str, Any]:
    with open("../data/location.csv", newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row["name"].lower() == mountain.lower():
                return row
    return None
