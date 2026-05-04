import re
from typing import Optional, Tuple


PASAL_REGEX = re.compile(r"pasal\s+(\d+)", re.IGNORECASE)
AYAT_REGEX = re.compile(r"ayat\s+(\d+)", re.IGNORECASE)


def parse_legal_query(query: str) -> Tuple[Optional[int], Optional[int]]:
    """
    Extract pasal & ayat dari query user.

    Contoh:
    "pasal 36 ayat 2" → (36, 2)
    "pasal 10" → (10, None)
    "ayat 3" → (None, 3)
    """

    pasal_match = PASAL_REGEX.search(query)
    ayat_match = AYAT_REGEX.search(query)

    pasal = int(pasal_match.group(1)) if pasal_match else None
    ayat = int(ayat_match.group(1)) if ayat_match else None

    return pasal, ayat