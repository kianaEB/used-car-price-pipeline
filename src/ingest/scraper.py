"""OPTIONAL polite cars.com scraper to refresh the local dataset.

NOT part of `make all`, tests, or CI (see SPEC 3). Off by default. If you implement it:
  - read and respect robots.txt and the site's Terms of Service
  - rate-limit (sleep between requests) and set a descriptive User-Agent
  - write output to data/raw/ in the canonical schema, then let the DQ layer validate it

Left as a documented stub on purpose -- do NOT make the pipeline depend on live scraping.
(The prototype's brittle, unthrottled scraping is one of the things we're moving away from.)
"""

from __future__ import annotations

import pandas as pd


def scrape(max_pages: int = 5, delay_seconds: float = 2.0) -> pd.DataFrame:
    """Scrape listings into a canonical-schema DataFrame. TODO (optional stretch goal)."""
    raise NotImplementedError(
        "Optional stretch goal -- respect robots.txt/ToS and rate-limit. See SPEC 3."
    )
