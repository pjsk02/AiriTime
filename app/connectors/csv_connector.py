"""CSV/Excel-upload sales connector (PRD.md section 6.1 v1 connector).

Covers the "CSV/Excel upload" path for un-integrated venues: a restaurant
that has no live POS connector yet can still get item-level sales into the
feature store by uploading a CSV export.
"""

from pathlib import Path

import pandas as pd

from app.connectors.base import SalesConnector

# Accepted column-name aliases per normalized field, checked case-insensitively
# and in this priority order (first match wins). Real-world POS exports vary
# their column names, so a small, documented set of aliases is accepted
# instead of requiring one exact header row.
_COLUMN_ALIASES: dict[str, list[str]] = {
    "location": ["location", "store", "restaurant"],
    "date": ["date", "sale_date"],
    "item": ["item", "menu_item", "item_name"],
    "qty_sold": ["qty_sold", "qty", "quantity", "units_sold"],
}


class CSVConnector(SalesConnector):
    """Reads item-level sales from a CSV file into normalized rows.

    The CSV must have one column for each of `location`, `date`, `item`,
    and `qty_sold`. Column names are matched case-insensitively against a
    small set of common aliases (first match wins, in this priority order):

      - location: `location`, `store`, `restaurant`
      - date: `date`, `sale_date`
      - item: `item`, `menu_item`, `item_name`
      - qty_sold: `qty_sold`, `qty`, `quantity`, `units_sold`

    Any other columns in the CSV are ignored. Raises `ValueError` if any
    field has no matching column, or if the date column contains a value
    pandas cannot parse. A CSV with a header row but zero data rows
    returns an empty DataFrame with the normalized schema, not an error.
    """

    def __init__(self, path: str | Path) -> None:
        """Store the CSV path; the file is only read when `fetch()` runs."""
        self.path = Path(path)

    def fetch(self) -> pd.DataFrame:
        """Read and normalize the CSV at `self.path`.

        Returns:
            A DataFrame with columns `location` (str), `date`
            (datetime.date), `item` (str), and `qty_sold` (float).

        Raises:
            ValueError: a required field has no matching column, or the
                date column cannot be parsed.
        """
        raw = pd.read_csv(self.path)
        column_map = self._resolve_columns(raw.columns)

        normalized = pd.DataFrame(
            {field: raw[column_map[field]] for field in _COLUMN_ALIASES}
        )

        if normalized.empty:
            return normalized.assign(
                location=pd.Series(dtype="object"),
                date=pd.Series(dtype="object"),
                item=pd.Series(dtype="object"),
                qty_sold=pd.Series(dtype="float64"),
            )

        try:
            parsed_dates = pd.to_datetime(normalized["date"])
        except (ValueError, TypeError) as exc:
            raise ValueError(f"could not parse 'date' column in {self.path}: {exc}") from exc
        normalized["date"] = parsed_dates.dt.date

        try:
            normalized["qty_sold"] = pd.to_numeric(normalized["qty_sold"]).astype("float64")
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"could not parse quantity column in {self.path} as numeric: {exc}"
            ) from exc

        normalized["location"] = normalized["location"].astype(str)
        normalized["item"] = normalized["item"].astype(str)

        return normalized.reset_index(drop=True)

    @staticmethod
    def _resolve_columns(columns: pd.Index) -> dict[str, str]:
        """Map each normalized field name to the matching CSV column name.

        Raises:
            ValueError: if a field has no matching alias among `columns`.
        """
        lookup = {str(col).strip().lower(): col for col in columns}
        resolved: dict[str, str] = {}
        for field, aliases in _COLUMN_ALIASES.items():
            match = next((lookup[alias] for alias in aliases if alias in lookup), None)
            if match is None:
                raise ValueError(
                    f"CSV is missing a column for '{field}' "
                    f"(accepted names: {', '.join(aliases)})"
                )
            resolved[field] = match
        return resolved
