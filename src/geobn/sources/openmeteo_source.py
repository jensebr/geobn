from __future__ import annotations

from datetime import date as _date

import requests

from ._point_sampling import _PointSamplingSource

_ARCHIVE_API = "https://archive-api.open-meteo.com/v1/archive"
_FORECAST_API = "https://api.open-meteo.com/v1/forecast"


class OpenMeteoSource(_PointSamplingSource):
    """Fetch a daily weather variable from the Open-Meteo API.

    The bounding box is derived at *fetch()* time from the reference grid,
    so no explicit coordinates are required in the constructor.

    A regular grid of *sample_points* × *sample_points* lat/lon points is
    queried and the resulting values are assembled into a coarse raster
    (EPSG:4326).  The alignment step in *GeoBayesianNetwork.infer()* then
    bilinearly resamples this to the reference grid resolution.

    Parameters
    ----------
    variable:
        Open-Meteo daily variable name, e.g. "precipitation_sum",
        "temperature_2m_mean".
    date:
        ISO date string "YYYY-MM-DD".  Defaults to today.
    sample_points:
        Number of sample points along each axis (total = sample_points²).
        Defaults to 5 (25 API calls).
    timeout:
        HTTP request timeout in seconds.
    """

    def __init__(
        self,
        variable: str,
        date: str | None = None,
        sample_points: int = 5,
        timeout: int = 10,
    ) -> None:
        super().__init__(sample_points=sample_points, timeout=timeout)
        self._variable = variable
        self._date = date or str(_date.today())

    def _query_point(self, lat: float, lon: float) -> float:
        api = _ARCHIVE_API
        params = {
            "latitude": lat,
            "longitude": lon,
            "daily": self._variable,
            "timezone": "UTC",
            "start_date": self._date,
            "end_date": self._date,
        }
        resp = requests.get(api, params=params, timeout=self._timeout)
        resp.raise_for_status()
        data = resp.json()

        daily = data.get("daily", {})
        values = daily.get(self._variable)
        if not values:
            raise ValueError(
                f"Open-Meteo returned no data for variable '{self._variable}' "
                f"on {self._date} at lat={lat:.4f}, lon={lon:.4f}.  "
                f"Check the variable name and date range."
            )
        return float(values[0]) if values[0] is not None else float("nan")
