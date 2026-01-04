from __future__ import annotations
from typing import Any, Dict, Optional
import requests

class TrafikverketClient:
    def __init__(self, api_key: str, base_url: str):
        self.api_key = api_key
        self.base_url = base_url

    def build_request(
        self,
        objecttype: str,
        schemaversion: str,
        namespace: Optional[str] = None,
        limit: int = 1000,
        filter_xml: str = "",
        include_fields_xml: str = "",
    ) -> str:

        # Only include namespace if provided
        namespace_attr = f' namespace="{namespace}"' if namespace else ""

        # Trafikverket expects <FILTER/> when empty.
        filter_block = "<FILTER/>" if not filter_xml.strip() else f"<FILTER>{filter_xml}</FILTER>"

        return f"""<REQUEST>
  <LOGIN authenticationkey="{self.api_key}" />
  <QUERY objecttype="{objecttype}"{namespace_attr} schemaversion="{schemaversion}" limit="{limit}">
    {filter_block}
    {include_fields_xml}
  </QUERY>
</REQUEST>
"""

    def post_xml(self, xml_body: str) -> Dict[str, Any]:
        r = requests.post(
            self.base_url,
            data=xml_body.encode("utf-8"),
            headers={
                "Content-Type": "text/xml; charset=utf-8",
                "Accept": "application/json",
            },
            timeout=60,
        )

        # If it fails, print the body — Trafikverket usually returns a helpful message.
        if not r.ok:
            raise requests.HTTPError(
                f"{r.status_code} {r.reason} from Trafikverket.\nResponse body:\n{r.text}",
                response=r,
            )

        return r.json()

class SMHIForecastClient:
    def __init__(self):
        self.session = requests.Session()

    def get_point_forecast(self, lat: float, lon: float) -> Dict[str, Any]:
        url = (
            "https://opendata-download.smhi.se/api/category/pmp3g/version/2/"
            f"geotype/point/lon/{lon}/lat/{lat}/data.json"
        )
        r = self.session.get(url, timeout=60)

        # If SMHI returns 4xx/5xx, surface the body
        if not r.ok:
            raise requests.HTTPError(
                f"SMHI error {r.status_code} for {url}\n"
                f"Content-Type: {r.headers.get('Content-Type')}\n"
                f"Body:\n{r.text[:1000]}"
            )

        # Guard against empty body / HTML
        content_type = (r.headers.get("Content-Type") or "").lower()
        if "json" not in content_type:
            raise ValueError(
                f"SMHI returned non-JSON content for {url}\n"
                f"Status: {r.status_code}\n"
                f"Content-Type: {r.headers.get('Content-Type')}\n"
                f"Body (first 1000 chars):\n{r.text[:1000]}"
            )

        if not r.text.strip():
            raise ValueError(
                f"SMHI returned empty body for {url}\n"
                f"Status: {r.status_code}\n"
                f"Content-Type: {r.headers.get('Content-Type')}"
            )

        return r.json()
