from __future__ import annotations

import httpx
import pytest

from stock_trading.core import Source
from stock_trading.sec import SecClient


def test_quarterly_archive_urls_include_both_official_sec_locations() -> None:
    urls = SecClient.quarterly_archive_urls(2026, 2)
    assert urls == (
        "https://www.sec.gov/files/structureddata/data/"
        "insider-transactions-data-sets/2026q2_form345.zip",
        "https://www.sec.gov/files/datastandardsinnovation/data/"
        "insider-transactions-data-sets/2026q2_form345.zip",
    )
    assert SecClient.quarterly_archive_url(2012, 1) == (
        "https://www.sec.gov/files/structureddata/data/"
        "insider-transactions-data-sets/2012q1_form345.zip"
    )


def test_quarterly_archive_falls_back_only_after_404() -> None:
    requested: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        requested.append(url)
        if "/files/structureddata/" in url:
            return httpx.Response(404, request=request)
        if "/files/datastandardsinnovation/" in url:
            return httpx.Response(200, content=b"zip-bytes", request=request)
        raise AssertionError(f"unexpected URL: {url}")

    transport = httpx.MockTransport(handler)
    with httpx.Client(transport=transport) as http_client:
        sec = SecClient("Stock-traiding test@example.com", client=http_client)
        raw = sec.fetch_quarterly_archive(2026, 2)

    assert raw.source is Source.SEC_QUARTERLY
    assert raw.source_record_id == "2026Q2"
    assert raw.content == b"zip-bytes"
    assert len(requested) == 2
    assert "/files/structureddata/" in requested[0]
    assert "/files/datastandardsinnovation/" in requested[1]


def test_quarterly_archive_does_not_mask_non_404_http_failure() -> None:
    requested: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested.append(str(request.url))
        return httpx.Response(403, request=request)

    transport = httpx.MockTransport(handler)
    with httpx.Client(transport=transport) as http_client:
        sec = SecClient("Stock-traiding test@example.com", client=http_client)
        with pytest.raises(httpx.HTTPStatusError):
            sec.fetch_quarterly_archive(2026, 2)

    assert len(requested) == 1
