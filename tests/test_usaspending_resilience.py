from __future__ import annotations

import httpx
import pytest

from stock_trading.contracts import UsaSpendingClient
from stock_trading.live.run_current_usaspending_shadow import _possibly_modeled_recipient


def test_usaspending_client_retries_transient_server_error() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls < 3:
            return httpx.Response(500, request=request, json={"detail": "temporary"})
        return httpx.Response(200, request=request, json={"generated_unique_award_id": "award-1"})

    client = UsaSpendingClient(
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        retry_delays_seconds=(0.0, 0.0),
        sleep=lambda _: None,
    )
    raw = client.fetch_award("award-1")

    assert calls == 3
    assert b"award-1" in raw.content_bytes


def test_usaspending_client_fails_closed_after_bounded_retries() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(500, request=request, json={"detail": "persistent"})

    client = UsaSpendingClient(
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        retry_delays_seconds=(0.0, 0.0),
        sleep=lambda _: None,
    )

    with pytest.raises(httpx.HTTPStatusError):
        client.fetch_award("award-1")
    assert calls == 3


def test_prefilter_rejects_generic_single_token_overlap() -> None:
    name_index = {"NATIONAL GRID": ("cmp_national_grid",)}

    assert not _possibly_modeled_recipient(
        "National Industries for the Blind",
        name_index,
    )


def test_prefilter_keeps_distinctive_subsidiary_root() -> None:
    name_index = {"MICROSOFT": ("cmp_microsoft",)}

    assert _possibly_modeled_recipient("Microsoft Federal LLC", name_index)
