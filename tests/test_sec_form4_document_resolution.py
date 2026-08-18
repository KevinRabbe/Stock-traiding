from __future__ import annotations

import httpx

from stock_trading.sec import SecClient


_CIK = "0001494259"
_ACCESSION = "0001494259-26-000034"


def _ownership_xml() -> bytes:
    return b"""<?xml version=\"1.0\"?>
<ownershipDocument>
  <documentType>4</documentType>
  <issuer><issuerCik>0001494259</issuerCik></issuer>
  <reportingOwner>
    <reportingOwnerId><rptOwnerCik>0002010096</rptOwnerCik></reportingOwnerId>
  </reportingOwner>
  <nonDerivativeTable />
</ownershipDocument>
"""


def test_form4_fetch_recovers_raw_xml_when_primary_is_html() -> None:
    requested: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested.append(str(request.url))
        path = request.url.path
        if path.endswith("/form4.html"):
            return httpx.Response(
                200,
                text="<html><body><br></body></html>",
                headers={"content-type": "text/html; charset=UTF-8"},
            )
        if path.endswith("/index.json"):
            return httpx.Response(
                200,
                json={
                    "directory": {
                        "item": [
                            {"name": "form4.html"},
                            {"name": "form4.xml"},
                            {"name": "index.xml"},
                        ]
                    }
                },
            )
        if path.endswith("/form4.xml"):
            return httpx.Response(
                200,
                content=_ownership_xml(),
                headers={"content-type": "application/xml"},
            )
        raise AssertionError(f"unexpected SEC request: {request.url}")

    transport = httpx.MockTransport(handler)
    with httpx.Client(transport=transport) as http:
        client = SecClient("test example@example.com", client=http)
        raw = client.fetch_filing_xml(_CIK, _ACCESSION, "form4.html")

    assert raw.source_record_id == _ACCESSION
    assert raw.content_type == "application/xml"
    assert raw.content == _ownership_xml()
    assert requested[-1].endswith("/form4.xml")


def test_form4_fetch_does_not_probe_directory_when_primary_is_valid_xml() -> None:
    requested: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested.append(str(request.url))
        assert request.url.path.endswith("/ownership.xml")
        return httpx.Response(
            200,
            content=_ownership_xml(),
            headers={"content-type": "application/xml"},
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as http:
        client = SecClient("test example@example.com", client=http)
        raw = client.fetch_filing_xml(_CIK, _ACCESSION, "ownership.xml")

    assert raw.content_type == "application/xml"
    assert len(requested) == 1


def test_form4_fetch_without_primary_discovers_xml_from_directory() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/index.json"):
            return httpx.Response(
                200,
                json={"directory": {"item": [{"name": "form4.xml"}]}},
            )
        if path.endswith("/form4.xml"):
            return httpx.Response(200, content=_ownership_xml())
        raise AssertionError(f"unexpected SEC request: {request.url}")

    with httpx.Client(transport=httpx.MockTransport(handler)) as http:
        client = SecClient("test example@example.com", client=http)
        raw = client.fetch_filing_xml(_CIK, _ACCESSION, None)

    assert raw.content_type == "application/xml"
    assert raw.content == _ownership_xml()


def test_form4_fetch_returns_actual_primary_type_when_no_valid_xml_exists() -> None:
    html = b"<html><body>rendered filing</body></html>"

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/form4.html"):
            return httpx.Response(
                200,
                content=html,
                headers={"content-type": "text/html; charset=UTF-8"},
            )
        if path.endswith("/index.json"):
            return httpx.Response(
                200,
                json={"directory": {"item": [{"name": "index.xml"}]}},
            )
        raise AssertionError(f"unexpected SEC request: {request.url}")

    with httpx.Client(transport=httpx.MockTransport(handler)) as http:
        client = SecClient("test example@example.com", client=http)
        raw = client.fetch_filing_xml(_CIK, _ACCESSION, "form4.html")

    assert raw.content == html
    assert raw.content_type == "text/html"
