from dataclasses import dataclass
from datetime import date, datetime

from stock_trading.core import as_utc


@dataclass(frozen=True, slots=True)
class OwnershipFiling:
    cik: str
    accession_number: str
    form: str
    filing_date: date
    report_date: date | None
    accepted_at: datetime
    primary_document: str

    @property
    def is_amendment(self) -> bool:
        return self.form == "4/A"


class SubmissionsParser:
    """Parse ownership filings from the SEC submissions JSON response."""

    def recent_form4_filings(self, payload: dict) -> tuple[OwnershipFiling, ...]:
        cik = str(payload.get("cik") or "").strip()
        if not cik:
            raise ValueError("SEC submissions payload is missing cik")
        cik = cik.zfill(10)

        recent = payload.get("filings", {}).get("recent", {})
        if not isinstance(recent, dict):
            raise ValueError("SEC submissions payload is missing filings.recent")

        forms = recent.get("form", [])
        accessions = recent.get("accessionNumber", [])
        filing_dates = recent.get("filingDate", [])
        report_dates = recent.get("reportDate", [])
        accepted_times = recent.get("acceptanceDateTime", [])
        primary_documents = recent.get("primaryDocument", [])

        lengths = {
            len(forms),
            len(accessions),
            len(filing_dates),
            len(report_dates),
            len(accepted_times),
            len(primary_documents),
        }
        if len(lengths) != 1:
            raise ValueError("SEC filings.recent columns have inconsistent lengths")

        results: list[OwnershipFiling] = []
        for index, form_value in enumerate(forms):
            form = str(form_value or "").strip().upper()
            if form not in {"4", "4/A"}:
                continue

            accession = str(accessions[index] or "").strip()
            filing_date_text = str(filing_dates[index] or "").strip()
            accepted_text = str(accepted_times[index] or "").strip()
            primary_document = str(primary_documents[index] or "").strip()
            if not all((accession, filing_date_text, accepted_text, primary_document)):
                continue

            report_date_text = str(report_dates[index] or "").strip()
            results.append(
                OwnershipFiling(
                    cik=cik,
                    accession_number=accession,
                    form=form,
                    filing_date=date.fromisoformat(filing_date_text),
                    report_date=(
                        date.fromisoformat(report_date_text) if report_date_text else None
                    ),
                    accepted_at=parse_sec_acceptance_time(accepted_text),
                    primary_document=primary_document,
                )
            )

        return tuple(results)


def parse_sec_acceptance_time(value: str) -> datetime:
    """Parse SEC acceptanceDateTime into an aware UTC datetime."""

    normalized = value.strip()
    if not normalized:
        raise ValueError("SEC acceptanceDateTime is empty")

    if normalized.endswith("Z"):
        parsed = datetime.fromisoformat(normalized[:-1] + "+00:00")
    else:
        parsed = datetime.fromisoformat(normalized)
        if parsed.tzinfo is None:
            # Do not guess a source timezone at this boundary. If the SEC ever
            # changes serialization, ingestion must adapt explicitly rather
            # than silently creating a point-in-time error.
            raise ValueError("SEC acceptanceDateTime must include a timezone")

    return as_utc(parsed)
