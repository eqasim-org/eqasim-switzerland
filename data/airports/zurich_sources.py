"""Re-extract reviewed monthly Zurich statistics without OCR (maintenance only)."""

import argparse
import csv
import io
import json
from pathlib import Path

from data.airports.common import atomic_write, file_digest

MONTHS = ('JAN', 'FEB', 'MRZ', 'APR', 'MAI', 'JUN', 'JUL', 'AUG', 'SEPT', 'OKT', 'NOV', 'DEZ')


def extract_monthly(pdf_file, year):
    try:
        from pypdf import PdfReader
    except ImportError as error:
        raise ImportError('Monthly PDF maintenance requires the optional pypdf package; normal stage execution does not.') from error
    reader = PdfReader(pdf_file)
    tables = []
    for page_number in (27, 25):
        rows = {}
        text = reader.pages[page_number - 1].extract_text(extraction_mode='layout')
        for line in text.splitlines():
            parts = line.split()
            if parts and parts[0] in MONTHS and parts[0] not in rows:
                rows[parts[0]] = [int(value.replace('’', '').replace("'", '')) for value in parts[1:5]]
        if set(rows) != set(MONTHS):
            raise ValueError(f'PDF layout changed: review the monthly table on page {page_number}')
        tables.append(rows)
    local_rows, directions = tables
    output = []
    for month, name in enumerate(MONTHS, 1):
        local, transfer, transit, total = local_rows[name]
        arrivals, departures, direction_transit, direction_total = directions[name]
        if (arrivals + departures - transfer != local or transit != direction_transit
                or total != local + transfer + transit or total != direction_total):
            raise ValueError(f'PDF extraction does not reconcile for {name}; review tables')
        output.append([year, month, local, transfer, transit, arrivals, departures])
    return output


def refresh_monthly(folder):
    folder = Path(folder)
    metadata_file = folder / 'data_from_websites/zurich_airport.json'
    metadata = json.loads(metadata_file.read_text(encoding='utf-8'))
    source = next(s for s in metadata['sources'] if s['id'] == 'statistics_2025')
    pdf = folder / source['local_file']
    rows = extract_monthly(pdf, metadata['traffic_year'])
    if sum(row[2] for row in rows) != metadata['annual_local_passengers']:
        raise ValueError('Extracted year differs from reviewed annual total; review annual metadata first')
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(['year', 'month', 'local_passengers', 'transfer_movements', 'transit_passengers',
                     'terminal_arrivals', 'terminal_departures'])
    writer.writerows(rows)
    contents = buffer.getvalue()
    name = metadata['monthly_file']
    atomic_write(folder / name, lambda p: p.write_text(contents, encoding='utf-8', newline=''))
    metadata['reviewed_inputs'][name] = file_digest(folder / name)
    source['sha256'] = file_digest(pdf)
    atomic_write(metadata_file, lambda p: p.write_text(json.dumps(metadata, indent=2, ensure_ascii=False)+'\n', encoding='utf-8'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract monthly Zurich tables from the archived statistics PDF after reviewing annual totals.')
    parser.add_argument('--folder', type=Path, required=True)
    args = parser.parse_args()
    refresh_monthly(args.folder)
