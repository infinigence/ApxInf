"""Calculate a bounded expense selection from fixed local files."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import io
import json
import os
from pathlib import Path
import re
import stat
import sys


SAFE_INTEGER = 2 ** 53 - 1
COLUMNS = ["id", "date", "description", "category", "currency", "amount", "status"]


@dataclass(frozen=True)
class Selection:
    currency: str
    status: str
    categories: tuple[str, ...]

    def document(self):
        return {"currency": self.currency, "status": self.status, "categories": list(self.categories)}


@dataclass(frozen=True)
class Row:
    identifier: str
    category: str
    currency: str
    cents: int
    status: str


def read_regular(path, limit):
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
            raise ValueError("The input is not a bounded regular file.")
        with os.fdopen(descriptor, "rb", closefd=False) as source:
            data = source.read(limit + 1)
        if len(data) > limit:
            raise ValueError("The input exceeds its byte limit.")
        return data
    finally:
        os.close(descriptor)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("The plan contains a duplicate field.")
        result[key] = value
    return result


def reject_constant(value):
    raise ValueError("The plan contains a nonfinite number.")


def parse_selection(data):
    if len(data) > 4096:
        raise ValueError("The plan exceeds its byte limit.")
    value = json.loads(data.decode("utf-8"), object_pairs_hook=unique_object, parse_constant=reject_constant)
    if type(value) is not dict or value.keys() != {"currency", "status", "categories"}:
        raise ValueError("The plan must contain exactly currency, status, and categories.")
    categories = value["categories"]
    if (not all(type(value[key]) is str and value[key].strip() for key in ("currency", "status")) or
            type(categories) is not list or not 1 <= len(categories) <= 16 or
            not all(type(item) is str and item.strip() for item in categories) or
            len(set(categories)) != len(categories)):
        raise ValueError("The plan contains an invalid selection.")
    return Selection(value["currency"], value["status"], tuple(categories))


def parse_cents(amount):
    if re.fullmatch(r"[+-]?[0-9]{1,12}\.[0-9]{2}", amount) is None:
        raise ValueError("An amount must have at most 12 integer digits and exactly two fractional digits.")
    sign = -1 if amount.startswith("-") else 1
    whole, fraction = amount.lstrip("+-").split(".")
    return checked_integer(sign * (int(whole) * 100 + int(fraction)))


def checked_integer(value):
    if abs(value) > SAFE_INTEGER:
        raise ValueError("A money integer exceeds the JSON safe-integer range.")
    return value


def parse_rows(data):
    if len(data) > 65536:
        raise ValueError("The CSV exceeds its byte limit.")
    reader = csv.reader(io.StringIO(data.decode("utf-8"), newline=""), strict=True)
    if next(reader, None) != COLUMNS:
        raise ValueError("The CSV columns differ from the required columns.")
    rows, identifiers = [], set()
    for fields in reader:
        if len(rows) >= 100 or len(fields) != len(COLUMNS) or not all(item.strip() for item in fields):
            raise ValueError("The CSV contains an invalid row or too many rows.")
        identifier, date, description, category, currency, amount, status = fields
        if identifier in identifiers:
            raise ValueError("The CSV contains a duplicate transaction ID.")
        identifiers.add(identifier)
        rows.append(Row(identifier, category, currency, parse_cents(amount), status))
    return rows


def calculate(rows, selection):
    totals = dict.fromkeys(selection.categories, 0)
    included, total = [], 0
    for row in rows:
        if row.currency != selection.currency or row.status != selection.status or row.category not in totals:
            continue
        included.append(row.identifier)
        totals[row.category] = checked_integer(totals[row.category] + row.cents)
        total = checked_integer(total + row.cents)
    return {"currency": selection.currency, "included_rows": included,
            "category_totals_cents": totals, "total_cents": total}


def main():
    try:
        if len(sys.argv) != 1:
            raise ValueError("The calculator accepts no arguments.")
        source = read_regular(Path("transactions.csv"), 65536)
        rules = read_regular(Path("expense-rules.txt"), 8192)
        plan = read_regular(Path("calculation-plan.json"), 4096)
        calculator = read_regular(Path("expense_calculator.py"), 32768)
        selection = parse_selection(plan)
        rows = parse_rows(source)
        receipt = {"schema": "apxinf-expense-calculation-v1",
                   "source_sha256": hashlib.sha256(source).hexdigest(),
                   "rules_sha256": hashlib.sha256(rules).hexdigest(),
                   "plan_sha256": hashlib.sha256(plan).hexdigest(),
                   "calculator_sha256": hashlib.sha256(calculator).hexdigest(),
                   "selection": selection.document(), "row_count": len(rows),
                   "summary": calculate(rows, selection)}
        output = json.dumps(receipt, ensure_ascii=False) + "\n"
        if len(output.encode("utf-8")) > 16384:
            raise ValueError("The receipt exceeds its byte limit.")
        sys.stdout.write(output)
        return 0
    except (OSError, ValueError, UnicodeError, csv.Error, RecursionError) as error:
        print(str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
