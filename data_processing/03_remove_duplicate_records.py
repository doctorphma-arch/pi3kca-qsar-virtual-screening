#!/usr/bin/env python3
"""Remove duplicate molecule records from the curated ChEMBL export.

Reads the preprocessed bioactivity table, keeps the first occurrence of each
molecule_chembl_id, and exports both the deduplicated table and the removed
records for auditing.

7,029 raw ChEMBL entries -> 6,857 after dropping records without a standard
value (01_data_collection.ipynb) -> 6,059 unique compounds here (798 removed).
"""

import os

INPUT_FILE = "../data/PIK3CA_preprocessed.csv"
OUTPUT_FILE = "../data/PIK3CA_deduplicated.csv"
REMOVED_FILE = "../data/PIK3CA_dedup_removed_records.csv"
KEY_COLUMN = 0  # molecule_chembl_id


def remove_duplicate_rows(input_filepath, output_filepath, removed_filepath,
                          column_index=0, delimiter=','):
    seen_values = set()
    duplicate_count = 0
    written_count = 0
    removed_count = 0
    header_line = None

    with open(input_filepath, 'r', encoding='utf-8') as infile, \
         open(output_filepath, 'w', encoding='utf-8') as outfile, \
         open(removed_filepath, 'w', encoding='utf-8') as removedfile:
        try:
            header_line = next(infile)
            outfile.write(header_line)
            removedfile.write(header_line)
        except StopIteration:
            pass

        for line in infile:
            parts = line.strip().split(delimiter)
            if not parts or column_index < 0 or column_index >= len(parts):
                outfile.write(line)
                written_count += 1
                continue

            key_value = parts[column_index]
            if key_value not in seen_values:
                outfile.write(line)
                seen_values.add(key_value)
                written_count += 1
            else:
                removedfile.write(line)
                removed_count += 1
                duplicate_count += 1

    return written_count, duplicate_count, removed_count


if __name__ == "__main__":
    w, d, r = remove_duplicate_rows(INPUT_FILE, OUTPUT_FILE, REMOVED_FILE, KEY_COLUMN)
    print(f"written: {w} rows -> {OUTPUT_FILE}")
    print(f"removed: {r} duplicate rows -> {REMOVED_FILE}")
