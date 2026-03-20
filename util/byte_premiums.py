"""
This is code from the Goldfish paper
https://github.com/tylerachang/goldfish/
"""

BYTE_PREMIUMS_PATH = './util/byte_premiums.tsv'

with open(BYTE_PREMIUMS_PATH, 'r', encoding='utf-8') as f:
    byte_premiums = f.read()

byte_premiums = byte_premiums.strip().split('\n')[1:]  # Skip header.
byte_premiums = [line.split('\t') for line in byte_premiums]
byte_premiums = [
    (split_line[0], float(split_line[1]))
    for split_line in byte_premiums
]
byte_premiums = dict(byte_premiums)
