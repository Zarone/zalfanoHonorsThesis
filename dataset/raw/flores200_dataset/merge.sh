#!/usr/bin/env bash

DIR1="dev"
DIR2="devtest"
OUT="merged"

mkdir -p "$OUT"

# Get all unique base filenames
for base in $(ls "$DIR1" "$DIR2" 2>/dev/null | sed 's/\.[^.]*$//' | sort -u); do
    # out_file="$OUT/$base.txt"

    base_lc=$(echo "$base" | tr '[:upper:]' '[:lower:]')
    out_file="$OUT/$base_lc.txt"
    : > "$out_file"


    for f in "$DIR1/$base".* "$DIR2/$base".*; do
        [ -f "$f" ] && cat "$f" >> "$out_file"
    done


done



