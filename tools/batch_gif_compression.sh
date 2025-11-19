#!/bin/bash

# Folder with gifs, or current if none given
DIR="${1:-.}"

# Loop over all .gif files in the directory
for gif in "$DIR"/*.gif; do
    [ -f "$gif" ] || continue  # skip if no .gif files

    echo "Processing $gif..."

    # Create temporary palette file
    palette=$(mktemp --suffix=.png)

    # Generate palette
    ffmpeg -y -i "$gif" -vf "fps=10,scale=300:-1:flags=lanczos,palettegen" "$palette"

    # Prepare output filename
    base="${gif%.gif}"
    out_gif="${base}_compressed.gif"

    # Create optimized gif
    ffmpeg -y -i "$gif" -i "$palette" -lavfi "fps=10,scale=300:-1:flags=lanczos [x]; [x][1:v] paletteuse" "$out_gif"

    # Clean up
    rm "$palette"

    echo "Saved: $out_gif"
done
