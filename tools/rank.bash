for f in *.wav; do 
    duration=$(ffprobe -i "$f" -show_entries format=duration -v quiet -of csv="p=0"); 
    echo "$duration $f"; 
done | sort -nr
