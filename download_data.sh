# Download scores dump and model checkpoints for PKUMMD datasets
#!/bin/bash
: ${data_dir:="./data/"}

function gd_download() {
    [ -e $2 ] || mkdir -p $2
    dst="$2/$3"
    echo "Downloading file ${dst}"
    python3 google_drive.py $1 "$dst"
    tar -xvf "$dst" -C $2 > /dev/null
    rm "$dst"
}

# PKUMMD data
dst_dir="${data_dir}/"
file="PKUMMD.tar.gz"
gd_download 1PtxcBRu0lkhud_Bq6qg4uSXw6ANJ1CvG "$dst_dir" ${file}
