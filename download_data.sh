# Download scores dump and model checkpoints for PKUMMD datasets
#!/bin/bash
: ${data_dir:="./data/PKUMMD"}

function gd_download() {
    [ -e $2 ] || mkdir -p $2
    dst="$2/$3"
    echo "Downloading file ${dst}"
    python3 google_drive.py $1 "$dst"
    tar -xvf "$dst" -C $2 > /dev/null
    rm "$dst"
}

# class file and annotation file of PKUMMD dataset
dst_dir="${data_dir}"
file="splits.tar.gz"
gd_download 1m8tpx8iSqpzalIRg6t_wS9F-tjc7fdBo "$dst_dir" ${file}

# scores dump for the remote model
dst_dir="${data_dir}/scores_dump/resnext-101/sample_duration_16/image_size_224/window_stride_4/"
file="val.tar.gz"
gd_download 1b2l-WKDYOCXMR93ewFXxTYTu9Qjl8kYL "$dst_dir" ${file}

# scores dump for the remote model
dst_dir="${data_dir}/scores_dump/resnet-18/sample_duration_16/image_size_224/window_stride_4/"
gd_download 1d83tqG059k2hSMkXP2u4rAGXxyJ-ZF8d "$dst_dir" ${file}

# siminet model trained on local model
dst_dir="${data_dir}/model_ckpt/siminet/"
file="siminet_resnet-18_window_16_4_size_224_epoch_99.pth.tar.gz"
gd_download 1SfXvBX6WX7D0r94SwK4OXeY8iWCKInuZ "$dst_dir" ${file}


