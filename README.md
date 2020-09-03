## Description

According to the fusion method described in the Clownfish paper, this repository contains scripts to fuse remote and local softmax prediction results.

## Software Requirements
The code is tested with the following installed software packages.

1. Python 3.6+
2. Pytorch 1.4.0
3. Scikit-learn 0.23.0
5. Request 2.24.0 # This is needed to download the data from Google drive.

## Download Data for PKUMMD Dataset
The following command will download softmax scores for the testing set videos (testing set from the cross-subject evaluation scheme). The remote and local results are dumped using the fine-tuned models, `ResNext-101` and `Resnet-18`, respectively. The sliding window parameters are 16 (window size) and 4 (window stride). The `Siminet` model is trained on the features extracted by the `Resnet-18` model.
```shell
$ ./download_data.py
```

## How to run
1. Run with default parameters such as, `fix_ma` as a similarity method, and remote lag 1. The output is saved in the `fusion_fix_ma_*.log` file.
```shell
$ ./run.sh
```
2. An example run with different parameters, similarity method = `siminet`, non-delayed remote results (i.e. remote_lag = 0).
```shell
$ remote_lag=0 sim_method="siminet" siminet_path="./data/PKUMMD/model_ckpt/siminet/siminet_resnet-18_window_16_4_size_224_epoch_99.pth"  ./run.sh
```
    
For more options, please do check the [run.sh](./run.sh) and [opts.py](./opts.py) script file.

## Note
If you find any bug or issue in the code (or in the paper), please do let us know. Moreover, if you find this code or paper useful, then please do cite our work.
