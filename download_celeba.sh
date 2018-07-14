#!/bin/sh
# ===========================
# Usage: ./setup.sh (model|data)?

if wget --help | grep -q 'show-progress'; then
    WGET_FLAG="-q --show-progress"
else
    WGET_FLAG=""
fi

# create a tmp directory for the downloading data
TMP_DIR="./tmp_download"
mkdir -p "${TMP_DIR}"

#create the directory for the pre-trained model
MODEL_DIR="./pretrained_results"
mkdir -p "${MODEL_DIR}"

#create the directory for the dataset
DATA_DIR="./data"
mkdir -p "${DATA_DIR}"

# downloading model
download_model() 
{
    # directory for celeba model
    TMP_MODEL_TAR_BALL="${TMP_DIR}/celeba_pretrained_results.tar.gz"

    MODEL_URL="http://files.ytzhang.net/lmdis-rep/release-v1/celeba/celeba_pretrained_results.tar.gz"
    echo "Downloading pre-trained models ..."
    wget ${WGET_FLAG} "${MODEL_URL}" -O "${TMP_MODEL_TAR_BALL}"
    echo "Uncompressing pre-trained models ..."
    tar -xzf "${TMP_MODEL_TAR_BALL}" -C "${TMP_DIR}"

    # move model to default directories
    echo "Move pre-trained image network model to ${MODEL_DIR} ..."
    mv "${TMP_DIR}/celeba_pretrained_results/celeba_10" "${MODEL_DIR}/celeba_10"
    mv "${TMP_DIR}/celeba_pretrained_results/celeba_30" "${MODEL_DIR}/celeba_30"

}

# downloading data
download_data() 
{
    # directory for celeba data
    TMP_DATA_TAR_BALL="${DATA_DIR}/celeba_data.tar.gz"
    DATA_URL="http://files.ytzhang.net/lmdis-rep/release-v1/celeba/celeba_data.tar.gz"
    echo "Downloading data ..."
    wget ${WGET_FLAG} "${DATA_URL}" -O "${TMP_DATA_TAR_BALL}"
    echo "Uncompressing data ..."
    tar -xzf "${TMP_DATA_TAR_BALL}" -C "${DATA_DIR}"
    rm -rf "${TMP_DATA_TAR_BALL}"

    if [ ! -d "${DATA_DIR}/celeba_images" ];then
        echo "Warning! Please download CelebA data from http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html and have the folder in data/celeba_images ..."
    elif [ ! -d "${DATA_DIR}/celeba_images/Eval" ];then
        echo "Warning! Please download CelebA data from http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html and have the Eval folder in data/celeba_images/Eval ..."
    elif [ ! -d "${DATA_DIR}/celeba_images/Img" ];then
        echo "Warning! Please download CelebA data from http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html and have the Img folder in data/celeba_images/Img ..."
    fi

    if [ ! -d "${DATA_DIR}/celeba_images/Img/img_align_celeba_png" ];then
        echo "The CelebA official website provide images in jpg format in Img/img_align_celeba, if you want to do experiment on the CelebA png images, you can download them from http://files.ytzhang.net/lmdis-rep/release-v1/celeba/img_align_celeba_png.tar.gz and save the images in data/celeba_images/Img/img_align_celeba_png"
    fi
}

# default to download all
if [ $# -eq 0 ]; then
    download_model
    download_data
else
    case $1 in
        "model") download_model
            ;;
        "data") download_data
            ;;
        *) echo "Usage: ./setup.sh [OPTION]"
           echo ""
           echo "No option will download both model and data."
           echo ""
           echo "OPTION:\n\tmodel: only download the pre-trained models (.npy)"
           echo "\tdata: only download the data(.json)"
            ;;
    esac
fi

# clear the tmp files
rm -rf "${TMP_DIR}"
