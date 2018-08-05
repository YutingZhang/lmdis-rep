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
    # directory for aflw model
    TMP_MODEL_TAR_BALL="${TMP_DIR}/cat_pretrained_results.tar.gz"

    MODEL_URL="http://files.ytzhang.net/lmdis-rep/release-v1/cat/cat_pretrained_results.tar.gz"
    echo "Downloading pre-trained models ..."
    wget ${WGET_FLAG} "${MODEL_URL}" -O "${TMP_MODEL_TAR_BALL}"
    echo "Uncompressing pre-trained models ..."
    tar -xzf "${TMP_MODEL_TAR_BALL}" -C "${TMP_DIR}"

    # move model to default directories
    echo "Move pre-trained image network model to ${MODEL_DIR} ..."
    mv "${TMP_DIR}/cat_pretrained_results/cat_10" "${MODEL_DIR}/cat_10"
    mv "${TMP_DIR}/cat_pretrained_results/cat_20" "${MODEL_DIR}/cat_20"

}

# downloading data
download_data() 
{
    # directory for cat data
    TMP_DATA_TAR_BALL="${DATA_DIR}/cat_data.tar.gz"
    DATA_URL="http://files.ytzhang.net/lmdis-rep/release-v1/cat/cat_data.tar.gz"
    echo "Downloading data ..."
    wget ${WGET_FLAG} "${DATA_URL}" -O "${TMP_DATA_TAR_BALL}"
    echo "Uncompressing data ..."
    tar -xzf "${TMP_DATA_TAR_BALL}" -C "${DATA_DIR}"
    rm -rf "${TMP_DATA_TAR_BALL}"

    TMP_IMAGE_TAR_BALL="${DATA_DIR}/cat_images.tar.gz"
    IMAGE_URL="http://files.ytzhang.net/lmdis-rep/release-v1/cat/cat_images.tar.gz"
    echo "Downloading images ..."
    wget ${WGET_FLAG} "${IMAGE_URL}" -O "${TMP_IMAGE_TAR_BALL}"
    echo "Uncompressing images ..."
    tar -xzf "${TMP_IMAGE_TAR_BALL}" -C "${DATA_DIR}" 
    rm -rf "${TMP_IMAGE_TAR_BALL}"
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
