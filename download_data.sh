DEPTH_DATA_URL="https://www.dropbox.com/s/qtab28cauzalqi7/depth_data.tar.gz?dl=1"
DATA_EXTRACT_DIR="./data"

PRETRAINED_URL="https://www.dropbox.com/s/356r36lfpyzhcht/pretrained_models.tar.gz?dl=1"
PRETRAINED_EXTRACT_DIR="./"

wget -c $DEPTH_DATA_URL -O - | tar -xz -C $DATA_EXTRACT_DIR

mkdir $PRETRAINED_DIR
wget -c $PRETRAINED_URL -O - | tar -xz -C $PRETRAINED_EXTRACT_DIR


# Loading Partial noisy
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1-3jIt7jottf3gG2-YfyA02T4rAI8DKQX' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1-3jIt7jottf3gG2-YfyA02T4rAI8DKQX" -O noisy_train.h5 && rm -rf /tmp/cookies.txt

#Loading fully noist
https://drive.google.com/file/d/1uPokym7qOsUV-8EhwzZW19HOsJLca4bG/view?usp=sharing

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1uPokym7qOsUV-8EhwzZW19HOsJLca4bG' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1uPokym7qOsUV-8EhwzZW19HOsJLca4bG" -O full_noisy_train.h5 && rm -rf /tmp/cookies.txt
