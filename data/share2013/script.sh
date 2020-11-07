python extract_ann.py --input_dir=/data/dai031/Corpora/ShAReCLEF2013/train/ann \
--text_dir=/data/dai031/Corpora/ShAReCLEF2013/train/text --split=train

python extract_ann.py --input_dir=/data/dai031/Corpora/ShAReCLEF2013/test/Gold_SN2012 \
--text_dir=/data/dai031/Corpora/ShAReCLEF2013/test/text --split=test

python tokenization.py --input_dir=/data/dai031/Corpora/ShAReCLEF2013/train/text --split=train
python tokenization.py --input_dir=/data/dai031/Corpora/ShAReCLEF2013/test/text --split=test

python convert_ann_using_token_idx.py

mkdir /data/dai031/Experiments/2020-11-07-01
python convert_text_inline.py --output_dir /data/dai031/Experiments/2020-11-07-01
