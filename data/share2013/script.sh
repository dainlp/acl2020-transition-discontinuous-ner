# Download shareclef-ehealth-2013-natural-language-processing-and-information-retrieval-for-clinical-care-1.0.zip from https://physionet.org/content/shareclefehealth2013/1.0/ and unzip it

SHARE2013_DIR=/home/gdpr/Corpora/ShARe2013/shareclef-ehealth-2013-natural-language-processing-and-information-retrieval-for-clinical-care-1.0
cd $SHARE2013_DIR
mkdir train
unzip Task1TrainSetCorpus199.zip
mv ALLREPORTS train/text
unzip Task1TrainSetGOLD199knowtatorehost.zip
mv ALLSAVED train/ann

mkdir test
unzip Task1TestSetCorpus100.zip
mv ALLREPORTS test/text
tar -xf Task1Gold_SN2012.tar.bz2
mv Gold_SN2012 test/ann

# run the code
cd /home/gdpr/Downloads/acl2020-transition-discontinuous-ner-master/data/share2013

python extract_ann.py --input_dir=$SHARE2013_DIR/train/ann --text_dir=$SHARE2013_DIR/train/text --split=train
python extract_ann.py --input_dir=$SHARE2013_DIR/test/ann --text_dir=$SHARE2013_DIR/test/text --split=test

python tokenization.py --input_dir=$SHARE2013_DIR/train/text --split=train
python tokenization.py --input_dir=$SHARE2013_DIR/test/text --split=test

python convert_ann_using_token_idx.py

mkdir processed_share2013
python convert_text_inline.py --output_dir processed_share2013
