# Download shareclef-ehealth-evaluation-lab-2014-task-2-disorder-attributes-in-clinical-reports-1.0.zip.zip from https://physionet.org/content/shareclefehealth2014task2/1.0/ and unzip it

SHARE2014_DIR=/home/gdpr/Corpora/ShAReCLEF2014-t2/shareclef-ehealth-evaluation-lab-2014-task-2-disorder-attributes-in-clinical-reports-1.0
cd $SHARE2014_DIR
mkdir train
unzip 2014ShAReCLEFeHealthTasks2_training_10Jan2014.zip
mv 2014ShAReCLEFeHealthTasks2_training_10Jan2014/2014ShAReCLEFeHealthTask2_training_corpus train/text
mv 2014ShAReCLEFeHealthTasks2_training_10Jan2014/2014ShAReCLEFeHealthTask2_training_pipedelimited train/ann

mkdir test
unzip ShAReCLEFeHealth2014Task2_test_default_values.zip
mv ShAReCLEFeHealth2014Task2_test_default_values_with_corpus/ShAReCLEFeHealth2104Task2_test_data_corpus test/text
unzip ShAReCLEFeHealth2014_test_data_gold.zip
mv ShAReCLEFeHealth2014_test_data_gold test/ann

# run the code
cd /home/gdpr/Downloads/acl2020-transition-discontinuous-ner-master/data/share2014

python extract_ann.py --ann_dir=$SHARE2014_DIR/train/ann --text_dir=$SHARE2014_DIR/train/text --split=train
python extract_ann.py --ann_dir=$SHARE2014_DIR/test/ann --text_dir=$SHARE2014_DIR/test/text --split=test

cp ../share2013/tokenization.py ./
python tokenization.py --input_dir=$SHARE2014_DIR/train/text --split=train
python tokenization.py --input_dir=$SHARE2014_DIR/test/text --split=test

cp ../share2013/convert_ann_using_token_idx.py ./
python convert_ann_using_token_idx.py

mkdir processed_share2014
cp ../share2013/convert_text_inline.py ./
python convert_text_inline.py --output_dir processed_share2014
