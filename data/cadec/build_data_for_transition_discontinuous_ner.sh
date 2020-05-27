mkdir /data/dai031/Experiments/CADEC/adr

echo "Extract annotations ..." >> build_data_for_transition_discontinous_ner.log
python extract_annotations.py --output_filepath /data/dai031/Experiments/CADEC/adr/ann --type_of_interest ADR --log_filepath build_data_for_transition_discontinous_ner.log

echo "Tokenization ..." >> build_data_for_transition_discontinous_ner.log
python tokenization.py --log_filepath build_data_for_transition_discontinous_ner.log

echo "Convert annotations from character level offsets to token level idx ..." >> build_data_for_transition_discontinous_ner.log
python convert_ann_using_token_idx.py --input_ann /data/dai031/Experiments/CADEC/adr/ann --output_ann /data/dai031/Experiments/CADEC/adr/tokens.ann --log_filepath build_data_for_transition_discontinous_ner.log

echo "Create text inline format ..." >> build_data_for_transition_discontinous_ner.log
python convert_text_inline.py --input_ann /data/dai031/Experiments/CADEC/adr/tokens.ann --output_filepath /data/dai031/Experiments/CADEC/adr/inline --log_filepath build_data_for_transition_discontinous_ner.log

echo "Split the data set into train, dev, test splits ..." >> build_data_for_transition_discontinous_ner.log
python split_train_test.py --input_filepath /data/dai031/Experiments/CADEC/adr/inline --output_dir /data/dai031/Experiments/CADEC/adr/split