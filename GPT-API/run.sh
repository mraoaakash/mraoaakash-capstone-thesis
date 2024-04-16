BASEPATH=/Volumes/miccai/PathLDM/inputs/TCGA_Dataset


# python scripts/cleaner.py \
#     --summaryfolder $BASEPATH/summaries/summaries_75 \
#     --master $BASEPATH/TCGA_Reports.csv \
#     --input $BASEPATH/input.json \
#     --outputfolder $BASEPATH/summaries/ \

python scripts/main.py \
    --input $BASEPATH/summaries \
    --output $BASEPATH/summaries \
    --token_len 35 \

# python scripts/npzer.py \
#     --input $BASEPATH/train_test_brca_tumor_75 \
#     --summary $BASEPATH/summaries \
#     --output $BASEPATH \
#     --token_len 50 \