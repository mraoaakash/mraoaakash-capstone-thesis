BASEPATH=/Users/mraoaakash/Documents/research/Capstone_Thesis/mraoaakash-capstone-thesis/GPT-API

cd $BASEPATH

# python scripts/cleaner.py \
#     --summaryfolder $BASEPATH/data/input/summaries/ \
#     --master $BASEPATH/data/input/TCGA_Reports.csv \
#     --input $BASEPATH/data/input/input.json \
#     --outputfolder $BASEPATH/data/output \

python scripts/main.py \
    --input $BASEPATH/data/output \
    --output $BASEPATH/data/output/ \
    --token_len 50 \