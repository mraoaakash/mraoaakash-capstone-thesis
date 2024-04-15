BASEPATH=/Users/mraoaakash/Documents/research/Capstone_Thesis/mraoaakash-capstone-thesis/GPT-API

cd $BASEPATH

python main.py \
    --summaryfolder $BASEPATH/data/input/summaries/ \
    --master $BASEPATH/data/input/TCGA_Reports.csv \
    --input $BASEPATH/data/input/input.json \
    --outputfolder $BASEPATH/data/output \