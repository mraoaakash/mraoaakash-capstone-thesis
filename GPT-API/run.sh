BASEPATH=/Volumes/miccai/PathLDM/inputs/TCGA_Dataset


# python scripts/cleaner.py \
#     --summaryfolder $BASEPATH/summaries/summaries_75 \
#     --master $BASEPATH/TCGA_Reports.csv \
#     --input $BASEPATH/input.json \
#     --outputfolder $BASEPATH/summaries/ \

# python scripts/main.py \
#     --input $BASEPATH/summaries \
#     --output $BASEPATH/summaries \
#     --token_len 35 \


# for loop for vales 20,35,50
# for i in 20 35 50
# do
#     echo "Processing for token length $i"
#     python scripts/npzer.py \
#         --input $BASEPATH/train_test_brca_tumor_$i \
#         --summary $BASEPATH/summaries \
#         --output $BASEPATH \
#         --token_len 35 
# done


for i in 20 35 50 75
do
    echo "Processing for token length $i"
    python scripts/evaluate_summaries.py \
        --input /mnt/storage/aakashrao/cifsShare/PathLDM/inputs/TCGA_Dataset/summaries \
        --output /mnt/storage/aakashrao/cifsShare/PathLDM/outputs/evaluate_summaries \
        --token_num $i 
done

