BASEPATH=/mnt/storage/aakashrao/cifsShare/PathLDM/inputs/TCGA_Dataset


# for loop for vales 20,35,50
for i in 75 # 20 35 50
do
    echo "Processing for token length $i"
    python scripts/extract_from_train.py \
        --data_dir $BASEPATH \
        --token_num  $i   \
        --outdir $BASEPATH/TCGA_Dataset_images/original_images/ 
done