SHAREPATH=/mnt/storage/aakashrao/cifsShare
BASEPATH=$SHAREPATH/PathLDM/inputs/TCGA_Dataset


# for loop for vales 20,35,50
for i in 75 # 20 35 50
do
    # echo "Processing for token length $i"
    # python scripts/extract_from_test.py \
    #     --data_dir $BASEPATH \
    #     --token_num  $i   \
    #     --outdir $BASEPATH/TCGA_Dataset_images/original_images

    python scripts/generate_images.py \
        --ckpt_path $SHAREPATH/PathLDM/outputs/04-08T22-44_testing/checkpoints/last.ckpt \
        --config_path $SHAREPATH/PathLDM/outputs/04-08T22-44_testing/configs/04-08T22-44-project.yaml \
        --data_dir $BASEPATH \
        --token_num $i \
        --outdir $BASEPATH/generated_$i 
done