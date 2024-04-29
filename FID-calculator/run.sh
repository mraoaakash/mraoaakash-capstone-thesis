SHAREPATH=/mnt/storage/aakashrao/cifsShare
BASEPATH=$SHAREPATH/PathLDM/inputs

# for loop for vales 20,35,50
# echo "Processing for token length $i"
# python scripts/extract_from_test.py \
#     --data_dir $BASEPATH \
#     --token_num  $i   \
#     --outdir /media/chs.gpu/DATA/aakash-work/PathLDM/input/original_images

echo "/media/chs.gpu/DATA/aakash-work/PathLDM/input/generated_$i"

# python scripts/generate_images.py \
#     --ckpt_path $SHAREPATH/PathLDM/outputs/04-08T22-44_testing/checkpoints/last.ckpt \
#     --config_path $SHAREPATH/PathLDM/outputs/04-08T22-44_testing/configs/04-08T22-44-project.yaml \
#     --data_dir $BASEPATH \
#     --token_num 75 \
#     --batch_size 32 \
#     --outdir /media/chs.gpu/DATA/aakash-work/PathLDM/input/generated_75

# python scripts/generate_images.py \
#     --ckpt_path $SHAREPATH/PathLDM/outputs/04-20T12-35_testing/checkpoints/last.ckpt \
#     --config_path $SHAREPATH/PathLDM/outputs/04-20T12-35_testing/configs/04-20T12-35-project.yaml \
#     --data_dir $BASEPATH \
#     --token_num 20 \
#     --batch_size 32 \
#     --outdir /media/chs.gpu/DATA/aakash-work/PathLDM/input/generated_20


# python -m pytorch_fid --save-stats /media/chs.gpu/DATA/aakash-work/PathLDM/input/original_images /media/chs.gpu/DATA/aakash-work/PathLDM/input/original_feed/images


# python -m pytorch_fid /media/chs.gpu/DATA/aakash-work/PathLDM/input/original_images /media/chs.gpu/DATA/aakash-work/PathLDM/input/generated_75/images --batch-size 32


python -m pytorch_fid /media/chs.gpu/DATA/aakash-work/PathLDM/input/original_images /media/chs.gpu/DATA/aakash-work/PathLDM/input/generated_20/images --batch-size 32