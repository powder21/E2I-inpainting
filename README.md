# E2I-inpainting
Testing code of the paper 'E2I: Generative Inpainting from Edge to Image'

Requirements:
tf=1.4,numpy=1.13,scipy=0.19

There are two model file to prepare: HED model(converted from caffemodel) and E2I model before testing.

Models Links:
[BaiduNetDisk(AccessCode:tnkn)](https://pan.baidu.com/s/1rlFJxqetWS6AYBksaaZyNA)
/[OneDrive](https://1drv.ms/u/s!Ap2bi3TSun55lSmGnHbT5Dk3PvSx?e=c5LWBA)

Put your corrupted images and masks in different folders respectively. Note that the name of images and masks are with the same prefix. For example, Places365_test_00000042_img.png(image) and Places365_test_00000042_mask.png(mask) are with the same prefix 'Places365_test_00000042'.

Example of testing scipt on Places2 dataset
```
python pipeline_master_tcsvt.py --img_dir /path/to/corrupted/image/folder --mask_dir /path/to/mask/folder --output_dir /path/to/output/folder --hed_dir /path/to/hed/model --checkpoints_path /path/to/E2I_model/folder
```
