python train.py --dataroot /nitthilan/data/ADGAN/data/images/fashion_resize --dirSem /nitthilan/data/ADGAN/data/ --pairLst /nitthilan/data/ADGAN/data/images/fashion-resize-pairs-train.csv --name fashion_adgan_test --model adgan --lambda_GAN 5 --lambda_A 1 --lambda_B 1 --dataset_mode keypoint --n_layers 3 --norm instance --batchSize 12 --pool_size 0 --resize_or_crop scale_width --gpu_ids 0,1,2 --BP_input_nc 18 --SP_input_nc 8 --no_flip --which_model_netG ADGen --niter 500 --niter_decay 500 --checkpoints_dir ./checkpoints --L1_type l1_plus_perL1 --n_layers_D 3 --with_D_PP 1 --with_D_PB 1 --display_id 0