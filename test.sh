for FOLD in 0 1 2 3 4 
do
CUDA_VISIBLE_DEVICES=1  nnUNet_train 3d_fullres nnUNetTrainerV2_FLARE_Big 22 $FOLD -p nnUNetPlansFLARE22Big
done
