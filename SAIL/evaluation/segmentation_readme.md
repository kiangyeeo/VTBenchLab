# Open Vocabulary Semantic Segmentation with SAIL

This repository provides instructions for open vocabulary semantic segmentation using the SAIL framework.

## 1. Install Dependencies

## Dependencies and Installation


```
# create new anaconda env
conda create -n SAILSeg python=3.10
conda activate SAILSeg

# install torch and dependencies
pip install -r seg_requirements.txt
```

Ensure you are using **CUDA Toolkit 12.1.1** for compatibility.

## 2. Download Datasets

Refer to the [MMSegmentation Dataset Preparation Guide](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md) to download the following datasets:

- **COCOStuff164K**
- **VOC20**
- **ADE20K**

Make sure the datasets are properly organized as per the instructions.

## 3. Evaluation Instructions

To evaluate, follow these steps:

1. **Set the Task and Configuration**:

   - Use the `--task` flag to specify the `segmentation` task.
   - Update the `data_root` in the corresponding configuration file (e.g., `evaluation/seg_configs/cfg_{task}_SAIL.py` for `voc20`, `ade20k`, or `coco_stuff164k`) to point to the dataset location.
   - Provide the configuration file path using the `--seg_task_config` flag.

2. **Run the Evaluation Script**:
   Use the following command to evaluate:

   ```bash
   python eval.py \
       --head-weights-path $checkpoint_path \
       --task $task \
       --dataset_root_dir $DATASET_ROOT_DIR \
       --linear-type star \
       --vision-model $vision_model \
       --text-model $text_model \
       --batch_size 32 \
       --seg_task_config ./evaluation/seg_configs/cfg_coco_stuff164k_SAIL.py  \
       --agg_mode concat \
       --width_factor 8
   ```

## Notes

- Ensure that all dataset paths are updated in the configuration files.
- Use the appropriate configuration file for the task you are evaluating. 

For any issues or questions, please refer to the relevant documentation or raise an issue in the repository.

This is built on top of [ClearCLIP](https://github.com/mc-lan/ClearCLIP/tree/main)

