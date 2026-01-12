# OSMO: Open-Source Tactile Glove for Human-to-Robot Skill Transfer
Jessica Yin, Haozhi Qi*, Youngsun Wi*, Sayantan Kundu, Mike Lambeta, William Yang, Changhao Wang, Tingfan Wu, Jitendra Malik, and Tess Hellebrekers

*equal contribution 

Project website: [jessicayin.github.io/osmo_tactile_glove](https://jessicayin.github.io/osmo_tactile_glove/)

ArXiv: [https://arxiv.org/abs/2512.08920](https://arxiv.org/abs/2512.08920)

# OSMO Hardware

Hardware guides and PCB files are on the `website` branch of this repo.

# OSMO Glove Data Pipeline

### Code has references to "bowie", which was the internal name of OSMO. Bowie and the OSMO glove may be treated as interchangeable when understanding the code.



1. Extract the hamer keypoints: (in osmo env)
 ```
 conda activate osmo
 python labs/glove2robot/postprocess/extract_hamer.py data/00/
  
 ```
 - sample data collect for paper can be downloaded via `data/download_data.sh` (TODO: upload data and update script)
2. **Optional but recommended** Inspect the extracted keypoints: (in osmo env) 
```
conda activate osmo
python scripts/plot_keypoints_with_osmo.py
```
3. Retarget the hamer keypoints to Psyonic to construct the dataset: (in osmo_kinematics env). 
   
   Requires camera extrinsics represented as a (4,4) rigid transformation matrix to be saved as a .npy file and the path of the file to be provided in `labs/glove2robot/config/config_extract_hamer.yaml` under `camera_calibration` 
```
conda activate osmo_kinematics
python kinematics/construct_retarget_dataset.py
```

## Setting up environments

Osmo Env
```
conda env create -f conda/osmo.yml
```
Osmo Kinematics Env
```
conda env create -f conda/osmo_kinematics.yml
```


## Extract Hamer
- The most computation-intensive step of the pipeline, we automate processing of multiple experiments using `labs/glove2robot/config/run_hardcoded_batches.sh`

## Plot Keypoints Osmo
- Used for visually validating the extract hamer keypoints and generating the combined visualization (including the magnetometer readings)

## Construct Retargeted Dataset
- Retargets the extracted hamer keypoints to Psyonic + Franka kinematics to construct the training dataset


# Train and Deploy Instructions
[Policy Training and Deployment](glovedp/README.md)

# BibTeX
If you find this work helpful, please consider citing:

```
@article{yin2025osmo,
    title={OSMO: Open-Source Tactile Glove for Human-to-Robot Skill Transfer},
    author={Jessica Yin and Haozhi Qi and Youngsun Wi and Sayantan Kundu and Mike Lambeta and William Yang and Changhao Wang and Tingfan Wu and Jitendra Malik and Tess Hellebrekers},
    journal={arXiv:2512.08920},
    year={2025}
}
```
