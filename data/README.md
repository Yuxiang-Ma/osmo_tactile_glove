# README

### Update 12/21/25
Sample demonstration is available here: https://drive.google.com/drive/folders/1SSE9pQetSoLbk82gnHd8Z3xHq7doRGgf?usp=sharing

There is a folder "00" which contains the following files to run through the postprocessing pipeline:
- `rgbs_public.pkl`
- `left_ir_public.pkl`
- `right_ir_public.pkl`
- `mags_public.pkl` (tactile glove data)

Each `.pkl` file contains a list of the data files, i.e. `(n,)`, and each item in the list is time-aligned with other data streams by list index such that `(rgbs_public[i], left_ir_public[i], right_ir_public[i], mags_public[i])` correspond.

Each item in `mags_public.pkl` contains a timestamp and a `(30,)` array which can be reshaped to `(10,3)` for the same order as the ROS2 node:
```
index_mag0_x, index_mag0_y, index_mag0_z
index_mag1_x, index_mag1_y, index_mag1_z
middle_mag0_x, middle_mag0_y, middle_mag0_z
middle_mag1_x, middle_mag1_y, middle_mag1_z
ring_mag0_x, ring_mag0_y, ring_mag0_z
ring_mag1_x, ring_mag1_y, ring_mag1_z
pinky_mag0_x, pinky_mag0_y, pinky_mag0_z
pinky_mag1_x, pinky_mag1_y, pinky_mag1_z
thumb_mag0_x, thumb_mag0_y, thumb_mag0_z
thumb_mag1_x, thumb_mag1_y, thumb_mag1_z
```
Note that these are from the fingertips of the glove, which is what we use in the paper to train our policy.

The left and right IR images have been "cropped" to the region of interest, where unnecessary parts of the image are replaced with white pixels. 

This preserves the original image dimensions (1280x720) so that the Realsense camera intrinsics are correct when running the vision models, while maintaining privacy for the demonstrator.


TODO: set up longer-term host for data and data download script
