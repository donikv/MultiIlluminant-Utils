Using the Dataset:

The dataset consists of a laboratory part, for images taken in a closed room
under controlled settings, and a real-world part, for images taken in various
other environments. 
The lab images are directly converted from the camera's (Sigma Foveon) raw
output, while the real-world images have undergone a non-linear color
enhancement, to emulate images as obtained after internal processing in the
camera.

In this package, the dataset is has a resolution of 452x260 pixels

To use the dataset, use the data from these directories:
- img/         the input images, in 16 bit color depth.
- groundtruth/ the computed pixelwise ground truth, in 8 bit color depth. Black
			   pixels indicate 'no information', e.g. due to very dark image
			   regions.
- masks/       the laboratory data is partially underexposed. Very dark pixels
               are excluded using these masks.

The input images are in 16 bit color depth. In order to get a preview with an 8
bit image viewer, have a look at the directories named "srgb8bit".

The laboratory images are exposed to 'red', 'blue' and 'white' illuminants.
Which illuminant was switched on is encoded in the file name: 'w', 'r', 'b'
denote 'white', 'red', and 'blue', respectively. 'n' denotes 'none', i.e., one
of the illuminants was switched off in this image.

The real-world images are exposed to either one indoor or outdoor light and a
colored projector, or to sun and no sun (shadow/ambient light).

Example evaluation code for some single-illuminant estimators is in the
subdirectory ./matlab_code

Index of the directory:

./lab/groundtruth        : ground truth for the laboratory dataset
./lab/img                : input images for the laboratory dataset
./lab/masks              : masks for removing too dark pixels
./lab/srgb8bit           : 8 bit images for preview in common image browsers

./realworld/groundtruth  : ground truth for the real-world dataset
./realworld/img          : input images for the real-world dataset
./realworld/srgb8bit     : 8 bit images for preview in common image browsers

./matlab_code            : example evaluation code for single-illuminant estimators

