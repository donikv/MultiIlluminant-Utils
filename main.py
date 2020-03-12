from Dataset import MIDataset
from dataset_utils import load_img_and_gt, visualize, calculate_histogram, plot_histograms, cluster
from transformation_utils import color_correct, color_correct_tensor

img, mask, gt = load_img_and_gt('bmug_b_r.png')
#print(img-gt)
#print(gt.shape)
dataset = MIDataset()
img, mask, gt = dataset[10]
hists = calculate_histogram(gt.numpy())
#cluster(img.numpy())
_, _, center = cluster(gt.numpy())
print(center * 255)
#print(gt.shape)
#print(img.shape)
cimg = color_correct_tensor(img, gt)
#cluster(cimg.numpy())

visualize(img, gt, mask, cimg)
