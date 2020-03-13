from Dataset import MIDataset
from dataset_utils import load_img_and_gt, visualize, calculate_histogram, plot_histograms, cluster, visualize_tensor
from transformation_utils import color_correct, color_correct_tensor, get_training_augmentation

img, mask, gt = load_img_and_gt('bmug_b_r.png')
#print(img-gt)
#print(gt.shape)
dataset = MIDataset(datatype='train', transforms=get_training_augmentation())
img, mask, gt = dataset[10]
#hists = calculate_histogram(gt.numpy())
#cluster(img.numpy())
#_, _, center = cluster(gt)
#print(center * 255)
#print(gt.shape)
#print(img.shape)
cimg = color_correct_tensor(img, gt)
#cluster(cimg.numpy())

visualize_tensor(img.cpu(), gt, mask, cimg)
