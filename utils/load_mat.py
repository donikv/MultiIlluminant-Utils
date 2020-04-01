from scipy.io import loadmat

def get_gt_for_image(name, path = 'data/dataset_hdr/real_illum/real_illum', ext = 'mat'):
    file = loadmat(f'{path}/{name}.{ext}')
    rgb = file['real_illum'][0][0][0].mean(0)
    return rgb