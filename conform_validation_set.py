from skimage import io
import os
from collections import defaultdict
dataset = []
class_count = defaultdict()
root_dir = "data/tiny-imagenet-200/val/"
dest_path = "data/tiny-imagenet-200/val-fixed/"
os.mkdir(dest_path)
file = open(root_dir+'val_annotations.txt', 'r')
for line in file.readlines():
    tokens = line.strip().split('\t')
    if len(tokens) != 6:
        raise ValueError("malformed line in val annotations")
    img_path, class_label = tokens[:2]
    image = io.imread(root_dir+"images/"+img_path)

    if not os.path.isdir(dest_path+class_label):
        os.mkdir(dest_path+class_label)
        os.mkdir(dest_path+class_label+"/images/")
        class_count[class_label]=0
    else:
        class_count[class_label]+=1
    io.imsave(dest_path+class_label+"/images/"+class_label+"_"+str(class_count[class_label])+".JPEG",image)

