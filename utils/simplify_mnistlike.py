import os, shutil
import csv
import pdb

root_dir = '../data/fmnist_png'
dest_dir = '../data/fmnist_easy'

train_dir = os.path.join(root_dir,'training')
test_dir = os.path.join(root_dir, 'testing')

# make directories
if not os.path.exists(dest_dir):
    os.mkdir(dest_dir)
    os.mkdir(dest_dir + '/train')
    os.mkdir(dest_dir + '/test')





for source in [train_dir, test_dir]:
    if source == train_dir:
        csv_name = 'train.csv'
        img_dest = os.path.join(dest_dir,'train')
    else:
        csv_name = 'test.csv'
        img_dest = os.path.join(dest_dir,'test')

    anns_map = []

    for clas in sorted(os.listdir(source)):
        clas_dir = os.path.join(source, clas)

        for img in os.listdir(clas_dir):
            anns_map.append((img,clas))
            shutil.copy(os.path.join(clas_dir,img), img_dest)
            print(img)



    csv_file = os.path.join(dest_dir,csv_name)
    with open(csv_file, 'w') as f:
        csv_out = csv.writer(f)
        csv_out.writerows(anns_map)
