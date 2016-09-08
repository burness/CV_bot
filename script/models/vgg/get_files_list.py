'''
Coding Just for Fun
Created by burness on 16/8/31.
'''
import os
from PIL import Image
import numpy as np

def get_files_list(data_path):
    samples = []
    targets = []
    label = 0
    try: # Python 2
        classes = sorted(os.walk(data_path).next()[1])
    except Exception: # Python 3
        classes = sorted(os.walk(data_path).__next__()[1])
    for c in classes:
        c_dir = os.path.join(data_path, c)
        try: # Python 2
            walk = os.walk(c_dir).next()
        except Exception: # Python 3
            walk = os.walk(c_dir).__next__()
        for sample in walk[2]:
            samples.append(os.path.join(c_dir, sample))
            targets.append(label)
        label += 1
    with open("files_list", "w") as fwrite:
        for index, i in enumerate(samples):
            # try:
            #     img = Image.open(i)
            #     img.load()
            #     img = np.asarray(img, dtype="float32")
            #     if img.shape[2] == 3:
            #         line = i+' '+str(targets[index])+'\n'
            #         fwrite.write(line)
            # except:
            #     continue
            line = i+' '+str(targets[index])+'\n'
            fwrite.write(line)


if __name__ == '__main__':
    get_files_list('/Users/burness/git_repository/CV_bot/data/new')
