import os, glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torchvision.transforms as transforms
import argparse
from tqdm import tqdm
import torchvision.models as models
import random
tqdm.pandas()

parser = argparse.ArgumentParser(description="Build embeddings for WSI compressed tensor")

parser.add_argument(
    "--input-dir", default='../../UBCData/', type=str, help="File directory of the patches"
)

if __name__ == "__main__":
    args = parser.parse_args()
    df_train = pd.read_csv(args.input_dir + "train.csv")
    for index, row in tqdm(df_train.iterrows(), total=df_train.shape[0], leave=True, position=0):
        try:
            if (str(row.image_id) + '.pt') not in os.listdir('../../CLAMFeatures/pt_files/'):
                print(row.image_id)
                os.system('cd temp && kaggle competitions download -c UBC-OCEAN -f train_images/' + str(row.image_id)  + '.png')
                os.system('cd temp && tar -xf ' + str(row.image_id)  + '.png.zip')
                os.system('cd temp && del /f /Q ' + str(row.image_id)  + '.png.zip')

                os.system('python create_patches_fp.py --source ./temp --save_dir ../../CLAMPatches --patch_size 256 --preset bwh_biopsy.csv --patch --seg')
                os.system('python extract_features_fp.py --data_h5_dir ../../CLAMPatches --data_slide_dir ./temp --csv_path ../../CLAMPatches/process_list_autogen.csv --feat_dir ../../CLAMFeatures --batch_size 512 --slide_ext .png')
                os.system('cd temp && del /f /Q ' + str(row.image_id)  + '.png')
            else:
                print(str(row.image_id) + ' already exists')
        except:
            print('Error')