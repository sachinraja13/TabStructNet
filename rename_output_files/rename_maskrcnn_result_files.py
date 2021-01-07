import cv2
import numpy as np
from PIL import ImageFilter, Image
import os
import os.path
import sys
import json

json_file_path = 'GT_json_file/'
output_jpg_folder = 'results/result_jpg/'
output_text_folder = 'results/result_text/'
output_row_pickle_folder = 'results/result_row_pkl/'
output_col_pickle_folder = 'results/result_col_pkl/'
rename_output_jpg_folder = 'rename_results/result_jpg/'
rename_output_text_folder = 'rename_results/result_text/'
rename_output_row_pickle_folder = 'rename_results/result_row_pkl/'
rename_output_col_pickle_folder = 'rename_results/result_col_pkl/'

from shutil import rmtree
try:
    rmtree(rename_output_jpg_folder)
except Exception as e:
    print(e)
    print("cannot remove " + rename_output_jpg_folder)
try:
    os.makedirs(rename_output_jpg_folder)
except:
    print("cannot create " + rename_output_jpg_folder)
try:
    rmtree(rename_output_text_folder)
except:
    print("cannot remove " + rename_output_text_folder)
try:
    os.makedirs(rename_output_text_folder)
except:
    print("cannot create " + rename_output_text_folder)
try:
    rmtree(rename_output_row_pickle_folder)
except:
    print("cannot remove " + rename_output_row_pickle_folder)
try:
    os.makedirs(rename_output_row_pickle_folder)
except:
    print("cannot create " + rename_output_row_pickle_folder)
try:
    rmtree(rename_output_col_pickle_folder)
except:
    print("cannot remove " + rename_output_col_pickle_folder)
try:
    os.makedirs(rename_output_col_pickle_folder)
except:
    print("cannot create " + rename_output_col_pickle_folder)

copy = "cp -r "
id_list = []
name_list = []

with open(json_file_path + '/instances_val2014.json') as json_file:
    data = json.load(json_file)
    for p in data["images"]:
        #print('id: ', p["id"])
        #print('name: ', p["file_name"])
        id_list.append(p["id"])
        n = p['file_name']
        name_list.append(n)
    print(name_list)
#print("id_list==>",id_list)

for text_file in os.listdir(output_text_folder):
    try:
        pass
        file_id = text_file.split(".txt")[0][3:]
        print(file_id)
        source_prefix = "AR_" + file_id
        source_jpg_path = output_jpg_folder + source_prefix + ".jpg"
        source_text_path = output_text_folder + text_file
        source_row_pkl_path = output_row_pickle_folder + source_prefix + "_row.pkl"
        source_col_pkl_path = output_col_pickle_folder + source_prefix + "_col.pkl"
        image = cv2.imread(source_jpg_path)

        index = id_list.index(int(file_id))
        #print("index", index)
        new_image_name = name_list[index]
        #print("new_name==", new_image_name)
        new_prefix = new_image_name.split(".jpg")[0]
        #print("new_text_name==", new_text_name)
        destination_jpg_path = rename_output_jpg_folder + new_image_name
        destination_text_path = rename_output_text_folder + new_prefix + ".txt"
        destination_row_pkl_path = rename_output_row_pickle_folder + new_prefix + ".pkl"
        destination_col_pkl_path = rename_output_col_pickle_folder + new_prefix + ".pkl"

        os.system(copy + source_text_path + " " + destination_text_path)
        os.system(copy + source_row_pkl_path + " " + destination_row_pkl_path)
        os.system(copy + source_col_pkl_path + " " + destination_col_pkl_path)

        cv2.imwrite(destination_jpg_path, image)
    except:
        print(text_file + " ERROR")
