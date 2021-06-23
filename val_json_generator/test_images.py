import numpy as np
import sys
import xml.etree.ElementTree as ET
import scipy
import cv2
import os
import os.path
import string
from PIL import Image
from shutil import rmtree

dir_list = ['images/']
output_xml_dir = 'xml/'
output_images_dir = 'val2014/'

try:
    rmtree(output_xml_dir)
except:
    pass
try:
    os.mkdir(output_xml_dir)
except:
    pass
try:
    rmtree(output_images_dir)
except:
    pass
try:
    os.mkdir(output_images_dir)
except:
    pass


def create_root(file_prefix, width, height, depth):
    root = ET.Element("annotations")
    ET.SubElement(root, "folder").text = "images"
    ET.SubElement(root, "filename").text = "{}".format(file_prefix)
    ET.SubElement(root,
                  "path").text = output_images_dir + "{}".format(file_prefix)
    source = ET.SubElement(root, "source")
    ET.SubElement(source, "database").text = "Unknown"
    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(size, "depth").text = str(depth)
    ET.SubElement(root, "segmentated").text = "0"
    return root


def create_object_annotation(root, table_list, table_information_list):
    length_table_list = len(table_list)
    print("length_table_list==>", length_table_list)
    for i in range(length_table_list):
        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = "table"
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = str(0)
        ET.SubElement(obj, "difficult").text = str(0)
        bbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bbox, "xmin").text = str(table_information_list[i][0])

        ET.SubElement(bbox, "ymin").text = str(table_information_list[i][2])
        ET.SubElement(bbox, "xmax").text = str(table_information_list[i][1])
        ET.SubElement(bbox, "ymax").text = str(table_information_list[i][3])
        cells = ET.SubElement(obj, "cells")
        table_details = table_list[i]
        for j in range(len(table_details)):
            cell_detail = table_details[j]
            cell = ET.SubElement(cells, "tablecell")
            ET.SubElement(cell, "dont_care").text = str(cell_detail[0])
            ET.SubElement(cell, "end_col").text = str(cell_detail[1])
            ET.SubElement(cell, "end_row").text = str(cell_detail[2])
            ET.SubElement(cell, "start_col").text = str(cell_detail[3])
            ET.SubElement(cell, "start_row").text = str(cell_detail[4])
            ET.SubElement(cell, "x0").text = str(cell_detail[5])
            ET.SubElement(cell, "x1").text = str(cell_detail[6])
            ET.SubElement(cell, "y0").text = str(cell_detail[7])
            ET.SubElement(cell, "y1").text = str(cell_detail[8])

    return root


for dir_item in dir_list:
    index = 0
    for img_file in os.listdir(dir_item):
        try:
            img_fname = img_file
            if '.TIFF' in img_fname:
                img_fname = img_fname.replace('.TIFF', '.jpg')
            if '.tiff' in img_fname:
                img_fname = img_fname.replace('.tiff', '.jpg')
            if '.JPG' in img_fname:
                img_fname = img_fname.replace('.JPG', '.jpg')
            if '.png' in img_fname:
                img_fname = img_fname.replace('.png', '.jpg')
            read_img_path = dir_item + img_file
            print(read_img_path)
            out_img_path = output_images_dir + dir_item[:-1] + '_' + str(
                index) + ".jpg"
            out_xml_path = output_xml_dir + dir_item[:
                                                     -1] + '_' + img_fname.replace(
                                                         '.jpg', '.xml')
            original_image = cv2.imread(read_img_path)
            cv2.imwrite(out_img_path, original_image)
            root = create_root(dir_item[:-1] + '_' + img_fname, 100, 100, 3)
            root = create_object_annotation(
                root, [[[False, 0, 0, 0, 0, 0, 100, 0, 100]]],
                [[0, 100, 0, 100]])
            tree = ET.ElementTree(root)
            tree.write("{}{}.xml".format(output_xml_dir,
                                         dir_item[:-1] + '_' + str(index)))
            index += 1

        except Exception as e:
            print(e)
