import cv2
import numpy as np
from PIL import ImageFilter, Image
import os
import os.path
import sys
from collections import OrderedDict
import json
import xml.etree.cElementTree as ET

#xml_file_path = 'unlv_tmp_xml/'
xml_file_path = 'xml/'
final_json_path = ''

############################################################
#structure json file for instance segmentation
final_json = {}
final_json['info'] = {}
final_json['licenses'] = {}
final_json['images'] = []
final_json['type'] = 'instances'
final_json['annotations'] = []
final_json['categories'] = []

###########################################################
#information of info field of json file
final_json['info']["description"] = 'equation Dataset'
final_json['info']["url"] = 'http://mscoco.org'
final_json['info']["version"] = '1.0'
final_json['info']["year"] = '2014'
final_json['info']["contributor"] = 'Microsoft COCO group'
final_json['info']["data_created"] = '2019/06/25'

###############################################################
#information of license information
final_json['licenses']["id"] = 0
final_json['licenses']["name"] = "Attribution-NonCommercial-ShareAlike License"
final_json['licenses'][
    "url"] = "http://images.cocodataset.org/val2014/COCO_val2014_000000391895.jpg"

##########################################################
#read GT xml file from folder
image_index = 0
image_id = 1000
for xml_file in os.listdir(xml_file_path):
    if xml_file.endswith('xml'):
        print("xml==>", xml_file)
        file_prefix = xml_file.split(".xml")[0]
        image_name = "{}.jpg".format(file_prefix)
        xml_path = xml_file_path + xml_file
        try:
            groundtree = ET.parse(xml_path)
            root = groundtree.getroot()
            objects = []

            for child2 in groundtree.findall('size'):
                width = int(child2.find('width').text)
                height = int(child2.find('height').text)
                depth = int(child2.find('depth').text)
            #print("width, height, depth", width, height, depth)
            #############################################################
            #information images and linence for json file
            #image information
            image_info = {}
            image_info["id"] = image_id  #like 1000
            image_info["width"] = width  #width of image
            image_info["height"] = height  #height of image
            image_info["file_name"] = image_name
            image_info["license"] = '0'  #like 0, 1, 2, 3 etc
            image_info[
                "url"] = "http://images.cocodataset.org/val2014/COCO_val2014_000000391895.jpg"
            image_info["date_captured"] = "2013-11-14 11:18:45"
            final_json['images'].append(image_info)
            #############################################################
            for child1 in groundtree.findall('object'):
                for child2 in child1.findall('cells'):
                    for child3 in child2.findall('tablecell'):
                        obj_struct = {}
                        obj_struct['name'] = 'tablecell'
                        obj_struct['bbox'] = [
                            int(child3.find('x0').text),
                            int(child3.find('y0').text),
                            int(child3.find('x1').text),
                            int(child3.find('y1').text)
                        ]
                        obj_struct['start_row'] = int(
                            child3.find('start_row').text)
                        obj_struct['start_col'] = int(
                            child3.find('start_col').text)
                        obj_struct['end_row'] = int(
                            child3.find('end_row').text)
                        obj_struct['end_col'] = int(
                            child3.find('end_col').text)
                        objects.append(obj_struct)
            for j in range(len(objects)):
                old = objects[j]
                caption = old['name']
                list_pt = old['bbox']
                x_min = list_pt[0]
                y_min = list_pt[1]
                x_max = list_pt[2]
                y_max = list_pt[3]
                if caption == 'tablecell':
                    category_id = 1

                    #print("x_min, y_min, x_max, y_max", x_min,y_min,x_max,y_max)
                    #################################################################
                    #annotation information
                    annotation_info = {}
                    annotation_info["id"] = image_index
                    annotation_info["image_id"] = image_id
                    annotation_info["category_id"] = category_id
                    annotation_info["segmentation"] = []
                    annotation_info["area"] = float(
                        (x_max - x_min) * (y_max - y_min))
                    annotation_info["bbox"] = [
                        x_min, y_min, x_max - x_min, y_max - y_min
                    ]
                    annotation_info["iscrowd"] = 0
                    annotation_info["start_row"] = old["start_row"]
                    annotation_info["start_col"] = old["start_col"]
                    annotation_info["end_row"] = old["end_row"]
                    annotation_info["end_col"] = old["end_col"]
                    segmentation_list = []
                    for l in range(y_min, y_max, 1):
                        segmentation_list.append(x_min)
                        segmentation_list.append(l)
                    for l in range(x_min, x_max, 1):
                        segmentation_list.append(l)
                        segmentation_list.append(y_max)
                    for l in range(y_max, y_min, -1):
                        segmentation_list.append(x_max)
                        segmentation_list.append(l)
                    for l in range(x_max, x_min, -1):
                        segmentation_list.append(l)
                        segmentation_list.append(y_min)
                    annotation_info['segmentation'].append(segmentation_list)
                    final_json['annotations'].append(annotation_info)
                    image_index = image_index + 1
                #################################################################
            image_id = image_id + 1
        except:
            pass

##########################################################################
#category information
category_info = {}
category_info["id"] = 1
category_info["name"] = "tablecell"
category_info["supercategory"] = "tablecell"
final_json['categories'].append(category_info)

#########################################################################
# if not os.path.exists(final_json_path):
#     os.makedirs(final_json_path)
output_file_name = final_json_path + "instances_val2014.json"
#print("****************************    Final json is **********************\n\n", final_json)
with open(output_file_name, 'w+') as outfile:
    json.dump(final_json, outfile)
