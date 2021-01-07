import tensorflow as tf
import os
import sys
import time
import numpy as np
import imgaug  # https://github.com/aleju/imgaug (pip3 install imgaug)
import cv2

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils

import zipfile
import urllib.request
import shutil
from mrcnn import visualize
import matplotlib
# Agg backend runs without a display
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

#result_path = os.path('/ssd_scratch/cvit/ajoy/data/coco/result/')
#if not os.path.exists(result_path):
#   os.makedirs(result_path)

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_MODEL_PATH = os.path.join('coco_model/coco/', "mask_rcnn_coco.h5")
#('/ssd_scratch/cvit/ajoy/annual_report_table_public_finetune/coco/log_after_finetune/logs/coco20190701T0805/',"mask_rcnn_coco_0120.h5")
#('/ssd_scratch/cvit/ajoy/annual_report_table_public_finetune/coco/logs/coco20190701T0805/',"mask_rcnn_coco_0080.h5")
#('/ssd_scratch/cvit/ajoy/data/coco/ini_model/', "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join('trained_model/tabnet/', "logs")
DEFAULT_DATASET_YEAR = "2014"

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

############################################################
#  Configurations
############################################################


class TabNetConfig(Config):
    """Configuration for training on MS COCO.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """
    # Give the configuration a recognizable name
    NAME = "tab"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Uncomment to train on 8 GPUs (default is 1)
    GPU_COUNT = 1

    #BACKBONE = "resnet50"
    #TRAIN_ROIS_PER_IMAGE = 100
    #MAX_GT_INSTANCES = 50
    #IMAGE_MIN_DIM = 256
    #IMAGE_MAX_DIM = 256
    #STEPS_PER_EPOCH = 3000
    #PRE_NMS_LIMIT = 600
    #POST_NMS_ROIS_TRAINING = 100
    #POST_NMS_ROIS_INFERENCE = 100
    #IMAGE_RESIZE_MODE = "crop"
    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # COCO has 80 classes


############################################################
#  Dataset
############################################################


class TabDataset(utils.Dataset):
    def load_tab(self,
                 dataset_dir,
                 subset,
                 year=DEFAULT_DATASET_YEAR,
                 class_ids=None,
                 class_map=None,
                 return_tab=False):

        tab = COCO("{}/annotations/instances_{}{}.json".format(
            dataset_dir, subset, year))
        if subset == "minival" or subset == "valminusminival":
            subset = "val"
        image_dir = "{}/{}{}".format(dataset_dir, subset, year)

        # Load all classes or a subset?
        if not class_ids:
            # All classes
            class_ids = sorted(tab.getCatIds())

        # All images or a subset?
        if class_ids:
            image_ids = []
            for id in class_ids:
                image_ids.extend(list(tab.getImgIds(catIds=[id])))
            # Remove duplicates
            image_ids = list(set(image_ids))
        else:
            # All images
            image_ids = list(tab.imgs.keys())

        # Add classes
        for i in class_ids:
            self.add_class("tab", i, tab.loadCats(i)[0]["name"])

        # Add images
        for i in image_ids:
            self.add_image("tab",
                           image_id=i,
                           path=os.path.join(image_dir,
                                             tab.imgs[i]['file_name']),
                           width=tab.imgs[i]["width"],
                           height=tab.imgs[i]["height"],
                           annotations=tab.loadAnns(
                               tab.getAnnIds(imgIds=[i],
                                             catIds=class_ids,
                                             iscrowd=None)))
        if return_tab:
            return tab

    def load_structure_information_from_boxes(self, boxes):
        start_rows = []
        start_cols = []
        end_rows = []
        end_cols = []
        y1_dict = {}
        y2_dict = {}
        x1_dict = {}
        x2_dict = {}
        struct_indices = {"y1": 1, "x1": 1, "y2": 1, "x2": 1}
        for i in range(len(boxes)):
            cell = boxes[i]
            y1 = cell[0]
            x1 = cell[1]
            y2 = cell[2]
            x2 = cell[3]
            if y1 not in y1_dict:
                y1_dict[y1] = 0
            if y2 not in y2_dict:
                y2_dict[y2] = 0
            if x1 not in x1_dict:
                x1_dict[x1] = 0
            if x2 not in x2_dict:
                x2_dict[x2] = 0
        y1_sorted_list = sorted(y1_dict.keys())
        y2_sorted_list = sorted(y2_dict.keys())
        x1_sorted_list = sorted(x1_dict.keys())
        x2_sorted_list = sorted(x2_dict.keys())
        index = 1
        for val in y1_sorted_list:
            y1_dict[val] = index
            index = index + 1
        index = 1
        for val in y2_sorted_list:
            y2_dict[val] = index
            index = index + 1
        index = 1
        for val in x1_sorted_list:
            x1_dict[val] = index
            index = index + 1
        index = 1
        for val in x2_sorted_list:
            x2_dict[val] = index
            index = index + 1
        for i in range(len(boxes)):
            cell = boxes[i]
            y1 = cell[0]
            x1 = cell[1]
            y2 = cell[2]
            x2 = cell[3]
            y1_index = y1_dict[y1]
            x1_index = x1_dict[x1]
            y2_index = y2_dict[y2]
            x2_index = x2_dict[x2]
            start_rows.append(y1_index)
            start_cols.append(x1_index)
            end_rows.append(y2_index)
            end_cols.append(x2_index)
        #print(boxes, start_rows, start_cols, end_rows, end_cols)
        return np.asarray(start_rows), np.asarray(start_cols), np.asarray(
            end_rows), np.asarray(end_cols)

    def load_cell_structure_information(self, image_id):
        image_info = self.image_info[image_id]
        annotations = self.image_info[image_id]["annotations"]
        start_rows = []
        start_cols = []
        end_rows = []
        end_cols = []
        for annotation in annotations:
            start_rows.append(annotation["start_row"])
            start_cols.append(annotation["start_col"])
            end_rows.append(annotation["end_row"])
            end_cols.append(annotation["end_col"])
        return np.asarray(start_rows), np.asarray(start_cols), np.asarray(
            end_rows), np.asarray(end_cols)

    def load_mask(self, image_id):

        image_info = self.image_info[image_id]
        if image_info["source"] != "tab":
            return super(TabDataset, self).load_mask(image_id)

        instance_masks = []
        class_ids = []
        annotations = self.image_info[image_id]["annotations"]
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        for annotation in annotations:
            class_id = self.map_source_class_id("tab.{}".format(
                annotation['category_id']))
            if class_id:
                m = self.annToMask(annotation, image_info["height"],
                                   image_info["width"])
                # Some objects are so small that they're less than 1 pixel area
                # and end up rounded out. Skip those objects.
                if m.max() < 1:
                    continue
                # Is it a crowd? If so, use a negative class ID.
                if annotation['iscrowd']:
                    # Use negative class ID for crowds
                    class_id *= -1
                    # For crowd masks, annToMask() sometimes returns a mask
                    # smaller than the given dimensions. If so, resize it.
                    if m.shape[0] != image_info["height"] or m.shape[
                            1] != image_info["width"]:
                        m = np.ones(
                            [image_info["height"], image_info["width"]],
                            dtype=bool)
                instance_masks.append(m)
                class_ids.append(class_id)

        # Pack instance masks into an array
        if class_ids:
            mask = np.stack(instance_masks, axis=2).astype(np.bool)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids
        else:
            # Call super class to return an empty mask
            return super(TabDataset, self).load_mask(image_id)

    def annToRLE(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        segm = ann['segmentation']
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, height, width)
            rle = maskUtils.merge(rles)
        elif isinstance(segm['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, height, width)
        else:
            # rle
            rle = ann['segmentation']
        return rle

    def annToMask(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        rle = self.annToRLE(ann, height, width)
        m = maskUtils.decode(rle)
        return m


def build_tabnet_results(dataset, image_ids, rois, class_ids, scores, masks):
    """Arrange resutls to match COCO specs in http://cocodataset.org/#format
    """
    # If no results, return an empty list
    if rois is None:
        return []

    results = []
    for image_id in image_ids:
        # Loop through detections
        for i in range(rois.shape[0]):
            class_id = class_ids[i]
            score = scores[i]
            bbox = np.around(rois[i], 1)
            mask = masks[:, :, i]

            result = {
                "image_id": image_id,
                "category_id": dataset.get_source_class_id(class_id, "tab"),
                "bbox":
                [bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0]],
                "score": score,
                "segmentation": maskUtils.encode(np.asfortranarray(mask))
            }
            results.append(result)
    return results


def evaluate_tabnet(dataset_path,
                    model,
                    dataset,
                    tab,
                    eval_type="bbox",
                    limit=0,
                    image_ids=None):

    image_ids = image_ids or dataset.image_ids

    # Limit to a subset
    if limit:
        image_ids = image_ids[:limit]

    tab_image_ids = [dataset.image_info[id]["id"] for id in image_ids]
    t_prediction = 0
    t_start = time.time()

    results = []
    from shutil import rmtree
    try:
        rmtree(dataset_path + '/result_col_pkl')
    except Exception as e:
        print(e)
        print("cannot remove " + dataset_path + '/result_col_pkl')
    try:
        os.makedirs(dataset_path + '/result_col_pkl')
    except:
        print("cannot create " + dataset_path + '/result_col_pkl')
    try:
        rmtree(dataset_path + '/result_row_pkl')
    except:
        print("cannot remove " + dataset_path + '/result_row_pkl')
    try:
        os.makedirs(dataset_path + '/result_row_pkl')
    except:
        print("cannot create " + dataset_path + '/result_row_pkl')
    try:
        rmtree(dataset_path + '/result_jpg')
    except:
        print("cannot remove " + dataset_path + '/result_jpg')
    try:
        os.makedirs(dataset_path + '/result_jpg')
    except:
        print("cannot create " + dataset_path + '/result_jpg')
    try:
        rmtree(dataset_path + '/result_text')
    except:
        print("cannot remove " + dataset_path + '/result_text')
    try:
        os.makedirs(dataset_path + '/result_text')
    except:
        print("cannot create " + dataset_path + '/result_text')
    for i, image_id in enumerate(image_ids):
        # Load image
        image = dataset.load_image(image_id)
        #image1 = cv2.imread(dataset.image_info[image_id]['path'])
        #image1 = cv2.imread(image)

        # Run detection
        t = time.time()
        r = model.detect([image], verbose=0)[0]
        t_prediction += (time.time() - t)

        image_results = build_tabnet_results(dataset, tab_image_ids[i:i + 1],
                                             r["rois"], r["class_ids"],
                                             r["scores"],
                                             r["masks"].astype(np.uint8))
        row_adj = r["row_adj"]
        col_adj = r["col_adj"]
        results.extend([image_results, row_adj, col_adj])

        #ey_instancesxtra visualiza result
        print("i==>", i)
        #print("box information=>",r['rois'])
        #print("score information=>", r['scores'])
        image_name = "AR_" + str(tab_image_ids[i]) + ".jpg"
        result_path = os.path.join(dataset_path + '/result_jpg/', image_name)
        result_col_path = os.path.join(
            dataset_path + '/result_col_pkl/',
            "AR_" + str(tab_image_ids[i]) + "_col.pkl")
        result_row_path = os.path.join(
            dataset_path + '/result_row_pkl/',
            "AR_" + str(tab_image_ids[i]) + "_row.pkl")
        #visualize.display_results(image, r['rois'], r['masks'], r['class_ids'], dataset.class_names, r['scores'], save_dir='/ssd_scratch/cvit/ajoy/data/coco/result/', img_name = image_name)
        import pickle
        f = open(result_row_path, 'wb')
        pickle.dump(row_adj, f)
        f.close()
        f = open(result_col_path, 'wb')
        pickle.dump(col_adj, f)
        f.close()
        output = visualize.display_instances(image_name,
                                             image,
                                             r['rois'],
                                             r['masks'],
                                             r['class_ids'],
                                             dataset.class_names,
                                             r['scores'],
                                             dataset_path=dataset_path)
        output.figure.savefig(result_path)  # save the figure to file
        #plt.close(output)

        #output=np.asarray(output)
        #output = Image.fromarray(output)
        #output.save('/ssd_scratch/cvit/ajoy/annual_report_table/coco/result/'+image_name,'.jpg')
        #output = np.float32(output)
        #output = cv2.cvtColor(cv2.UMat(output),cv2.COLOR_BGR2RGB)
        #result_path = os.path.join('/ssd_scratch/cvit/ajoy/annual_report_table/coco/result/', image_name)
        #cv2.imwrite(result_path, output)
        #cv2.imshow('output_image', output)
        #cv2.waitKey(10101010101010101010)
        #cv2.destroyAllWindows()

        #plt.savefig("{}/{}.png".format('/ssd_scratch/cvit/ajoy/data/coco/result/', dataset.image_info[image_id]["id"]))
    # Load results. This modifies results with additional attributes.
    tab_results = tab.loadRes(results)

    # Evaluate
    tabEval = COCOeval(tab, tab_results, eval_type)
    tabEval.params.imgIds = tab_image_ids
    tabEval.evaluate()
    tabEval.accumulate()
    tabEval.summarize()

    print("Prediction time: {}. Average {}/image".format(
        t_prediction, t_prediction / len(image_ids)))
    print("Total time: ", time.time() - t_start)


############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train Mask R-CNN on Tables.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'evaluate' on tables")
    parser.add_argument('--dataset',
                        required=True,
                        metavar="/path/to/dataset/",
                        help='Directory of the table dataset')
    parser.add_argument('--year',
                        required=False,
                        default=DEFAULT_DATASET_YEAR,
                        metavar="<year>",
                        help='(keep default=2014)')
    parser.add_argument('--model',
                        required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs',
                        required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--limit',
                        required=False,
                        default=5000,
                        metavar="<image count>",
                        help='Images to use for evaluation (default=500)')

    args = parser.parse_args()
    print("Command: ", args.command)
    print("Model: ", args.model)
    print("Dataset: ", args.dataset)
    print("Year: ", args.year)
    print("Logs: ", args.logs)
    args.logs = args.dataset + "/logs"

    # Configurations
    if args.command == "train":
        config = TabNetConfig()
    else:

        class InferenceConfig(TabNetConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            DETECTION_MIN_CONFIDENCE = 0

        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training",
                                  config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference",
                                  config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.model.lower() == "coco":
        model_path = COCO_MODEL_PATH
    elif args.model.lower() == "last":
        # Find last trained weights
        model_path = model.find_last()
    elif args.model.lower() == "imagenet":
        # Start from ImageNet trained weights
        model_path = model.get_imagenet_weights()
    else:
        model_path = args.model

    # Load weights
    #print("Loading weights ", model_path)
    #model.load_weights(model_path, by_name=True)

    #if no of object class are different than coco
    print("Loading weights ", model_path)
    if args.model.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(model_path,
                           by_name=True,
                           exclude=[
                               "mrcnn_class_logits", "mrcnn_bbox_fc",
                               "mrcnn_bbox", "mrcnn_mask"
                           ])
    else:
        model.load_weights(model_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        # Training dataset. Use the training set and 35K from the
        # validation set, as as in the Mask RCNN paper.
        dataset_train = TabDataset()
        dataset_train.load_tab(args.dataset, "train", year=args.year)
        dataset_train.prepare()

        # Validation dataset
        dataset_val = TabDataset()
        val_type = "val"
        dataset_val.load_tab(args.dataset, val_type, year=args.year)
        dataset_val.prepare()

        # Image Augmentation
        # Right/Left flip 50% of the time
        augmentation = imgaug.augmenters.Fliplr(0.5)

        # *** This training schedule is an example. Update to your needs ***

        # Training - Stage 1
        print("Training network heads")
        model.train(
            dataset_train,
            dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=115,  #20,#40,
            layers='heads',
            augmentation=augmentation)

        # Training - Stage 2
        # Finetune layers from ResNet stage 4 and up
        print("Fine tune Resnet stage 4 and up")
        model.train(
            dataset_train,
            dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=120,  #50, #40, #120,
            layers='4+',
            augmentation=augmentation)

        # Training - Stage 3
        # Fine tune all layers
        print("Fine tune all layers")
        model.train(
            dataset_train,
            dataset_val,
            learning_rate=config.LEARNING_RATE / 10,
            epochs=125,  #80, #40, #80, #160,
            layers='all',
            augmentation=augmentation)

    elif args.command == "evaluate":
        # Validation dataset
        dataset_val = TabDataset()
        val_type = "val"
        tab = dataset_val.load_tab(
            args.dataset,
            val_type,
            year=args.year,
            return_tab=True,
        )
        dataset_val.prepare()
        print("Running TAB evaluation on {} images.".format(args.limit))
        evaluate_tabnet(args.dataset,
                        model,
                        dataset_val,
                        tab,
                        "bbox",
                        limit=int(args.limit))
        #segmentation evaluation
        #evaluate_coco(model, dataset_val, coco, "segm", limit=int(args.limit))
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'evaluate'".format(args.command))
