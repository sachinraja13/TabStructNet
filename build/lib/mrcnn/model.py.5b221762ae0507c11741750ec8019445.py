"""
Mask R-CNN
The main Mask R-CNN model implementation.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""

import os
import random
import datetime
import re
import math
import logging
from collections import OrderedDict
import multiprocessing
import numpy as np
import tensorflow as tf
import keras
import keras.backend as K
import keras.layers as KL
import keras.engine as KE
import keras.models as KM

from mrcnn import utils

# Requires TensorFlow 1.3+ and Keras 2.0.8+.
from distutils.version import LooseVersion
assert LooseVersion(tf.__version__) >= LooseVersion("1.3")
assert LooseVersion(keras.__version__) >= LooseVersion('2.0.8')

############################################################
#  Utility Functions
############################################################


def gauss(x):
    return tf.exp(-1 * x * x)


def gauss_of_lin(x):
    return tf.exp(-1 * (tf.abs(x)))


def euclidean_squared(A, B):
    """
    Returns euclidean distance between two batches of shape [B,N,F] and [B,M,F] where B is batch size, N is number of
    examples in the batch of first set, M is number of examples in the batch of second set, F is number of spatial
    features.
    Returns:
    A matrix of size [B, N, M] where each element [i,j] denotes euclidean distance between ith entry in first set and
    jth in second set.
    """

    shape_A = A.get_shape().as_list()
    shape_B = B.get_shape().as_list()

    assert (A.dtype == tf.float32 or A.dtype
            == tf.float64) and (B.dtype == tf.float32 or B.dtype == tf.float64)
    assert len(shape_A) == 3 and len(shape_B) == 3
    assert shape_A[0] == shape_B[0]  # and shape_A[1] == shape_B[1]

    # Finds euclidean distance using property (a-b)^2 = a^2 + b^2 - 2ab
    sub_factor = -2 * tf.matmul(A, tf.transpose(B, perm=[0, 2, 1
                                                         ]))  # -2ab term
    dotA = tf.expand_dims(tf.reduce_sum(A * A, axis=2), axis=2)  # a^2 term
    dotB = tf.expand_dims(tf.reduce_sum(B * B, axis=2), axis=1)  # b^2 term
    return tf.abs(sub_factor + dotA + dotB)


def log(text, array=None):
    """Prints a text message. And, optionally, if a Numpy array is provided it
    prints it's shape, min, and max values.
    """
    if array is not None:
        text = text.ljust(25)
        text += ("shape: {:20}  ".format(str(array.shape)))
        if array.size:
            text += ("min: {:10.5f}  max: {:10.5f}".format(
                array.min(), array.max()))
        else:
            text += ("min: {:10}  max: {:10}".format("", ""))
        text += "  {}".format(array.dtype)
    print(text)


class BatchNorm(KL.BatchNormalization):
    """Extends the Keras BatchNormalization class to allow a central place
    to make changes if needed.

    Batch normalization has a negative effect on training if batches are small
    so this layer is often frozen (via setting in Config class) and functions
    as linear layer.
    """
    def call(self, inputs, training=None):
        """
        Note about training values:
            None: Train BN layers. This is the normal mode
            False: Freeze BN layers. Good when batch size is small
            True: (don't use). Set layer in training mode even when making inferences
        """
        return super(self.__class__, self).call(inputs, training=training)


def compute_backbone_shapes(config, image_shape):
    """Computes the width and height of each stage of the backbone network.

    Returns:
        [N, (height, width)]. Where N is the number of stages
    """
    if callable(config.BACKBONE):
        return config.COMPUTE_BACKBONE_SHAPE(image_shape)

    # Currently supports ResNet only
    assert config.BACKBONE in ["resnet50", "resnet101"]
    return np.array([[
        int(math.ceil(image_shape[0] / stride)),
        int(math.ceil(image_shape[1] / stride))
    ] for stride in config.BACKBONE_STRIDES])


############################################################
#  Resnet Graph
############################################################

# Code adopted from:
# https://github.com/fchollet/deep-learning-models/blob/master/resnet50.py


def identity_block(input_tensor,
                   kernel_size,
                   filters,
                   stage,
                   block,
                   use_bias=True,
                   train_bn=True):
    """The identity_block is the block that has no conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        use_bias: Boolean. To use or not use a bias in conv layers.
        train_bn: Boolean. Train or freeze Batch Norm layers
    """
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = KL.Conv2D(nb_filter1, (1, 1),
                  name=conv_name_base + '2a',
                  use_bias=use_bias)(input_tensor)
    x = BatchNorm(name=bn_name_base + '2a')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter2, (kernel_size, kernel_size),
                  padding='same',
                  name=conv_name_base + '2b',
                  use_bias=use_bias)(x)
    x = BatchNorm(name=bn_name_base + '2b')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter3, (1, 1),
                  name=conv_name_base + '2c',
                  use_bias=use_bias)(x)
    x = BatchNorm(name=bn_name_base + '2c')(x, training=train_bn)

    x = KL.Add()([x, input_tensor])
    x = KL.Activation('relu', name='res' + str(stage) + block + '_out')(x)
    return x


def conv_block(input_tensor,
               kernel_size,
               filters,
               stage,
               block,
               strides=(2, 2),
               use_bias=True,
               train_bn=True):
    """conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        use_bias: Boolean. To use or not use a bias in conv layers.
        train_bn: Boolean. Train or freeze Batch Norm layers
    Note that from stage 3, the first conv layer at main path is with subsample=(2,2)
    And the shortcut should have subsample=(2,2) as well
    """
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = KL.Conv2D(nb_filter1, (1, 1),
                  strides=strides,
                  name=conv_name_base + '2a',
                  use_bias=use_bias)(input_tensor)
    x = BatchNorm(name=bn_name_base + '2a')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter2, (kernel_size, kernel_size),
                  padding='same',
                  name=conv_name_base + '2b',
                  use_bias=use_bias)(x)
    x = BatchNorm(name=bn_name_base + '2b')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter3, (1, 1),
                  name=conv_name_base + '2c',
                  use_bias=use_bias)(x)
    x = BatchNorm(name=bn_name_base + '2c')(x, training=train_bn)

    shortcut = KL.Conv2D(nb_filter3, (1, 1),
                         strides=strides,
                         name=conv_name_base + '1',
                         use_bias=use_bias)(input_tensor)
    shortcut = BatchNorm(name=bn_name_base + '1')(shortcut, training=train_bn)

    x = KL.Add()([x, shortcut])
    x = KL.Activation('relu', name='res' + str(stage) + block + '_out')(x)
    return x


def resnet_graph(input_image, architecture, stage5=False, train_bn=True):
    """Build a ResNet graph.
        architecture: Can be resnet50 or resnet101
        stage5: Boolean. If False, stage5 of the network is not created
        train_bn: Boolean. Train or freeze Batch Norm layers
    """
    assert architecture in ["resnet50", "resnet101"]
    # Stage 1
    x = KL.ZeroPadding2D((3, 3))(input_image)
    x = KL.Conv2D(64, (7, 7), strides=(2, 2), name='conv1', use_bias=True)(x)
    x = BatchNorm(name='bn_conv1')(x, training=train_bn)
    x = KL.Activation('relu')(x)
    C1 = x = KL.MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)
    # Stage 2
    x = conv_block(x,
                   3, [64, 64, 256],
                   stage=2,
                   block='a',
                   strides=(1, 1),
                   train_bn=train_bn)
    x = identity_block(x,
                       3, [64, 64, 256],
                       stage=2,
                       block='b',
                       train_bn=train_bn)
    C2 = x = identity_block(x,
                            3, [64, 64, 256],
                            stage=2,
                            block='c',
                            train_bn=train_bn)
    # Stage 3
    x = conv_block(x,
                   3, [128, 128, 512],
                   stage=3,
                   block='a',
                   train_bn=train_bn)
    x = identity_block(x,
                       3, [128, 128, 512],
                       stage=3,
                       block='b',
                       train_bn=train_bn)
    x = identity_block(x,
                       3, [128, 128, 512],
                       stage=3,
                       block='c',
                       train_bn=train_bn)
    C3 = x = identity_block(x,
                            3, [128, 128, 512],
                            stage=3,
                            block='d',
                            train_bn=train_bn)
    # Stage 4
    x = conv_block(x,
                   3, [256, 256, 1024],
                   stage=4,
                   block='a',
                   train_bn=train_bn)
    block_count = {"resnet50": 5, "resnet101": 22}[architecture]
    for i in range(block_count):
        x = identity_block(x,
                           3, [256, 256, 1024],
                           stage=4,
                           block=chr(98 + i),
                           train_bn=train_bn)
    C4 = x
    # Stage 5
    if stage5:
        x = conv_block(x,
                       3, [512, 512, 2048],
                       stage=5,
                       block='a',
                       train_bn=train_bn)
        x = identity_block(x,
                           3, [512, 512, 2048],
                           stage=5,
                           block='b',
                           train_bn=train_bn)
        C5 = x = identity_block(x,
                                3, [512, 512, 2048],
                                stage=5,
                                block='c',
                                train_bn=train_bn)
    else:
        C5 = None
    return [C1, C2, C3, C4, C5]


############################################################
#  Proposal Layer
############################################################


def apply_box_deltas_graph(boxes, deltas):
    """Applies the given deltas to the given boxes.
    boxes: [N, (y1, x1, y2, x2)] boxes to update
    deltas: [N, (dy, dx, log(dh), log(dw))] refinements to apply
    """
    # Convert to y, x, h, w
    height = boxes[:, 2] - boxes[:, 0]
    width = boxes[:, 3] - boxes[:, 1]
    center_y = boxes[:, 0] + 0.5 * height
    center_x = boxes[:, 1] + 0.5 * width
    # Apply deltas
    center_y += deltas[:, 0] * height
    center_x += deltas[:, 1] * width
    height *= tf.exp(deltas[:, 2])
    width *= tf.exp(deltas[:, 3])
    # Convert back to y1, x1, y2, x2
    y1 = center_y - 0.5 * height
    x1 = center_x - 0.5 * width
    y2 = y1 + height
    x2 = x1 + width
    result = tf.stack([y1, x1, y2, x2], axis=1, name="apply_box_deltas_out")
    return result


def clip_boxes_graph(boxes, window):
    """
    boxes: [N, (y1, x1, y2, x2)]
    window: [4] in the form y1, x1, y2, x2
    """
    # Split
    wy1, wx1, wy2, wx2 = tf.split(window, 4)
    y1, x1, y2, x2 = tf.split(boxes, 4, axis=1)
    # Clip
    y1 = tf.maximum(tf.minimum(y1, wy2), wy1)
    x1 = tf.maximum(tf.minimum(x1, wx2), wx1)
    y2 = tf.maximum(tf.minimum(y2, wy2), wy1)
    x2 = tf.maximum(tf.minimum(x2, wx2), wx1)
    clipped = tf.concat([y1, x1, y2, x2], axis=1, name="clipped_boxes")
    clipped.set_shape((clipped.shape[0], 4))
    return clipped


class ProposalLayer(KE.Layer):
    """Receives anchor scores and selects a subset to pass as proposals
    to the second stage. Filtering is done based on anchor scores and
    non-max suppression to remove overlaps. It also applies bounding
    box refinement deltas to anchors.

    Inputs:
        rpn_probs: [batch, num_anchors, (bg prob, fg prob)]
        rpn_bbox: [batch, num_anchors, (dy, dx, log(dh), log(dw))]
        anchors: [batch, num_anchors, (y1, x1, y2, x2)] anchors in normalized coordinates

    Returns:
        Proposals in normalized coordinates [batch, rois, (y1, x1, y2, x2)]
    """
    def __init__(self, proposal_count, nms_threshold, config=None, **kwargs):
        super(ProposalLayer, self).__init__(**kwargs)
        self.config = config
        self.proposal_count = proposal_count
        self.nms_threshold = nms_threshold

    def call(self, inputs):
        # Box Scores. Use the foreground class confidence. [Batch, num_rois, 1]
        scores = inputs[0][:, :, 1]
        # Box deltas [batch, num_rois, 4]
        deltas = inputs[1]
        deltas = deltas * np.reshape(self.config.RPN_BBOX_STD_DEV, [1, 1, 4])
        # Anchors
        anchors = inputs[2]

        # Improve performance by trimming to top anchors by score
        # and doing the rest on the smaller subset.
        pre_nms_limit = tf.minimum(self.config.PRE_NMS_LIMIT,
                                   tf.shape(anchors)[1])
        ix = tf.nn.top_k(scores,
                         pre_nms_limit,
                         sorted=True,
                         name="top_anchors").indices
        scores = utils.batch_slice([scores, ix], lambda x, y: tf.gather(x, y),
                                   self.config.IMAGES_PER_GPU)
        deltas = utils.batch_slice([deltas, ix], lambda x, y: tf.gather(x, y),
                                   self.config.IMAGES_PER_GPU)
        pre_nms_anchors = utils.batch_slice([anchors, ix],
                                            lambda a, x: tf.gather(a, x),
                                            self.config.IMAGES_PER_GPU,
                                            names=["pre_nms_anchors"])

        # Apply deltas to anchors to get refined anchors.
        # [batch, N, (y1, x1, y2, x2)]
        boxes = utils.batch_slice([pre_nms_anchors, deltas],
                                  lambda x, y: apply_box_deltas_graph(x, y),
                                  self.config.IMAGES_PER_GPU,
                                  names=["refined_anchors"])

        # Clip to image boundaries. Since we're in normalized coordinates,
        # clip to 0..1 range. [batch, N, (y1, x1, y2, x2)]
        window = np.array([0, 0, 1, 1], dtype=np.float32)
        boxes = utils.batch_slice(boxes,
                                  lambda x: clip_boxes_graph(x, window),
                                  self.config.IMAGES_PER_GPU,
                                  names=["refined_anchors_clipped"])

        # Filter out small boxes
        # According to Xinlei Chen's paper, this reduces detection accuracy
        # for small objects, so we're skipping it.

        # Non-max suppression
        def nms(boxes, scores):
            indices = tf.image.non_max_suppression(
                boxes,
                scores,
                self.proposal_count,
                self.nms_threshold,
                name="rpn_non_max_suppression")
            proposals = tf.gather(boxes, indices)
            # Pad if needed
            padding = tf.maximum(self.proposal_count - tf.shape(proposals)[0],
                                 0)
            proposals = tf.pad(proposals, [(0, padding), (0, 0)])
            return proposals

        proposals = utils.batch_slice([boxes, scores], nms,
                                      self.config.IMAGES_PER_GPU)
        return proposals

    def compute_output_shape(self, input_shape):
        return (None, self.proposal_count, 4)


############################################################
#  ROIAlign Layer
############################################################


def log2_graph(x):
    """Implementation of Log2. TF doesn't have a native implementation."""
    return tf.log(x) / tf.log(2.0)


class PyramidROIAlign(KE.Layer):
    """Implements ROI Pooling on multiple levels of the feature pyramid.

    Params:
    - pool_shape: [pool_height, pool_width] of the output pooled regions. Usually [7, 7]

    Inputs:
    - boxes: [batch, num_boxes, (y1, x1, y2, x2)] in normalized
             coordinates. Possibly padded with zeros if not enough
             boxes to fill the array.
    - image_meta: [batch, (meta data)] Image details. See compose_image_meta()
    - feature_maps: List of feature maps from different levels of the pyramid.
                    Each is [batch, height, width, channels]

    Output:
    Pooled regions in the shape: [batch, num_boxes, pool_height, pool_width, channels].
    The width and height are those specific in the pool_shape in the layer
    constructor.
    """
    def __init__(self, pool_shape, **kwargs):
        super(PyramidROIAlign, self).__init__(**kwargs)
        self.pool_shape = tuple(pool_shape)

    def call(self, inputs):
        # Crop boxes [batch, num_boxes, (y1, x1, y2, x2)] in normalized coords
        boxes = inputs[0]

        # Image meta
        # Holds details about the image. See compose_image_meta()
        image_meta = inputs[1]

        # Feature Maps. List of feature maps from different level of the
        # feature pyramid. Each is [batch, height, width, channels]
        feature_maps = inputs[2:]

        # Assign each ROI to a level in the pyramid based on the ROI area.
        y1, x1, y2, x2 = tf.split(boxes, 4, axis=2)
        h = y2 - y1
        w = x2 - x1
        # Use shape of first image. Images in a batch must have the same size.
        image_shape = parse_image_meta_graph(image_meta)['image_shape'][0]
        # Equation 1 in the Feature Pyramid Networks paper. Account for
        # the fact that our coordinates are normalized here.
        # e.g. a 224x224 ROI (in pixels) maps to P4
        image_area = tf.cast(image_shape[0] * image_shape[1], tf.float32)
        roi_level = log2_graph(tf.sqrt(h * w) / (224.0 / tf.sqrt(image_area)))
        roi_level = tf.minimum(
            5, tf.maximum(2, 4 + tf.cast(tf.round(roi_level), tf.int32)))
        roi_level = tf.squeeze(roi_level, 2)

        # Loop through levels and apply ROI pooling to each. P2 to P5.
        pooled = []
        box_to_level = []
        for i, level in enumerate(range(2, 6)):
            ix = tf.where(tf.equal(roi_level, level))
            level_boxes = tf.gather_nd(boxes, ix)

            # Box indices for crop_and_resize.
            box_indices = tf.cast(ix[:, 0], tf.int32)

            # Keep track of which box is mapped to which level
            box_to_level.append(ix)

            # Stop gradient propogation to ROI proposals
            level_boxes = tf.stop_gradient(level_boxes)
            box_indices = tf.stop_gradient(box_indices)

            # Crop and Resize
            # From Mask R-CNN paper: "We sample four regular locations, so
            # that we can evaluate either max or average pooling. In fact,
            # interpolating only a single value at each bin center (without
            # pooling) is nearly as effective."
            #
            # Here we use the simplified approach of a single value per bin,
            # which is how it's done in tf.crop_and_resize()
            # Result: [batch * num_boxes, pool_height, pool_width, channels]
            pooled.append(
                tf.image.crop_and_resize(feature_maps[i],
                                         level_boxes,
                                         box_indices,
                                         self.pool_shape,
                                         method="bilinear"))

        # Pack pooled features into one tensor
        pooled = tf.concat(pooled, axis=0)

        # Pack box_to_level mapping into one array and add another
        # column representing the order of pooled boxes
        box_to_level = tf.concat(box_to_level, axis=0)
        box_range = tf.expand_dims(tf.range(tf.shape(box_to_level)[0]), 1)
        box_to_level = tf.concat([tf.cast(box_to_level, tf.int32), box_range],
                                 axis=1)

        # Rearrange pooled features to match the order of the original boxes
        # Sort box_to_level by batch then box index
        # TF doesn't have a way to sort by two columns, so merge them and sort.
        sorting_tensor = box_to_level[:, 0] * 100000 + box_to_level[:, 1]
        ix = tf.nn.top_k(sorting_tensor,
                         k=tf.shape(box_to_level)[0]).indices[::-1]
        ix = tf.gather(box_to_level[:, 2], ix)
        pooled = tf.gather(pooled, ix)

        # Re-add the batch dimension
        shape = tf.concat([tf.shape(boxes)[:2], tf.shape(pooled)[1:]], axis=0)
        pooled = tf.reshape(pooled, shape)
        return pooled

    def compute_output_shape(self, input_shape):
        return input_shape[0][:2] + self.pool_shape + (input_shape[2][-1], )


############################################################
#  Detection Target Layer
############################################################


def overlaps_graph(boxes1, boxes2):
    """Computes IoU overlaps between two sets of boxes.
    boxes1, boxes2: [N, (y1, x1, y2, x2)].
    """
    # 1. Tile boxes2 and repeat boxes1. This allows us to compare
    # every boxes1 against every boxes2 without loops.
    # TF doesn't have an equivalent to np.repeat() so simulate it
    # using tf.tile() and tf.reshape.
    b1 = tf.reshape(
        tf.tile(tf.expand_dims(boxes1, 1), [1, 1, tf.shape(boxes2)[0]]),
        [-1, 4])
    b2 = tf.tile(boxes2, [tf.shape(boxes1)[0], 1])
    # 2. Compute intersections
    b1_y1, b1_x1, b1_y2, b1_x2 = tf.split(b1, 4, axis=1)
    b2_y1, b2_x1, b2_y2, b2_x2 = tf.split(b2, 4, axis=1)
    y1 = tf.maximum(b1_y1, b2_y1)
    x1 = tf.maximum(b1_x1, b2_x1)
    y2 = tf.minimum(b1_y2, b2_y2)
    x2 = tf.minimum(b1_x2, b2_x2)
    intersection = tf.maximum(x2 - x1, 0) * tf.maximum(y2 - y1, 0)
    # 3. Compute unions
    b1_area = (b1_y2 - b1_y1) * (b1_x2 - b1_x1)
    b2_area = (b2_y2 - b2_y1) * (b2_x2 - b2_x1)
    union = b1_area + b2_area - intersection
    # 4. Compute IoU and reshape to [boxes1, boxes2]
    iou = intersection / union
    overlaps = tf.reshape(iou, [tf.shape(boxes1)[0], tf.shape(boxes2)[0]])
    return overlaps


def detection_targets_graph(proposals, gt_class_ids, gt_boxes, gt_masks,
                            config):
    """Generates detection targets for one image. Subsamples proposals and
    generates target class IDs, bounding box deltas, and masks for each.

    Inputs:
    proposals: [POST_NMS_ROIS_TRAINING, (y1, x1, y2, x2)] in normalized coordinates. Might
               be zero padded if there are not enough proposals.
    gt_class_ids: [MAX_GT_INSTANCES] int class IDs
    gt_boxes: [MAX_GT_INSTANCES, (y1, x1, y2, x2)] in normalized coordinates.
    gt_masks: [height, width, MAX_GT_INSTANCES] of boolean type.

    Returns: Target ROIs and corresponding class IDs, bounding box shifts,
    and masks.
    rois: [TRAIN_ROIS_PER_IMAGE, (y1, x1, y2, x2)] in normalized coordinates
    class_ids: [TRAIN_ROIS_PER_IMAGE]. Integer class IDs. Zero padded.
    deltas: [TRAIN_ROIS_PER_IMAGE, (dy, dx, log(dh), log(dw))]
    masks: [TRAIN_ROIS_PER_IMAGE, height, width]. Masks cropped to bbox
           boundaries and resized to neural network output size.

    Note: Returned arrays might be zero padded if not enough target ROIs.
    """
    # Assertions
    asserts = [
        tf.Assert(tf.greater(tf.shape(proposals)[0], 0), [proposals],
                  name="roi_assertion"),
    ]
    with tf.control_dependencies(asserts):
        proposals = tf.identity(proposals)

    # Remove zero padding
    proposals, _ = trim_zeros_graph(proposals, name="trim_proposals")
    gt_boxes, non_zeros = trim_zeros_graph(gt_boxes, name="trim_gt_boxes")
    gt_class_ids = tf.boolean_mask(gt_class_ids,
                                   non_zeros,
                                   name="trim_gt_class_ids")
    gt_masks = tf.gather(gt_masks,
                         tf.where(non_zeros)[:, 0],
                         axis=2,
                         name="trim_gt_masks")

    # Handle COCO crowds
    # A crowd box in COCO is a bounding box around several instances. Exclude
    # them from training. A crowd box is given a negative class ID.
    crowd_ix = tf.where(gt_class_ids < 0)[:, 0]
    non_crowd_ix = tf.where(gt_class_ids > 0)[:, 0]
    crowd_boxes = tf.gather(gt_boxes, crowd_ix)
    gt_class_ids = tf.gather(gt_class_ids, non_crowd_ix)
    gt_boxes = tf.gather(gt_boxes, non_crowd_ix)
    gt_masks = tf.gather(gt_masks, non_crowd_ix, axis=2)

    # Compute overlaps matrix [proposals, gt_boxes]
    overlaps = overlaps_graph(proposals, gt_boxes)

    # Compute overlaps with crowd boxes [proposals, crowd_boxes]
    crowd_overlaps = overlaps_graph(proposals, crowd_boxes)
    crowd_iou_max = tf.reduce_max(crowd_overlaps, axis=1)
    no_crowd_bool = (crowd_iou_max < 0.001)

    # Determine positive and negative ROIs
    roi_iou_max = tf.reduce_max(overlaps, axis=1)
    # 1. Positive ROIs are those with >= 0.5 IoU with a GT box
    positive_roi_bool = (roi_iou_max >= 0.5)
    positive_indices = tf.where(positive_roi_bool)[:, 0]
    # 2. Negative ROIs are those with < 0.5 with every GT box. Skip crowds.
    negative_indices = tf.where(
        tf.logical_and(roi_iou_max < 0.5, no_crowd_bool))[:, 0]

    # Subsample ROIs. Aim for 33% positive
    # Positive ROIs
    positive_count = int(config.TRAIN_ROIS_PER_IMAGE *
                         config.ROI_POSITIVE_RATIO)
    positive_indices = tf.random_shuffle(positive_indices)[:positive_count]
    positive_count = tf.shape(positive_indices)[0]
    # Negative ROIs. Add enough to maintain positive:negative ratio.
    r = 1.0 / config.ROI_POSITIVE_RATIO
    negative_count = tf.cast(r * tf.cast(positive_count, tf.float32),
                             tf.int32) - positive_count
    negative_indices = tf.random_shuffle(negative_indices)[:negative_count]
    # Gather selected ROIs
    positive_rois = tf.gather(proposals, positive_indices)
    negative_rois = tf.gather(proposals, negative_indices)

    # Assign positive ROIs to GT boxes.
    positive_overlaps = tf.gather(overlaps, positive_indices)
    roi_gt_box_assignment = tf.cond(
        tf.greater(tf.shape(positive_overlaps)[1], 0),
        true_fn=lambda: tf.argmax(positive_overlaps, axis=1),
        false_fn=lambda: tf.cast(tf.constant([]), tf.int64))
    roi_gt_boxes = tf.gather(gt_boxes, roi_gt_box_assignment)
    roi_gt_class_ids = tf.gather(gt_class_ids, roi_gt_box_assignment)

    # Compute bbox refinement for positive ROIs
    deltas = utils.box_refinement_graph(positive_rois, roi_gt_boxes)
    deltas /= config.BBOX_STD_DEV

    # Assign positive ROIs to GT masks
    # Permute masks to [N, height, width, 1]
    transposed_masks = tf.expand_dims(tf.transpose(gt_masks, [2, 0, 1]), -1)
    # Pick the right mask for each ROI
    roi_masks = tf.gather(transposed_masks, roi_gt_box_assignment)

    # Compute mask targets
    boxes = positive_rois
    if config.USE_MINI_MASK:
        # Transform ROI coordinates from normalized image space
        # to normalized mini-mask space.
        y1, x1, y2, x2 = tf.split(positive_rois, 4, axis=1)
        gt_y1, gt_x1, gt_y2, gt_x2 = tf.split(roi_gt_boxes, 4, axis=1)
        gt_h = gt_y2 - gt_y1
        gt_w = gt_x2 - gt_x1
        y1 = (y1 - gt_y1) / gt_h
        x1 = (x1 - gt_x1) / gt_w
        y2 = (y2 - gt_y1) / gt_h
        x2 = (x2 - gt_x1) / gt_w
        boxes = tf.concat([y1, x1, y2, x2], 1)
    box_ids = tf.range(0, tf.shape(roi_masks)[0])
    masks = tf.image.crop_and_resize(tf.cast(roi_masks, tf.float32), boxes,
                                     box_ids, config.MASK_SHAPE)
    # Remove the extra dimension from masks.
    masks = tf.squeeze(masks, axis=3)

    # Threshold mask pixels at 0.5 to have GT masks be 0 or 1 to use with
    # binary cross entropy loss.
    masks = tf.round(masks)

    # Append negative ROIs and pad bbox deltas and masks that
    # are not used for negative ROIs with zeros.
    rois = tf.concat([positive_rois, negative_rois], axis=0)
    N = tf.shape(negative_rois)[0]
    P = tf.maximum(config.TRAIN_ROIS_PER_IMAGE - tf.shape(rois)[0], 0)
    rois = tf.pad(rois, [(0, P), (0, 0)])
    roi_gt_class_ids = tf.pad(roi_gt_class_ids, [(0, N + P)])
    deltas = tf.pad(deltas, [(0, N + P), (0, 0)])
    masks = tf.pad(masks, [[0, N + P], (0, 0), (0, 0)])
    return rois, roi_gt_class_ids, deltas, masks


class DetectionTargetLayer(KE.Layer):
    """Subsamples proposals and generates target box refinement, class_ids,
    and masks for each.

    Inputs:
    proposals: [batch, N, (y1, x1, y2, x2)] in normalized coordinates. Might
               be zero padded if there are not enough proposals.
    gt_class_ids: [batch, MAX_GT_INSTANCES] Integer class IDs.
    gt_boxes: [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)] in normalized
              coordinates.
    gt_masks: [batch, height, width, MAX_GT_INSTANCES] of boolean type

    Returns: Target ROIs and corresponding class IDs, bounding box shifts,
    and masks.
    rois: [batch, TRAIN_ROIS_PER_IMAGE, (y1, x1, y2, x2)] in normalized
          coordinates
    target_class_ids: [batch, TRAIN_ROIS_PER_IMAGE]. Integer class IDs.
    target_deltas: [batch, TRAIN_ROIS_PER_IMAGE, (dy, dx, log(dh), log(dw)]
    target_mask: [batch, TRAIN_ROIS_PER_IMAGE, height, width]
                 Masks cropped to bbox boundaries and resized to neural
                 network output size.

    Note: Returned arrays might be zero padded if not enough target ROIs.
    """
    def __init__(self, config, **kwargs):
        super(DetectionTargetLayer, self).__init__(**kwargs)
        self.config = config

    def call(self, inputs):
        proposals = inputs[0]
        gt_class_ids = inputs[1]
        gt_boxes = inputs[2]
        gt_masks = inputs[3]

        # Slice the batch and run a graph for each slice
        # TODO: Rename target_bbox to target_deltas for clarity
        names = ["rois", "target_class_ids", "target_bbox", "target_mask"]
        outputs = utils.batch_slice(
            [proposals, gt_class_ids, gt_boxes, gt_masks],
            lambda w, x, y, z: detection_targets_graph(w, x, y, z, self.config
                                                       ),
            self.config.IMAGES_PER_GPU,
            names=names)
        return outputs

    def compute_output_shape(self, input_shape):
        return [
            (None, self.config.TRAIN_ROIS_PER_IMAGE, 4),  # rois
            (None, self.config.TRAIN_ROIS_PER_IMAGE),  # class_ids
            (None, self.config.TRAIN_ROIS_PER_IMAGE, 4),  # deltas
            (None, self.config.TRAIN_ROIS_PER_IMAGE, self.config.MASK_SHAPE[0],
             self.config.MASK_SHAPE[1])  # masks
        ]

    def compute_mask(self, inputs, mask=None):
        return [None, None, None, None]


############################################################
# Structure Detection Layer
############################################################


def refine_structure_detections_graph(image_features,
                                      rois,
                                      probs,
                                      deltas,
                                      window,
                                      gt_class_ids,
                                      gt_boxes,
                                      gt_start_rows,
                                      gt_start_cols,
                                      gt_end_rows,
                                      gt_end_cols,
                                      config,
                                      scale=0.25):
    """Refine classified proposals and filter overlaps and return final
    detections.

    Inputs:
        rois: [N, (y1, x1, y2, x2)] in normalized coordinates
        probs: [N, num_classes]. Class probabilities.
        deltas: [N, num_classes, (dy, dx, log(dh), log(dw))]. Class-specific
                bounding box deltas.
        window: (y1, x1, y2, x2) in normalized coordinates. The part of the image
            that contains the image excluding the padding.

    Returns detections shaped: [num_detections, (y1, x1, y2, x2, class_id, score)] where
        coordinates are normalized.
    """
    # Class IDs per ROI
    class_ids = tf.argmax(probs, axis=1, output_type=tf.int32)
    # Class probability of the top class of each ROI
    indices = tf.stack([tf.range(probs.shape[0]), class_ids], axis=1)
    class_scores = tf.gather_nd(probs, indices)
    # Class-specific bounding box deltas
    deltas_specific = tf.gather_nd(deltas, indices)
    # Apply bounding box deltas
    # Shape: [boxes, (y1, x1, y2, x2)] in normalized coordinates
    refined_rois = apply_box_deltas_graph(
        rois, deltas_specific * config.BBOX_STD_DEV)
    refined_rois = clip_boxes_graph(refined_rois, window)
    # Clip boxes to image window
    #refined_rois = clip_boxes_graph(refined_rois, window)

    # TODO: Filter out boxes with zero area

    # Filter out background boxes
    keep = tf.where(class_ids > 0)[:, 0]
    # Filter out low confidence boxes
    #if config.DETECTION_MIN_CONFIDENCE:
    #    conf_keep = tf.where(class_scores >= 0.0)[:, 0]
    #    keep = tf.sets.set_intersection(tf.expand_dims(keep, 0),
    #                                    tf.expand_dims(conf_keep, 0))
    #    keep = tf.sparse_tensor_to_dense(keep)[0]

    # Apply per-class NMS
    # 1. Prepare variables
    pre_nms_class_ids = tf.gather(class_ids, keep)
    pre_nms_scores = tf.gather(class_scores, keep)
    pre_nms_rois = tf.gather(refined_rois, keep)
    unique_pre_nms_class_ids = tf.unique(pre_nms_class_ids)[0]

    def nms_keep_map(class_id):
        """Apply Non-Maximum Suppression on ROIs of the given class."""
        # Indices of ROIs of the given class
        ixs = tf.where(tf.equal(pre_nms_class_ids, class_id))[:, 0]
        # Apply NMS
        class_keep = tf.image.non_max_suppression(
            tf.gather(pre_nms_rois, ixs),
            tf.gather(pre_nms_scores, ixs),
            max_output_size=config.TRAIN_ROIS_PER_IMAGE,
            iou_threshold=config.DETECTION_NMS_THRESHOLD)
        # Map indices
        class_keep = tf.gather(keep, tf.gather(ixs, class_keep))
        # Pad with -1 so returned tensors have the same shape
        gap = config.TRAIN_ROIS_PER_IMAGE - tf.shape(class_keep)[0]
        class_keep = tf.pad(class_keep, [(0, gap)],
                            mode='CONSTANT',
                            constant_values=-1)
        # Set shape so map_fn() can infer result shape
        class_keep.set_shape([config.TRAIN_ROIS_PER_IMAGE])
        return class_keep

    # 2. Map over class IDs
    nms_keep = tf.map_fn(nms_keep_map,
                         unique_pre_nms_class_ids,
                         dtype=tf.int64)
    # 3. Merge results into one list, and remove -1 padding
    nms_keep = tf.reshape(nms_keep, [-1])
    nms_keep = tf.gather(nms_keep, tf.where(nms_keep > -1)[:, 0])
    # 4. Compute intersection between keep and nms_keep
    keep = tf.sets.set_intersection(tf.expand_dims(keep, 0),
                                    tf.expand_dims(nms_keep, 0))
    keep = tf.sparse_tensor_to_dense(keep)[0]
    # Keep top detections
    roi_count = config.TRAIN_ROIS_PER_IMAGE
    class_scores_keep = tf.gather(class_scores, keep)
    num_keep = tf.minimum(tf.shape(class_scores_keep)[0], roi_count)
    top_ids = tf.nn.top_k(class_scores_keep, k=num_keep, sorted=True)[1]
    keep = tf.gather(keep, top_ids)
    #detected_regions = refined_rois
    detected_regions = tf.gather(refined_rois, keep)
    detected_regions, _ = trim_zeros_graph(detected_regions,
                                           name="trim_proposals")
    gt_boxes, non_zeros = trim_zeros_graph(gt_boxes, name="trim_gt_boxes")
    gt_class_ids = tf.boolean_mask(gt_class_ids,
                                   non_zeros,
                                   name="trim_gt_class_ids")

    crowd_ix = tf.where(gt_class_ids < 0)[:, 0]
    non_crowd_ix = tf.where(gt_class_ids > 0)[:, 0]
    crowd_boxes = tf.gather(gt_boxes, crowd_ix)
    gt_class_ids = tf.gather(gt_class_ids, non_crowd_ix)
    gt_boxes = tf.gather(gt_boxes, non_crowd_ix)

    # Compute overlaps matrix [proposals, gt_boxes]
    overlaps = overlaps_graph(detected_regions, gt_boxes)

    # Compute overlaps with crowd boxes [proposals, crowd_boxes]
    crowd_overlaps = overlaps_graph(detected_regions, crowd_boxes)
    crowd_iou_max = tf.reduce_max(crowd_overlaps, axis=1)
    no_crowd_bool = (crowd_iou_max < 0.001)

    # Determine positive and negative ROIs
    roi_iou_max = tf.reduce_max(overlaps, axis=1)
    # 1. Positive ROIs are those with >= 0.5 IoU with a GT box
    positive_roi_bool = (roi_iou_max >= 0.2)
    positive_indices = tf.where(positive_roi_bool)[:, 0]
    # 2. Negative ROIs are those with < 0.5 with every GT box. Skip crowds.
    positive_overlaps = tf.gather(overlaps, positive_indices)
    roi_gt_box_assignment = tf.cond(
        tf.greater(tf.shape(positive_overlaps)[1], 0),
        true_fn=lambda: tf.argmax(positive_overlaps, axis=1),
        false_fn=lambda: tf.cast(tf.constant([]), tf.int64))
    positive_preds = tf.gather(detected_regions, positive_indices)
    roi_gt_boxes = tf.gather(gt_boxes, roi_gt_box_assignment)

    roi_gt_start_rows = tf.gather(gt_start_rows, roi_gt_box_assignment)
    roi_gt_start_cols = tf.gather(gt_start_cols, roi_gt_box_assignment)
    roi_gt_end_rows = tf.gather(gt_end_rows, roi_gt_box_assignment)
    roi_gt_end_cols = tf.gather(gt_end_cols, roi_gt_box_assignment)
    adj_rows = tf.math.logical_or(
        tf.math.logical_and(
            tf.math.logical_and(
                tf.math.greater(
                    tf.expand_dims(roi_gt_start_rows, 0) *
                    tf.expand_dims(roi_gt_start_rows, 1), tf.constant(0)),
                tf.math.greater_equal(tf.expand_dims(roi_gt_start_rows, 0),
                                      tf.expand_dims(roi_gt_start_rows, 1))),
            tf.math.logical_and(
                tf.math.greater(
                    tf.expand_dims(roi_gt_end_rows, 0) *
                    tf.expand_dims(roi_gt_end_rows, 1), tf.constant(0)),
                tf.math.less_equal(tf.expand_dims(roi_gt_end_rows, 0),
                                   tf.expand_dims(roi_gt_end_rows, 1)))),
        tf.math.logical_and(
            tf.math.logical_and(
                tf.math.greater(
                    tf.expand_dims(roi_gt_start_rows, 0) *
                    tf.expand_dims(roi_gt_start_rows, 1), tf.constant(0)),
                tf.math.less_equal(tf.expand_dims(roi_gt_start_rows, 0),
                                   tf.expand_dims(roi_gt_start_rows, 1))),
            tf.math.logical_and(
                tf.math.greater(
                    tf.expand_dims(roi_gt_end_rows, 0) *
                    tf.expand_dims(roi_gt_end_rows, 1), tf.constant(0)),
                tf.math.greater_equal(tf.expand_dims(roi_gt_end_rows, 0),
                                      tf.expand_dims(roi_gt_end_rows, 1)))))
    adj_rows = tf.cast(adj_rows, tf.float32)
    adj_cols = tf.math.logical_or(
        tf.math.logical_and(
            tf.math.logical_and(
                tf.math.greater(
                    tf.expand_dims(roi_gt_start_cols, 0) *
                    tf.expand_dims(roi_gt_start_cols, 1), tf.constant(0)),
                tf.math.greater_equal(tf.expand_dims(roi_gt_start_cols, 0),
                                      tf.expand_dims(roi_gt_start_cols, 1))),
            tf.math.logical_and(
                tf.math.greater(
                    tf.expand_dims(roi_gt_end_cols, 0) *
                    tf.expand_dims(roi_gt_end_cols, 1), tf.constant(0)),
                tf.math.less_equal(tf.expand_dims(roi_gt_end_cols, 0),
                                   tf.expand_dims(roi_gt_end_cols, 1)))),
        tf.math.logical_and(
            tf.math.logical_and(
                tf.math.greater(
                    tf.expand_dims(roi_gt_start_cols, 0) *
                    tf.expand_dims(roi_gt_start_cols, 1), tf.constant(0)),
                tf.math.less_equal(tf.expand_dims(roi_gt_start_cols, 0),
                                   tf.expand_dims(roi_gt_start_cols, 1))),
            tf.math.logical_and(
                tf.math.greater(
                    tf.expand_dims(roi_gt_end_cols, 0) *
                    tf.expand_dims(roi_gt_end_cols, 1), tf.constant(0)),
                tf.math.greater_equal(tf.expand_dims(roi_gt_end_cols, 0),
                                      tf.expand_dims(roi_gt_end_cols, 1)))))
    adj_cols = tf.cast(adj_cols, tf.float32)
    print(image_features.shape)
    conv_head_y = image_features
    conv_head_x = K.permute_dimensions(image_features, [1, 0, 2])
    gt_y1, gt_x1, gt_y2, gt_x2 = tf.split(roi_gt_boxes, 4, axis=1)
    roi_gt_boxes = tf.concat([gt_y1, gt_x1, gt_y2, gt_x2], 1)
    pred_boxes_denorm = denorm_boxes_graph(positive_preds, [256, 256])
    pred_boxes_denorm = tf.cast(pred_boxes_denorm, tf.float32)
    vertices_y, vertices_x, vertices_y2, vertices_x2 = tf.split(
        pred_boxes_denorm, 4, axis=1)
    vertices_y = tf.squeeze(vertices_y)
    vertices_x = tf.squeeze(vertices_x)
    vertices_y2 = tf.squeeze(vertices_y2)
    vertices_x2 = tf.squeeze(vertices_x2)
    vertices_x_mid = (vertices_x + vertices_x2) / 2
    vertices_y_mid = (vertices_y + vertices_y2) / 2
    vertices_y = tf.cast(vertices_y, tf.int32)
    vertices_x = tf.cast(vertices_x, tf.int32)
    vertices_y2 = tf.cast(vertices_y2, tf.int32)
    vertices_x2 = tf.cast(vertices_x2, tf.int32)
    vertices_x_mid = tf.cast(vertices_x_mid, tf.int32)
    vertices_y_mid = tf.cast(vertices_y_mid, tf.int32)
    vertices_x_mid = tf.reshape(vertices_x_mid,
                                [tf.shape(pred_boxes_denorm)[0], 1])
    vertices_y_mid = tf.reshape(vertices_y_mid,
                                [tf.shape(pred_boxes_denorm)[0], 1])
    #vertices_range = tf.range(0, tf.shape(pred_boxes_denorm)[0])[..., tf.newaxis]
    #vertices_range = tf.cast(vertices_range, tf.float32)
    indexing_tensor = tf.concat((vertices_y_mid, vertices_x_mid), axis=-1)
    indexing_tensor = tf.cast(indexing_tensor, tf.int32)
    indexing_tensor_x_mid = tf.cast(vertices_x_mid, tf.int32)
    indexing_tensor_y_mid = tf.cast(vertices_y_mid, tf.int32)
    gathered_x_mid = tf.gather_nd(conv_head_x, indexing_tensor_x_mid)
    gathered_y_mid = tf.gather_nd(conv_head_y, indexing_tensor_y_mid)

    #indexing_tensor = tf.reshape(indexing_tensor, [-1, 3])
    vertex_features = tf.gather_nd(image_features, indexing_tensor)
    conv_width = tf.shape(image_features)[1]
    conv_depth = tf.shape(image_features)[-1]
    conv_range = tf.range(0, conv_width, dtype=tf.int32)
    mask_gt_y1 = tf.math.greater_equal(conv_range, vertices_y[:, tf.newaxis])
    print("mask_gt_y1 : ######" + str(mask_gt_y1.shape) + " ######")
    mask_lt_y2 = tf.math.less_equal(conv_range, vertices_y2[:, tf.newaxis])
    print("mask_lt_y2 : ######" + str(mask_lt_y2.shape) + " ######")
    mask_y_mid = tf.math.logical_and(mask_gt_y1, mask_lt_y2)
    mask_y_mid = tf.tile(tf.expand_dims(mask_y_mid, -1), [1, 1, conv_depth])

    mask_gt_x1 = tf.math.greater_equal(conv_range, vertices_x[:, tf.newaxis])
    print("mask_gt_y1 : ######" + str(mask_gt_x1.shape) + " ######")
    mask_lt_x2 = tf.math.less_equal(conv_range, vertices_x2[:, tf.newaxis])
    print("mask_lt_y2 : ######" + str(mask_lt_x2.shape) + " ######")
    mask_x_mid = tf.math.logical_and(mask_gt_x1, mask_lt_x2)
    mask_x_mid = tf.tile(tf.expand_dims(mask_x_mid, -1), [1, 1, conv_depth])
    print("Here.....")
    print(conv_width)
    print(conv_depth)
    gathered_values_x_mid = tf.where(
        mask_x_mid, gathered_x_mid,
        tf.zeros([tf.shape(positive_preds)[0], conv_width, conv_depth],
                 dtype=tf.float32))

    gathered_values_y_mid = tf.where(
        mask_y_mid, gathered_y_mid,
        tf.zeros([tf.shape(positive_preds)[0], conv_width, conv_depth],
                 dtype=tf.float32))
    print("Here...1")

    vertex_features = tf.reshape(vertex_features,
                                 [-1, config.TOP_DOWN_PYRAMID_SIZE])
    print("Here...2")

    lstm_input_x = gathered_values_x_mid
    lstm_input_y = gathered_values_y_mid
    print("%%%%%" + str(lstm_input_x.shape) + "#######")
    print("%%%%%" + str(lstm_input_y.shape) + "#######")

    stuct_info = tf.concat([
        tf.expand_dims(roi_gt_start_rows, -1),
        tf.expand_dims(roi_gt_start_cols, -1),
        tf.expand_dims(roi_gt_end_rows, -1),
        tf.expand_dims(roi_gt_end_cols, -1)
    ], 1)
    P = tf.maximum(config.TRAIN_ROIS_PER_IMAGE - tf.shape(positive_preds)[0],
                   0)
    pred_boxes = tf.pad(positive_preds, [(0, P), (0, 0)])
    roi_gt_boxes = tf.pad(roi_gt_boxes, [(0, P), (0, 0)])
    stuct_info = tf.pad(stuct_info, [(0, P), (0, 0)])
    adj_rows = tf.pad(adj_rows, [(0, P), (0, P)])
    adj_cols = tf.pad(adj_cols, [(0, P), (0, P)])
    vertex_features = tf.pad(vertex_features, [(0, P), (0, 0)])
    pred_boxes_denorm = tf.pad(pred_boxes_denorm, [(0, P), (0, 0)])

    print("Graph vertices collection....")
    print("Image features..." + str(image_features.shape))

    print("Indexing tensor..." + str(indexing_tensor.shape))

    #print(pred_boxes.shape)
    #print(stuct_info.shape)
    #print(roi_gt_boxes.shape)

    return pred_boxes, pred_boxes_denorm, roi_gt_boxes, stuct_info, adj_rows, adj_cols, vertex_features


class StructureDetectionLayer(KE.Layer):
    """Takes classified proposal boxes and their bounding box deltas and
    returns the final detection boxes.

    Returns:
    [batch, num_detections, (y1, x1, y2, x2, class_id, class_score)] where
    coordinates are normalized.
    """
    def __init__(self, config=None, **kwargs):
        super(StructureDetectionLayer, self).__init__(**kwargs)
        self.config = config

    def call(self, inputs):
        rois = inputs[0]
        mrcnn_class = inputs[1]
        mrcnn_bbox = inputs[2]
        image_meta = inputs[3]
        gt_class_ids = inputs[4]
        gt_boxes = inputs[5]
        gt_start_rows = inputs[6]
        gt_start_cols = inputs[7]
        gt_end_rows = inputs[8]
        gt_end_cols = inputs[9]
        image_features = inputs[10]

        # Get windows of images in normalized coordinates. Windows are the area
        # in the image that excludes the padding.
        # Use the shape of the first image in the batch to normalize the window
        # because we know that all images get resized to the same size.
        m = parse_image_meta_graph(image_meta)
        image_shape = m['image_shape'][0]
        window = norm_boxes_graph(m['window'], image_shape[:2])

        # Run detection refinement graph on each item in the batch
        detections_batch = utils.batch_slice(
            [
                image_features, rois, mrcnn_class, mrcnn_bbox, window,
                gt_class_ids, gt_boxes, gt_start_rows, gt_start_cols,
                gt_end_rows, gt_end_cols
            ], lambda imf, x, y, w, z, a, b,
            c, d, e, f: refine_structure_detections_graph(
                imf, x, y, w, z, a, b, c, d, e, f, self.config),
            self.config.IMAGES_PER_GPU)

        # Reshape output
        # [batch, num_detections, (y1, x1, y2, x2, class_id, class_score)] in
        # normalized coordinates
        return detections_batch

    def compute_output_shape(self, input_shape):
        return [(None, self.config.TRAIN_ROIS_PER_IMAGE, 4),
                (None, self.config.TRAIN_ROIS_PER_IMAGE, 4),
                (None, self.config.TRAIN_ROIS_PER_IMAGE, 4),
                (None, self.config.TRAIN_ROIS_PER_IMAGE, 4),
                (None, self.config.TRAIN_ROIS_PER_IMAGE,
                 self.config.TRAIN_ROIS_PER_IMAGE),
                (None, self.config.TRAIN_ROIS_PER_IMAGE,
                 self.config.TRAIN_ROIS_PER_IMAGE),
                (None, self.config.TRAIN_ROIS_PER_IMAGE,
                 self.config.TOP_DOWN_PYRAMID_SIZE)]


class CreateZeroMask(KL.Layer):
    '''
    Creates a mask based on the 0th index of the vertex
    To apply, use keras.Layers.Multiply
    '''
    def __init__(self, **kwargs):
        super(CreateZeroMask, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], 1)

    def call(self, inputs):
        zeros = tf.zeros(shape=tf.shape(inputs)[:-1])
        mask = tf.where(inputs[:, :, 0] > 0, zeros + 1., zeros)
        mask = tf.expand_dims(mask, axis=2)
        return mask

    def get_config(self):
        #config = {'my_configoption': self.my_configoption}
        base_config = super(CreateZeroMask, self).get_config()
        return dict(list(base_config.items()))  # + list(config.items() ))


class GlobalExchange(KL.Layer):
    def __init__(self, **kwargs):
        super(GlobalExchange, self).__init__(**kwargs)

    def build(self, input_shape):
        # tf.ragged FIXME?
        self.num_vertices = input_shape[1]
        super(GlobalExchange, self).build(input_shape)

    def call(self, x):
        mean = tf.reduce_mean(x, axis=1, keepdims=True)
        # tf.ragged FIXME?
        # maybe just use tf.shape(x)[1] instead?
        mean = tf.tile(mean, [1, self.num_vertices, 1])
        return tf.concat([x, mean], axis=-1)

    def compute_output_shape(self, input_shape):
        return input_shape[:2] + (input_shape[2] * 2, )


class GravNet(KL.Layer):
    def __init__(self,
                 n_neighbours,
                 n_dimensions,
                 n_filters,
                 n_propagate,
                 name,
                 also_coordinates=False,
                 feature_dropout=-1,
                 coordinate_kernel_initializer=keras.initializers.Orthogonal(),
                 other_kernel_initializer='glorot_uniform',
                 fix_coordinate_space=False,
                 coordinate_activation=None,
                 masked_coordinate_offset=None,
                 **kwargs):
        super(GravNet, self).__init__(**kwargs)

        self.n_neighbours = n_neighbours
        self.n_dimensions = n_dimensions
        self.n_filters = n_filters
        self.n_propagate = n_propagate
        self.name = name
        self.also_coordinates = also_coordinates
        self.feature_dropout = feature_dropout
        self.masked_coordinate_offset = masked_coordinate_offset

        self.input_feature_transform = keras.layers.Dense(
            n_propagate,
            name=name + '_FLR',
            kernel_initializer=other_kernel_initializer)
        self.input_spatial_transform = keras.layers.Dense(
            n_dimensions,
            name=name + '_S',
            kernel_initializer=coordinate_kernel_initializer,
            activation=coordinate_activation)
        self.output_feature_transform = keras.layers.Dense(
            n_filters,
            activation='tanh',
            name=name + '_Fout',
            kernel_initializer=other_kernel_initializer)

        self._sublayers = [
            self.input_feature_transform, self.input_spatial_transform,
            self.output_feature_transform
        ]
        if fix_coordinate_space:
            self.input_spatial_transform = None
            self._sublayers = [
                self.input_feature_transform, self.output_feature_transform
            ]

    def build(self, input_shape):
        if self.masked_coordinate_offset is not None:
            input_shape = input_shape[0]

        self.input_feature_transform.build(input_shape)
        if self.input_spatial_transform is not None:
            self.input_spatial_transform.build(input_shape)

        # tf.ragged FIXME?
        self.output_feature_transform.build(
            (input_shape[0], input_shape[1],
             input_shape[2] + self.input_feature_transform.units * 2))

        for layer in self._sublayers:
            self._trainable_weights.extend(layer.trainable_weights)
            self._non_trainable_weights.extend(layer.non_trainable_weights)

        super(GravNet, self).build(input_shape)

    def call(self, x):

        if self.masked_coordinate_offset is not None:
            if not isinstance(x, list):
                raise Exception(
                    'GravNet: in mask mode, input must be list of input,mask')
            mask = x[1]
            x = x[0]

        features = self.input_feature_transform(x)

        if self.feature_dropout > 0 and self.feature_dropout < 1:
            features = keras.layers.Dropout(self.feature_dropout)(features)

        if self.input_spatial_transform is not None:
            coordinates = self.input_spatial_transform(x)
        else:
            coordinates = x[:, :, 0:self.n_dimensions]

        if self.masked_coordinate_offset is not None:
            sel_mask = tf.tile(mask, [1, 1, tf.shape(coordinates)[2]])
            coordinates = tf.where(
                sel_mask > 0., coordinates,
                tf.zeros_like(coordinates) - self.masked_coordinate_offset)

        collected_neighbours = self.collect_neighbours(coordinates, features)

        updated_features = tf.concat([x, collected_neighbours], axis=-1)
        output = self.output_feature_transform(updated_features)

        if self.masked_coordinate_offset is not None:
            output *= mask

        if self.also_coordinates:
            return [output, coordinates]
        return output

    def compute_output_shape(self, input_shape):
        if self.masked_coordinate_offset is not None:
            input_shape = input_shape[0]
        if self.also_coordinates:
            return [(input_shape[0], input_shape[1],
                     self.output_feature_transform.units),
                    (input_shape[0], input_shape[1], self.n_dimensions)]

        # tf.ragged FIXME? tf.shape() might do the trick already
        return (input_shape[0], input_shape[1],
                self.output_feature_transform.units)

    def collect_neighbours(self, coordinates, features):

        # tf.ragged FIXME?
        # for euclidean_squared see caloGraphNN.py
        distance_matrix = euclidean_squared(coordinates, coordinates)
        ranked_distances, ranked_indices = tf.nn.top_k(-distance_matrix,
                                                       self.n_neighbours)

        neighbour_indices = ranked_indices[:, :, 1:]

        n_batches = tf.shape(features)[0]

        # tf.ragged FIXME? or could that work?
        n_vertices = tf.shape(features)[1]
        n_features = tf.shape(features)[2]

        batch_range = tf.range(0, n_batches)
        batch_range = tf.expand_dims(batch_range, axis=1)
        batch_range = tf.expand_dims(batch_range, axis=1)
        batch_range = tf.expand_dims(batch_range, axis=1)  # (B, 1, 1, 1)

        # tf.ragged FIXME? n_vertices
        batch_indices = tf.tile(
            batch_range,
            [1, n_vertices, self.n_neighbours - 1, 1])  # (B, V, N-1, 1)
        vertex_indices = tf.expand_dims(neighbour_indices,
                                        axis=3)  # (B, V, N-1, 1)
        indices = tf.concat([batch_indices, vertex_indices], axis=-1)

        neighbour_features = tf.gather_nd(features, indices)  # (B, V, N-1, F)

        distance = -ranked_distances[:, :, 1:]

        weights = gauss_of_lin(distance * 10.)
        weights = tf.expand_dims(weights, axis=-1)

        # weight the neighbour_features
        neighbour_features *= weights

        neighbours_max = tf.reduce_max(neighbour_features, axis=2)
        neighbours_mean = tf.reduce_mean(neighbour_features, axis=2)

        return tf.concat([neighbours_max, neighbours_mean], axis=-1)

    def get_config(self):
        config = {
            'n_neighbours': self.n_neighbours,
            'n_dimensions': self.n_dimensions,
            'n_filters': self.n_filters,
            'n_propagate': self.n_propagate,
            'name': self.name,
            'also_coordinates': self.also_coordinates,
            'feature_dropout': self.feature_dropout,
            'masked_coordinate_offset': self.masked_coordinate_offset
        }
        base_config = super(GravNet, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class GarNet(KL.Layer):
    def __init__(self, n_aggregators, n_filters, n_propagate, name, **kwargs):
        super(GarNet, self).__init__(**kwargs)

        self.n_aggregators = n_aggregators
        self.n_filters = n_filters
        self.n_propagate = n_propagate
        self.name = name

        self.input_feature_transform = keras.layers.Dense(n_propagate,
                                                          name=name + '_FLR')
        self.aggregator_distance = keras.layers.Dense(n_aggregators,
                                                      name=name + '_S')
        self.output_feature_transform = keras.layers.Dense(n_filters,
                                                           activation='tanh',
                                                           name=name + '_Fout')

        self._sublayers = [
            self.input_feature_transform, self.aggregator_distance,
            self.output_feature_transform
        ]

    def build(self, input_shape):
        self.input_feature_transform.build(input_shape)
        self.aggregator_distance.build(input_shape)

        # tf.ragged FIXME? tf.shape()?
        self.output_feature_transform.build(
            (input_shape[0], input_shape[1],
             input_shape[2] + self.aggregator_distance.units +
             2 * self.aggregator_distance.units *
             (self.input_feature_transform.units +
              self.aggregator_distance.units)))

        for layer in self._sublayers:
            self._trainable_weights.extend(layer.trainable_weights)
            self._non_trainable_weights.extend(layer.non_trainable_weights)

        super(GarNet, self).build(input_shape)

    def call(self, x):
        features = self.input_feature_transform(x)  # (B, V, F)
        distance = self.aggregator_distance(x)  # (B, V, S)

        edge_weights = gauss(distance)

        features = tf.concat([features, edge_weights], axis=-1)  # (B, V, F+S)

        # vertices -> aggregators
        edge_weights_trans = tf.transpose(edge_weights,
                                          perm=(0, 2, 1))  # (B, S, V)
        aggregated_max = self.apply_edge_weights(
            features, edge_weights_trans,
            aggregation=tf.reduce_max)  # (B, S, F+S)
        aggregated_mean = self.apply_edge_weights(
            features, edge_weights_trans,
            aggregation=tf.reduce_mean)  # (B, S, F+S)

        aggregated = tf.concat([aggregated_max, aggregated_mean],
                               axis=-1)  # (B, S, 2*(F+S))

        # aggregators -> vertices
        updated_features = self.apply_edge_weights(
            aggregated, edge_weights)  # (B, V, 2*S*(F+S))

        updated_features = tf.concat([x, updated_features, edge_weights],
                                     axis=-1)  # (B, V, X+2*S*(F+S)+S)

        return self.output_feature_transform(updated_features)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1],
                self.output_feature_transform.units)

    def apply_edge_weights(self, features, edge_weights, aggregation=None):
        features = tf.expand_dims(features, axis=1)  # (B, 1, v, f)
        edge_weights = tf.expand_dims(edge_weights, axis=3)  # (B, A, v, 1)

        # tf.ragged FIXME? broadcasting should work
        out = edge_weights * features  # (B, u, v, f)
        # tf.ragged FIXME? these values won't work
        n = features.shape[-2].value * features.shape[-1].value

        if aggregation:
            out = aggregation(out, axis=2)  # (B, u, f)
            n = features.shape[-1].value

        # tf.ragged FIXME? there might be a chance to spell out batch dim instead and use -1 for vertices?
        return tf.reshape(out, [-1, out.shape[1].value, n])  # (B, u, n)

    def get_config(self):
        config = {
            'n_aggregators': self.n_aggregators,
            'n_filters': self.n_filters,
            'n_propagate': self.n_propagate,
            'name': self.name
        }
        base_config = super(GarNet, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


    # tf.ragged FIXME? the last one should be no problem
class weighted_sum_layer(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(weighted_sum_layer, self).__init__(**kwargs)

    def get_config(self):
        base_config = super(weighted_sum_layer, self).get_config()
        return dict(list(base_config.items()))

    def compute_output_shape(self, input_shape):
        assert input_shape[2] > 1
        inshape = list(input_shape)
        return tuple((inshape[0], input_shape[2] - 1))

    def call(self, inputs):
        # input #B x E x F
        weights = inputs[:, :, 0:1]  #B x E x 1
        tosum = inputs[:, :, 1:]
        weighted = weights * tosum  #broadcast to B x E x F-1
        return tf.reduce_sum(weighted, axis=1)


############################################################
#  Detection Layer
############################################################


def refine_detections_graph(rois, probs, deltas, window, image_features,
                            config):
    """Refine classified proposals and filter overlaps and return final
    detections.

    Inputs:
        rois: [N, (y1, x1, y2, x2)] in normalized coordinates
        probs: [N, num_classes]. Class probabilities.
        deltas: [N, num_classes, (dy, dx, log(dh), log(dw))]. Class-specific
                bounding box deltas.
        window: (y1, x1, y2, x2) in normalized coordinates. The part of the image
            that contains the image excluding the padding.

    Returns detections shaped: [num_detections, (y1, x1, y2, x2, class_id, score)] where
        coordinates are normalized.
    """
    # Class IDs per ROI
    class_ids = tf.argmax(probs, axis=1, output_type=tf.int32)
    # Class probability of the top class of each ROI
    indices = tf.stack([tf.range(probs.shape[0]), class_ids], axis=1)
    class_scores = tf.gather_nd(probs, indices)
    # Class-specific bounding box deltas
    deltas_specific = tf.gather_nd(deltas, indices)
    # Apply bounding box deltas
    # Shape: [boxes, (y1, x1, y2, x2)] in normalized coordinates
    refined_rois = apply_box_deltas_graph(
        rois, deltas_specific * config.BBOX_STD_DEV)
    # Clip boxes to image window
    refined_rois = clip_boxes_graph(refined_rois, window)

    # TODO: Filter out boxes with zero area

    # Filter out background boxes
    keep = tf.where(class_ids > 0)[:, 0]
    # Filter out low confidence boxes
    if config.DETECTION_MIN_CONFIDENCE:
        conf_keep = tf.where(
            class_scores >= config.DETECTION_MIN_CONFIDENCE)[:, 0]
        keep = tf.sets.set_intersection(tf.expand_dims(keep, 0),
                                        tf.expand_dims(conf_keep, 0))
        keep = tf.sparse_tensor_to_dense(keep)[0]

    # Apply per-class NMS
    # 1. Prepare variables
    pre_nms_class_ids = tf.gather(class_ids, keep)
    pre_nms_scores = tf.gather(class_scores, keep)
    pre_nms_rois = tf.gather(refined_rois, keep)
    unique_pre_nms_class_ids = tf.unique(pre_nms_class_ids)[0]

    def nms_keep_map(class_id):
        """Apply Non-Maximum Suppression on ROIs of the given class."""
        # Indices of ROIs of the given class
        ixs = tf.where(tf.equal(pre_nms_class_ids, class_id))[:, 0]
        # Apply NMS
        class_keep = tf.image.non_max_suppression(
            tf.gather(pre_nms_rois, ixs),
            tf.gather(pre_nms_scores, ixs),
            max_output_size=config.DETECTION_MAX_INSTANCES,
            iou_threshold=config.DETECTION_NMS_THRESHOLD)
        # Map indices
        class_keep = tf.gather(keep, tf.gather(ixs, class_keep))
        # Pad with -1 so returned tensors have the same shape
        gap = config.DETECTION_MAX_INSTANCES - tf.shape(class_keep)[0]
        class_keep = tf.pad(class_keep, [(0, gap)],
                            mode='CONSTANT',
                            constant_values=-1)
        # Set shape so map_fn() can infer result shape
        class_keep.set_shape([config.DETECTION_MAX_INSTANCES])
        return class_keep

    # 2. Map over class IDs
    nms_keep = tf.map_fn(nms_keep_map,
                         unique_pre_nms_class_ids,
                         dtype=tf.int64)
    # 3. Merge results into one list, and remove -1 padding
    nms_keep = tf.reshape(nms_keep, [-1])
    nms_keep = tf.gather(nms_keep, tf.where(nms_keep > -1)[:, 0])
    # 4. Compute intersection between keep and nms_keep
    keep = tf.sets.set_intersection(tf.expand_dims(keep, 0),
                                    tf.expand_dims(nms_keep, 0))
    keep = tf.sparse_tensor_to_dense(keep)[0]
    # Keep top detections
    roi_count = config.DETECTION_MAX_INSTANCES
    class_scores_keep = tf.gather(class_scores, keep)
    num_keep = tf.minimum(tf.shape(class_scores_keep)[0], roi_count)
    top_ids = tf.nn.top_k(class_scores_keep, k=num_keep, sorted=True)[1]
    keep = tf.gather(keep, top_ids)

    # Arrange output as [N, (y1, x1, y2, x2, class_id, score)]
    # Coordinates are normalized.
    boxes = tf.gather(refined_rois, keep)
    pred_boxes_denorm = denorm_boxes_graph(boxes, [256, 256])
    vertices_y, vertices_x, vertices_y2, vertices_x2 = tf.split(
        pred_boxes_denorm, 4, axis=1)
    vertices_y = tf.squeeze(vertices_y)
    vertices_x = tf.squeeze(vertices_x)
    vertices_y2 = tf.squeeze(vertices_y2)
    vertices_x2 = tf.squeeze(vertices_x2)
    vertices_x_mid = (vertices_x + vertices_x2) / 2
    vertices_y_mid = (vertices_y + vertices_y2) / 2
    vertices_x_mid = tf.reshape(vertices_x_mid,
                                [tf.shape(pred_boxes_denorm)[0], 1])
    vertices_y_mid = tf.reshape(vertices_y_mid,
                                [tf.shape(pred_boxes_denorm)[0], 1])
    indexing_tensor = tf.concat(
        (vertices_y_mid[..., tf.newaxis], vertices_x_mid[..., tf.newaxis]),
        axis=-1)
    indexing_tensor = tf.cast(indexing_tensor, tf.int32)
    vertex_features = tf.gather_nd(image_features, indexing_tensor)
    vertex_features = tf.reshape(vertex_features,
                                 [-1, config.TOP_DOWN_PYRAMID_SIZE])
    #pred_boxes_denorm = tf.cast(pred_boxes_denorm, tf.float32)
    detections = tf.concat([
        tf.gather(refined_rois, keep),
        tf.to_float(tf.gather(class_ids, keep))[..., tf.newaxis],
        tf.gather(class_scores, keep)[..., tf.newaxis]
    ],
                           axis=1)

    # Pad with zeros if detections < DETECTION_MAX_INSTANCES
    gap = config.DETECTION_MAX_INSTANCES - tf.shape(detections)[0]
    detections = tf.pad(detections, [(0, gap), (0, 0)], "CONSTANT")
    pred_boxes_denorm = tf.pad(pred_boxes_denorm, [(0, gap), (0, 0)])
    pred_boxes_denorm = tf.cast(pred_boxes_denorm, tf.float32)
    vertex_features = tf.pad(vertex_features, [(0, gap), (0, 0)])

    return detections, pred_boxes_denorm, vertex_features


class DetectionLayer(KE.Layer):
    """Takes classified proposal boxes and their bounding box deltas and
    returns the final detection boxes.

    Returns:
    [batch, num_detections, (y1, x1, y2, x2, class_id, class_score)] where
    coordinates are normalized.
    """
    def __init__(self, config=None, **kwargs):
        super(DetectionLayer, self).__init__(**kwargs)
        self.config = config

    def call(self, inputs):
        rois = inputs[0]
        mrcnn_class = inputs[1]
        mrcnn_bbox = inputs[2]
        image_meta = inputs[3]
        image_features = inputs[4]

        # Get windows of images in normalized coordinates. Windows are the area
        # in the image that excludes the padding.
        # Use the shape of the first image in the batch to normalize the window
        # because we know that all images get resized to the same size.
        m = parse_image_meta_graph(image_meta)
        image_shape = m['image_shape'][0]
        window = norm_boxes_graph(m['window'], image_shape[:2])

        # Run detection refinement graph on each item in the batch
        detections_batch = utils.batch_slice(
            [rois, mrcnn_class, mrcnn_bbox, window, image_features],
            lambda x, y, w, z, u: refine_detections_graph(
                x, y, w, z, u, self.config), self.config.IMAGES_PER_GPU)

        # Reshape output
        # [batch, num_detections, (y1, x1, y2, x2, class_id, class_score)] in
        # normalized coordinates
        return detections_batch
        #tf.reshape(
        #    detections_batch,
        #    [self.config.BATCH_SIZE, self.config.DETECTION_MAX_INSTANCES, 6]),tf.reshape(
        #    pred_boxes_denorm,
        #    [self.config.BATCH_SIZE, self.config.DETECTION_MAX_INSTANCES, 4]), tf.reshape(
        #    vertex_features,
        #    [self.config.BATCH_SIZE, self.config.DETECTION_MAX_INSTANCES, self.config.TOP_DOWN_PYRAMID_SIZE])

    def compute_output_shape(self, input_shape):
        return [(None, self.config.DETECTION_MAX_INSTANCES, 6),
                (None, self.config.DETECTION_MAX_INSTANCES, 4),
                (None, self.config.DETECTION_MAX_INSTANCES,
                 self.config.TOP_DOWN_PYRAMID_SIZE)]


############################################################
#  Region Proposal Network (RPN)
############################################################


def rpn_graph(feature_map, anchors_per_location, anchor_stride):
    """Builds the computation graph of Region Proposal Network.

    feature_map: backbone features [batch, height, width, depth]
    anchors_per_location: number of anchors per pixel in the feature map
    anchor_stride: Controls the density of anchors. Typically 1 (anchors for
                   every pixel in the feature map), or 2 (every other pixel).

    Returns:
        rpn_class_logits: [batch, H * W * anchors_per_location, 2] Anchor classifier logits (before softmax)
        rpn_probs: [batch, H * W * anchors_per_location, 2] Anchor classifier probabilities.
        rpn_bbox: [batch, H * W * anchors_per_location, (dy, dx, log(dh), log(dw))] Deltas to be
                  applied to anchors.
    """
    # TODO: check if stride of 2 causes alignment issues if the feature map
    # is not even.
    # Shared convolutional base of the RPN
    shared = KL.Conv2D(512, (3, 3),
                       padding='same',
                       activation='relu',
                       strides=anchor_stride,
                       name='rpn_conv_shared')(feature_map)

    # Anchor Score. [batch, height, width, anchors per location * 2].
    x = KL.Conv2D(2 * anchors_per_location, (1, 1),
                  padding='valid',
                  activation='linear',
                  name='rpn_class_raw')(shared)

    # Reshape to [batch, anchors, 2]
    rpn_class_logits = KL.Lambda(
        lambda t: tf.reshape(t, [tf.shape(t)[0], -1, 2]))(x)

    # Softmax on last dimension of BG/FG.
    rpn_probs = KL.Activation("softmax",
                              name="rpn_class_xxx")(rpn_class_logits)

    # Bounding box refinement. [batch, H, W, anchors per location * depth]
    # where depth is [x, y, log(w), log(h)]
    x = KL.Conv2D(anchors_per_location * 4, (1, 1),
                  padding="valid",
                  activation='linear',
                  name='rpn_bbox_pred')(shared)

    # Reshape to [batch, anchors, 4]
    rpn_bbox = KL.Lambda(lambda t: tf.reshape(t, [tf.shape(t)[0], -1, 4]))(x)

    return [rpn_class_logits, rpn_probs, rpn_bbox]


def build_rpn_model(anchor_stride, anchors_per_location, depth):
    """Builds a Keras model of the Region Proposal Network.
    It wraps the RPN graph so it can be used multiple times with shared
    weights.

    anchors_per_location: number of anchors per pixel in the feature map
    anchor_stride: Controls the density of anchors. Typically 1 (anchors for
                   every pixel in the feature map), or 2 (every other pixel).
    depth: Depth of the backbone feature map.

    Returns a Keras Model object. The model outputs, when called, are:
    rpn_class_logits: [batch, H * W * anchors_per_location, 2] Anchor classifier logits (before softmax)
    rpn_probs: [batch, H * W * anchors_per_location, 2] Anchor classifier probabilities.
    rpn_bbox: [batch, H * W * anchors_per_location, (dy, dx, log(dh), log(dw))] Deltas to be
                applied to anchors.
    """
    input_feature_map = KL.Input(shape=[None, None, depth],
                                 name="input_rpn_feature_map")
    outputs = rpn_graph(input_feature_map, anchors_per_location, anchor_stride)
    return KM.Model([input_feature_map], outputs, name="rpn_model")


############################################################
#  Feature Pyramid Network Heads
############################################################


def fpn_classifier_graph(rois,
                         feature_maps,
                         image_meta,
                         pool_size,
                         num_classes,
                         train_bn=True,
                         fc_layers_size=1024):
    """Builds the computation graph of the feature pyramid network classifier
    and regressor heads.

    rois: [batch, num_rois, (y1, x1, y2, x2)] Proposal boxes in normalized
          coordinates.
    feature_maps: List of feature maps from different layers of the pyramid,
                  [P2, P3, P4, P5]. Each has a different resolution.
    image_meta: [batch, (meta data)] Image details. See compose_image_meta()
    pool_size: The width of the square feature map generated from ROI Pooling.
    num_classes: number of classes, which determines the depth of the results
    train_bn: Boolean. Train or freeze Batch Norm layers
    fc_layers_size: Size of the 2 FC layers

    Returns:
        logits: [batch, num_rois, NUM_CLASSES] classifier logits (before softmax)
        probs: [batch, num_rois, NUM_CLASSES] classifier probabilities
        bbox_deltas: [batch, num_rois, NUM_CLASSES, (dy, dx, log(dh), log(dw))] Deltas to apply to
                     proposal boxes
    """
    # ROI Pooling
    # Shape: [batch, num_rois, POOL_SIZE, POOL_SIZE, channels]
    x = PyramidROIAlign([pool_size, pool_size],
                        name="roi_align_classifier")([rois, image_meta] +
                                                     feature_maps)
    # Two 1024 FC layers (implemented with Conv2D for consistency)
    x = KL.TimeDistributed(KL.Conv2D(fc_layers_size, (pool_size, pool_size),
                                     padding="valid"),
                           name="mrcnn_class_conv1")(x)
    x = KL.TimeDistributed(BatchNorm(),
                           name='mrcnn_class_bn1')(x, training=train_bn)
    x = KL.Activation('relu')(x)
    x = KL.TimeDistributed(KL.Conv2D(fc_layers_size, (1, 1)),
                           name="mrcnn_class_conv2")(x)
    x = KL.TimeDistributed(BatchNorm(),
                           name='mrcnn_class_bn2')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    shared = KL.Lambda(lambda x: K.squeeze(K.squeeze(x, 3), 2),
                       name="pool_squeeze")(x)

    # Classifier head
    mrcnn_class_logits = KL.TimeDistributed(KL.Dense(num_classes),
                                            name='mrcnn_class_logits')(shared)
    mrcnn_probs = KL.TimeDistributed(KL.Activation("softmax"),
                                     name="mrcnn_class")(mrcnn_class_logits)

    # BBox head
    # [batch, num_rois, NUM_CLASSES * (dy, dx, log(dh), log(dw))]
    x = KL.TimeDistributed(KL.Dense(num_classes * 4, activation='linear'),
                           name='mrcnn_bbox_fc')(shared)
    # Reshape to [batch, num_rois, NUM_CLASSES, (dy, dx, log(dh), log(dw))]
    s = K.int_shape(x)
    mrcnn_bbox = KL.Reshape((s[1], num_classes, 4), name="mrcnn_bbox")(x)

    return mrcnn_class_logits, mrcnn_probs, mrcnn_bbox


def build_fpn_mask_graph(rois,
                         feature_maps,
                         image_meta,
                         pool_size,
                         num_classes,
                         train_bn=True):
    """Builds the computation graph of the mask head of Feature Pyramid Network.

    rois: [batch, num_rois, (y1, x1, y2, x2)] Proposal boxes in normalized
          coordinates.
    feature_maps: List of feature maps from different layers of the pyramid,
                  [P2, P3, P4, P5]. Each has a different resolution.
    image_meta: [batch, (meta data)] Image details. See compose_image_meta()
    pool_size: The width of the square feature map generated from ROI Pooling.
    num_classes: number of classes, which determines the depth of the results
    train_bn: Boolean. Train or freeze Batch Norm layers

    Returns: Masks [batch, num_rois, MASK_POOL_SIZE, MASK_POOL_SIZE, NUM_CLASSES]
    """
    # ROI Pooling
    # Shape: [batch, num_rois, MASK_POOL_SIZE, MASK_POOL_SIZE, channels]
    x = PyramidROIAlign([pool_size, pool_size],
                        name="roi_align_mask")([rois, image_meta] +
                                               feature_maps)

    # Conv layers
    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"),
                           name="mrcnn_mask_conv1")(x)
    x = KL.TimeDistributed(BatchNorm(),
                           name='mrcnn_mask_bn1')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"),
                           name="mrcnn_mask_conv2")(x)
    x = KL.TimeDistributed(BatchNorm(),
                           name='mrcnn_mask_bn2')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"),
                           name="mrcnn_mask_conv3")(x)
    x = KL.TimeDistributed(BatchNorm(),
                           name='mrcnn_mask_bn3')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"),
                           name="mrcnn_mask_conv4")(x)
    x = KL.TimeDistributed(BatchNorm(),
                           name='mrcnn_mask_bn4')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv2DTranspose(256, (2, 2),
                                              strides=2,
                                              activation="relu"),
                           name="mrcnn_mask_deconv")(x)
    x = KL.TimeDistributed(KL.Conv2D(num_classes, (1, 1),
                                     strides=1,
                                     activation="sigmoid"),
                           name="mrcnn_mask")(x)
    return x


############################################################
#  Loss Functions
############################################################


def smooth_l1_loss(y_true, y_pred):
    """Implements Smooth-L1 loss.
    y_true and y_pred are typically: [N, 4], but could be any shape.
    """
    diff = K.abs(y_true - y_pred)
    less_than_one = K.cast(K.less(diff, 1.0), "float32")
    loss = (less_than_one * 0.5 * diff**2) + (1 - less_than_one) * (diff - 0.5)
    return loss


def adjacency_loss(y_true, y_pred):
    """Implements Smooth-L1 loss.
    y_true and y_pred are typically: [N, 4], but could be any shape.
    """
    print(y_true.shape)
    print(y_pred.shape)
    y_true = tf.cast(y_true, tf.int32)
    total_count = tf.cast(tf.size(y_true), tf.float32)
    count_ones = tf.cast(tf.math.count_nonzero(y_true),
                         tf.float32) + tf.constant([1.0], tf.float32)
    count_zeros = total_count - count_ones
    pos_weight = tf.math.divide(count_zeros, count_ones)
    #loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.one_hot(y_true, depth=2), logits=y_pred)
    loss = tf.nn.weighted_cross_entropy_with_logits(logits=y_pred,
                                                    targets=tf.one_hot(
                                                        y_true, depth=2),
                                                    pos_weight=pos_weight)
    loss = tf.reduce_mean(loss)
    return loss


def rpn_class_loss_graph(rpn_match, rpn_class_logits):
    """RPN anchor classifier loss.

    rpn_match: [batch, anchors, 1]. Anchor match type. 1=positive,
               -1=negative, 0=neutral anchor.
    rpn_class_logits: [batch, anchors, 2]. RPN classifier logits for BG/FG.
    """
    # Squeeze last dim to simplify
    rpn_match = tf.squeeze(rpn_match, -1)
    # Get anchor classes. Convert the -1/+1 match to 0/1 values.
    anchor_class = K.cast(K.equal(rpn_match, 1), tf.int32)
    # Positive and Negative anchors contribute to the loss,
    # but neutral anchors (match value = 0) don't.
    indices = tf.where(K.not_equal(rpn_match, 0))
    # Pick rows that contribute to the loss and filter out the rest.
    rpn_class_logits = tf.gather_nd(rpn_class_logits, indices)
    anchor_class = tf.gather_nd(anchor_class, indices)
    # Cross entropy loss
    loss = K.sparse_categorical_crossentropy(target=anchor_class,
                                             output=rpn_class_logits,
                                             from_logits=True)
    loss = K.switch(tf.size(loss) > 0, K.mean(loss), tf.constant(0.0))
    return loss


def rpn_bbox_structural_loss_graph(roi_box_coordinate, roi_gt_coordinate,
                                   segments):
    #tf.InteractiveSession()
    segments = tf.cast(segments, tf.float32)
    roi_box_pairwise_diff = tf.expand_dims(
        roi_box_coordinate, 1) - tf.expand_dims(roi_box_coordinate, 2)
    roi_gt_box_pairwise_diff = tf.expand_dims(
        roi_gt_coordinate, 1) - tf.expand_dims(roi_gt_coordinate, 2)
    pairwise_equal_segments = tf.equal(tf.expand_dims(segments, 1),
                                       tf.expand_dims(segments, 2))
    pairwise_equal_segments = tf.cast(pairwise_equal_segments, tf.float32)
    #print(roi_gt_box_pairwise_diff.shape)
    #print(pairwise_equal_segments.eval())
    pairwise_diff_square = tf.square(
        (roi_gt_box_pairwise_diff * pairwise_equal_segments) -
        (roi_box_pairwise_diff * pairwise_equal_segments)) / 2
    sum_error_per_channel = tf.math.reduce_sum(pairwise_diff_square,
                                               axis=[1, 2])
    count_non_zero_per_channel = tf.cast(
        tf.math.count_nonzero(pairwise_diff_square, axis=[1, 2]), tf.float32)
    #print(sum_error_per_channel.shape)
    #print(count_non_zero_per_channel.shape)
    avg_loss = sum_error_per_channel / tf.math.add(
        count_non_zero_per_channel, tf.constant([0.000001], dtype=tf.float32))
    #print(avg_loss.shape)
    total_loss = tf.math.reduce_sum(avg_loss)
    #zero = tf.constant(0, dtype=tf.float32)
    #non_zero_gts = tf.gather_nd(roi_gt_coordinate, tf.where(tf.not_equal(roi_gt_coordinate, zero)))
    #gt_equals = tf.equal(tf.expand_dims(non_zero_gts, 0), tf.expand_dims(non_zero_gts, 1))
    #gt_equals = tf.cast(gt_equals, tf.float32)
    #gt_equals = tf.matrix_set_diag(gt_equals, tf.zeros(tf.shape(gt_equals)[0]))
    #return tf.math.reduce_sum(gt_equals)

    #return pairwise_equal_segments
    #return tf.square((roi_gt_box_pairwise_diff * pairwise_equal_segments))
    return total_loss


def rpn_bbox_loss_graph(config, target_bbox, rpn_match, rpn_bbox):
    """Return the RPN bounding box loss graph.

    config: the model config object.
    target_bbox: [batch, max positive anchors, (dy, dx, log(dh), log(dw))].
        Uses 0 padding to fill in unsed bbox deltas.
    rpn_match: [batch, anchors, 1]. Anchor match type. 1=positive,
               -1=negative, 0=neutral anchor.
    rpn_bbox: [batch, anchors, (dy, dx, log(dh), log(dw))]
    """
    # Positive anchors contribute to the loss, but negative and
    # neutral anchors (match value of 0 or -1) don't.
    rpn_match = K.squeeze(rpn_match, -1)
    indices = tf.where(K.equal(rpn_match, 1))

    # Pick bbox deltas that contribute to the loss
    rpn_bbox = tf.gather_nd(rpn_bbox, indices)

    # Trim target bounding box deltas to the same length as rpn_bbox.
    batch_counts = K.sum(K.cast(K.equal(rpn_match, 1), tf.int32), axis=1)
    target_bbox = batch_pack_graph(target_bbox, batch_counts,
                                   config.IMAGES_PER_GPU)

    loss = smooth_l1_loss(target_bbox, rpn_bbox)

    loss = K.switch(tf.size(loss) > 0, K.mean(loss), tf.constant(0.0))
    return loss


def mrcnn_class_loss_graph(target_class_ids, pred_class_logits,
                           active_class_ids):
    """Loss for the classifier head of Mask RCNN.

    target_class_ids: [batch, num_rois]. Integer class IDs. Uses zero
        padding to fill in the array.
    pred_class_logits: [batch, num_rois, num_classes]
    active_class_ids: [batch, num_classes]. Has a value of 1 for
        classes that are in the dataset of the image, and 0
        for classes that are not in the dataset.
    """
    # During model building, Keras calls this function with
    # target_class_ids of type float32. Unclear why. Cast it
    # to int to get around it.
    target_class_ids = tf.cast(target_class_ids, 'int64')

    # Find predictions of classes that are not in the dataset.
    pred_class_ids = tf.argmax(pred_class_logits, axis=2)
    # TODO: Update this line to work with batch > 1. Right now it assumes all
    #       images in a batch have the same active_class_ids
    pred_active = tf.gather(active_class_ids[0], pred_class_ids)

    # Loss
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=target_class_ids, logits=pred_class_logits)

    # Erase losses of predictions of classes that are not in the active
    # classes of the image.
    loss = loss * pred_active

    # Computer loss mean. Use only predictions that contribute
    # to the loss to get a correct mean.
    loss = tf.reduce_sum(loss) / tf.reduce_sum(pred_active)
    return loss


def mrcnn_bbox_loss_graph(target_bbox, target_class_ids, pred_bbox):
    """Loss for Mask R-CNN bounding box refinement.

    target_bbox: [batch, num_rois, (dy, dx, log(dh), log(dw))]
    target_class_ids: [batch, num_rois]. Integer class IDs.
    pred_bbox: [batch, num_rois, num_classes, (dy, dx, log(dh), log(dw))]
    """
    # Reshape to merge batch and roi dimensions for simplicity.
    target_class_ids = K.reshape(target_class_ids, (-1, ))
    target_bbox = K.reshape(target_bbox, (-1, 4))
    pred_bbox = K.reshape(pred_bbox, (-1, K.int_shape(pred_bbox)[2], 4))

    # Only positive ROIs contribute to the loss. And only
    # the right class_id of each ROI. Get their indices.
    positive_roi_ix = tf.where(target_class_ids > 0)[:, 0]
    positive_roi_class_ids = tf.cast(
        tf.gather(target_class_ids, positive_roi_ix), tf.int64)
    indices = tf.stack([positive_roi_ix, positive_roi_class_ids], axis=1)

    # Gather the deltas (predicted and true) that contribute to loss
    target_bbox = tf.gather(target_bbox, positive_roi_ix)
    pred_bbox = tf.gather_nd(pred_bbox, indices)

    # Smooth-L1 Loss
    loss = K.switch(
        tf.size(target_bbox) > 0,
        smooth_l1_loss(y_true=target_bbox, y_pred=pred_bbox), tf.constant(0.0))
    loss = K.mean(loss)
    return loss


def mrcnn_mask_loss_graph(target_masks, target_class_ids, pred_masks):
    """Mask binary cross-entropy loss for the masks head.

    target_masks: [batch, num_rois, height, width].
        A float32 tensor of values 0 or 1. Uses zero padding to fill array.
    target_class_ids: [batch, num_rois]. Integer class IDs. Zero padded.
    pred_masks: [batch, proposals, height, width, num_classes] float32 tensor
                with values from 0 to 1.
    """
    # Reshape for simplicity. Merge first two dimensions into one.
    target_class_ids = K.reshape(target_class_ids, (-1, ))
    mask_shape = tf.shape(target_masks)
    target_masks = K.reshape(target_masks, (-1, mask_shape[2], mask_shape[3]))
    pred_shape = tf.shape(pred_masks)
    pred_masks = K.reshape(pred_masks,
                           (-1, pred_shape[2], pred_shape[3], pred_shape[4]))
    # Permute predicted masks to [N, num_classes, height, width]
    pred_masks = tf.transpose(pred_masks, [0, 3, 1, 2])

    # Only positive ROIs contribute to the loss. And only
    # the class specific mask of each ROI.
    positive_ix = tf.where(target_class_ids > 0)[:, 0]
    positive_class_ids = tf.cast(tf.gather(target_class_ids, positive_ix),
                                 tf.int64)
    indices = tf.stack([positive_ix, positive_class_ids], axis=1)

    # Gather the masks (predicted and true) that contribute to loss
    y_true = tf.gather(target_masks, positive_ix)
    y_pred = tf.gather_nd(pred_masks, indices)

    # Compute binary cross entropy. If no positive ROIs, then return 0.
    # shape: [batch, roi, num_classes]
    loss = K.switch(
        tf.size(y_true) > 0, K.binary_crossentropy(target=y_true,
                                                   output=y_pred),
        tf.constant(0.0))
    loss = K.mean(loss)
    return loss


############################################################
#  Data Generator
############################################################


def load_image_gt(dataset,
                  config,
                  image_id,
                  augment=False,
                  augmentation=None,
                  use_mini_mask=False):
    """Load and return ground truth data for an image (image, mask, bounding boxes).

    augment: (deprecated. Use augmentation instead). If true, apply random
        image augmentation. Currently, only horizontal flipping is offered.
    augmentation: Optional. An imgaug (https://github.com/aleju/imgaug) augmentation.
        For example, passing imgaug.augmenters.Fliplr(0.5) flips images
        right/left 50% of the time.
    use_mini_mask: If False, returns full-size masks that are the same height
        and width as the original image. These can be big, for example
        1024x1024x100 (for 100 instances). Mini masks are smaller, typically,
        224x224 and are generated by extracting the bounding box of the
        object and resizing it to MINI_MASK_SHAPE.

    Returns:
    image: [height, width, 3]
    shape: the original shape of the image before resizing and cropping.
    class_ids: [instance_count] Integer class IDs
    bbox: [instance_count, (y1, x1, y2, x2)]
    mask: [height, width, instance_count]. The height and width are those
        of the image unless use_mini_mask is True, in which case they are
        defined in MINI_MASK_SHAPE.
    """
    # Load image and mask
    image = dataset.load_image(image_id)
    mask, class_ids = dataset.load_mask(image_id)
    start_rows, start_cols, end_rows, end_cols = dataset.load_cell_structure_information(
        image_id)
    original_shape = image.shape
    image, window, scale, padding, crop = utils.resize_image(
        image,
        min_dim=config.IMAGE_MIN_DIM,
        min_scale=config.IMAGE_MIN_SCALE,
        max_dim=config.IMAGE_MAX_DIM,
        mode=config.IMAGE_RESIZE_MODE)
    mask = utils.resize_mask(mask, scale, padding, crop)

    # Random horizontal flips.
    # TODO: will be removed in a future update in favor of augmentation
    if augment:
        logging.warning("'augment' is deprecated. Use 'augmentation' instead.")
        if random.randint(0, 1):
            image = np.fliplr(image)
            mask = np.fliplr(mask)

    # Augmentation
    # This requires the imgaug lib (https://github.com/aleju/imgaug)
    if augmentation:
        import imgaug

        # Augmenters that are safe to apply to masks
        # Some, such as Affine, have settings that make them unsafe, so always
        # test your augmentation on masks
        MASK_AUGMENTERS = [
            "Sequential", "SomeOf", "OneOf", "Sometimes", "Fliplr", "Flipud",
            "CropAndPad", "Affine", "PiecewiseAffine"
        ]

        def hook(images, augmenter, parents, default):
            """Determines which augmenters to apply to masks."""
            return augmenter.__class__.__name__ in MASK_AUGMENTERS

        # Store shapes before augmentation to compare
        image_shape = image.shape
        mask_shape = mask.shape
        # Make augmenters deterministic to apply similarly to images and masks
        det = augmentation.to_deterministic()
        image = det.augment_image(image)
        # Change mask to np.uint8 because imgaug doesn't support np.bool
        mask = det.augment_image(mask.astype(np.uint8),
                                 hooks=imgaug.HooksImages(activator=hook))
        # Verify that shapes didn't change
        assert image.shape == image_shape, "Augmentation shouldn't change image size"
        assert mask.shape == mask_shape, "Augmentation shouldn't change mask size"
        # Change mask back to bool
        mask = mask.astype(np.bool)

    # Note that some boxes might be all zeros if the corresponding mask got cropped out.
    # and here is to filter them out
    _idx = np.sum(mask, axis=(0, 1)) > 0
    mask = mask[:, :, _idx]
    class_ids = class_ids[_idx]
    # Bounding boxes. Note that some boxes might be all zeros
    # if the corresponding mask got cropped out.
    # bbox: [num_instances, (y1, x1, y2, x2)]
    bbox = utils.extract_bboxes(mask)
    start_rows, start_cols, end_rows, end_cols = dataset.load_structure_information_from_boxes(
        bbox)

    #TODO : Extract structure information from boxes

    # Active classes
    # Different datasets have different classes, so track the
    # classes supported in the dataset of this image.
    active_class_ids = np.zeros([dataset.num_classes], dtype=np.int32)
    source_class_ids = dataset.source_class_ids[dataset.image_info[image_id]
                                                ["source"]]
    active_class_ids[source_class_ids] = 1

    # Resize masks to smaller size to reduce memory usage
    if use_mini_mask:
        mask = utils.minimize_mask(bbox, mask, config.MINI_MASK_SHAPE)

    # Image meta data
    image_meta = compose_image_meta(image_id, original_shape, image.shape,
                                    window, scale, active_class_ids)

    return image, image_meta, class_ids, bbox, mask, start_rows, start_cols, end_rows, end_cols


def build_detection_targets(rpn_rois, gt_class_ids, gt_boxes, gt_masks,
                            config):
    """Generate targets for training Stage 2 classifier and mask heads.
    This is not used in normal training. It's useful for debugging or to train
    the Mask RCNN heads without using the RPN head.

    Inputs:
    rpn_rois: [N, (y1, x1, y2, x2)] proposal boxes.
    gt_class_ids: [instance count] Integer class IDs
    gt_boxes: [instance count, (y1, x1, y2, x2)]
    gt_masks: [height, width, instance count] Ground truth masks. Can be full
              size or mini-masks.

    Returns:
    rois: [TRAIN_ROIS_PER_IMAGE, (y1, x1, y2, x2)]
    class_ids: [TRAIN_ROIS_PER_IMAGE]. Integer class IDs.
    bboxes: [TRAIN_ROIS_PER_IMAGE, NUM_CLASSES, (y, x, log(h), log(w))]. Class-specific
            bbox refinements.
    masks: [TRAIN_ROIS_PER_IMAGE, height, width, NUM_CLASSES). Class specific masks cropped
           to bbox boundaries and resized to neural network output size.
    """
    assert rpn_rois.shape[0] > 0
    assert gt_class_ids.dtype == np.int32, "Expected int but got {}".format(
        gt_class_ids.dtype)
    assert gt_boxes.dtype == np.int32, "Expected int but got {}".format(
        gt_boxes.dtype)
    assert gt_masks.dtype == np.bool_, "Expected bool but got {}".format(
        gt_masks.dtype)

    # It's common to add GT Boxes to ROIs but we don't do that here because
    # according to XinLei Chen's paper, it doesn't help.

    # Trim empty padding in gt_boxes and gt_masks parts
    instance_ids = np.where(gt_class_ids > 0)[0]
    assert instance_ids.shape[0] > 0, "Image must contain instances."
    gt_class_ids = gt_class_ids[instance_ids]
    gt_boxes = gt_boxes[instance_ids]
    gt_masks = gt_masks[:, :, instance_ids]

    # Compute areas of ROIs and ground truth boxes.
    rpn_roi_area = (rpn_rois[:, 2] - rpn_rois[:, 0]) * \
        (rpn_rois[:, 3] - rpn_rois[:, 1])
    gt_box_area = (gt_boxes[:, 2] - gt_boxes[:, 0]) * \
        (gt_boxes[:, 3] - gt_boxes[:, 1])

    # Compute overlaps [rpn_rois, gt_boxes]
    overlaps = np.zeros((rpn_rois.shape[0], gt_boxes.shape[0]))
    for i in range(overlaps.shape[1]):
        gt = gt_boxes[i]
        overlaps[:, i] = utils.compute_iou(gt, rpn_rois, gt_box_area[i],
                                           rpn_roi_area)

    # Assign ROIs to GT boxes
    rpn_roi_iou_argmax = np.argmax(overlaps, axis=1)
    rpn_roi_iou_max = overlaps[np.arange(overlaps.shape[0]),
                               rpn_roi_iou_argmax]
    # GT box assigned to each ROI
    rpn_roi_gt_boxes = gt_boxes[rpn_roi_iou_argmax]
    rpn_roi_gt_class_ids = gt_class_ids[rpn_roi_iou_argmax]

    # Positive ROIs are those with >= 0.5 IoU with a GT box.
    fg_ids = np.where(rpn_roi_iou_max > 0.5)[0]

    # Negative ROIs are those with max IoU 0.1-0.5 (hard example mining)
    # TODO: To hard example mine or not to hard example mine, that's the question
    # bg_ids = np.where((rpn_roi_iou_max >= 0.1) & (rpn_roi_iou_max < 0.5))[0]
    bg_ids = np.where(rpn_roi_iou_max < 0.5)[0]

    # Subsample ROIs. Aim for 33% foreground.
    # FG
    fg_roi_count = int(config.TRAIN_ROIS_PER_IMAGE * config.ROI_POSITIVE_RATIO)
    if fg_ids.shape[0] > fg_roi_count:
        keep_fg_ids = np.random.choice(fg_ids, fg_roi_count, replace=False)
    else:
        keep_fg_ids = fg_ids
    # BG
    remaining = config.TRAIN_ROIS_PER_IMAGE - keep_fg_ids.shape[0]
    if bg_ids.shape[0] > remaining:
        keep_bg_ids = np.random.choice(bg_ids, remaining, replace=False)
    else:
        keep_bg_ids = bg_ids
    # Combine indices of ROIs to keep
    keep = np.concatenate([keep_fg_ids, keep_bg_ids])
    # Need more?
    remaining = config.TRAIN_ROIS_PER_IMAGE - keep.shape[0]
    if remaining > 0:
        # Looks like we don't have enough samples to maintain the desired
        # balance. Reduce requirements and fill in the rest. This is
        # likely different from the Mask RCNN paper.

        # There is a small chance we have neither fg nor bg samples.
        if keep.shape[0] == 0:
            # Pick bg regions with easier IoU threshold
            bg_ids = np.where(rpn_roi_iou_max < 0.5)[0]
            assert bg_ids.shape[0] >= remaining
            keep_bg_ids = np.random.choice(bg_ids, remaining, replace=False)
            assert keep_bg_ids.shape[0] == remaining
            keep = np.concatenate([keep, keep_bg_ids])
        else:
            # Fill the rest with repeated bg rois.
            keep_extra_ids = np.random.choice(keep_bg_ids,
                                              remaining,
                                              replace=True)
            keep = np.concatenate([keep, keep_extra_ids])
    assert keep.shape[0] == config.TRAIN_ROIS_PER_IMAGE, \
        "keep doesn't match ROI batch size {}, {}".format(
            keep.shape[0], config.TRAIN_ROIS_PER_IMAGE)

    # Reset the gt boxes assigned to BG ROIs.
    rpn_roi_gt_boxes[keep_bg_ids, :] = 0
    rpn_roi_gt_class_ids[keep_bg_ids] = 0

    # For each kept ROI, assign a class_id, and for FG ROIs also add bbox refinement.
    rois = rpn_rois[keep]
    roi_gt_boxes = rpn_roi_gt_boxes[keep]
    roi_gt_class_ids = rpn_roi_gt_class_ids[keep]
    roi_gt_assignment = rpn_roi_iou_argmax[keep]

    # Class-aware bbox deltas. [y, x, log(h), log(w)]
    bboxes = np.zeros((config.TRAIN_ROIS_PER_IMAGE, config.NUM_CLASSES, 4),
                      dtype=np.float32)
    pos_ids = np.where(roi_gt_class_ids > 0)[0]
    bboxes[pos_ids, roi_gt_class_ids[pos_ids]] = utils.box_refinement(
        rois[pos_ids], roi_gt_boxes[pos_ids, :4])
    # Normalize bbox refinements
    bboxes /= config.BBOX_STD_DEV

    # Generate class-specific target masks
    masks = np.zeros((config.TRAIN_ROIS_PER_IMAGE, config.MASK_SHAPE[0],
                      config.MASK_SHAPE[1], config.NUM_CLASSES),
                     dtype=np.float32)
    for i in pos_ids:
        class_id = roi_gt_class_ids[i]
        assert class_id > 0, "class id must be greater than 0"
        gt_id = roi_gt_assignment[i]
        class_mask = gt_masks[:, :, gt_id]

        if config.USE_MINI_MASK:
            # Create a mask placeholder, the size of the image
            placeholder = np.zeros(config.IMAGE_SHAPE[:2], dtype=bool)
            # GT box
            gt_y1, gt_x1, gt_y2, gt_x2 = gt_boxes[gt_id]
            gt_w = gt_x2 - gt_x1
            gt_h = gt_y2 - gt_y1
            # Resize mini mask to size of GT box
            placeholder[gt_y1:gt_y2, gt_x1:gt_x2] = \
                np.round(utils.resize(class_mask, (gt_h, gt_w))).astype(bool)
            # Place the mini batch in the placeholder
            class_mask = placeholder

        # Pick part of the mask and resize it
        y1, x1, y2, x2 = rois[i].astype(np.int32)
        m = class_mask[y1:y2, x1:x2]
        mask = utils.resize(m, config.MASK_SHAPE)
        masks[i, :, :, class_id] = mask

    return rois, roi_gt_class_ids, bboxes, masks


def build_rpn_targets(image_shape, anchors, gt_class_ids, gt_boxes, config):
    """Given the anchors and GT boxes, compute overlaps and identify positive
    anchors and deltas to refine them to match their corresponding GT boxes.

    anchors: [num_anchors, (y1, x1, y2, x2)]
    gt_class_ids: [num_gt_boxes] Integer class IDs.
    gt_boxes: [num_gt_boxes, (y1, x1, y2, x2)]

    Returns:
    rpn_match: [N] (int32) matches between anchors and GT boxes.
               1 = positive anchor, -1 = negative anchor, 0 = neutral
    rpn_bbox: [N, (dy, dx, log(dh), log(dw))] Anchor bbox deltas.
    """
    # RPN Match: 1 = positive anchor, -1 = negative anchor, 0 = neutral
    rpn_match = np.zeros([anchors.shape[0]], dtype=np.int32)
    # RPN bounding boxes: [max anchors per image, (dy, dx, log(dh), log(dw))]
    rpn_bbox = np.zeros((config.RPN_TRAIN_ANCHORS_PER_IMAGE, 4))

    # Handle COCO crowds
    # A crowd box in COCO is a bounding box around several instances. Exclude
    # them from training. A crowd box is given a negative class ID.
    crowd_ix = np.where(gt_class_ids < 0)[0]
    if crowd_ix.shape[0] > 0:
        # Filter out crowds from ground truth class IDs and boxes
        non_crowd_ix = np.where(gt_class_ids > 0)[0]
        crowd_boxes = gt_boxes[crowd_ix]
        gt_class_ids = gt_class_ids[non_crowd_ix]
        gt_boxes = gt_boxes[non_crowd_ix]
        # Compute overlaps with crowd boxes [anchors, crowds]
        crowd_overlaps = utils.compute_overlaps(anchors, crowd_boxes)
        crowd_iou_max = np.amax(crowd_overlaps, axis=1)
        no_crowd_bool = (crowd_iou_max < 0.001)
    else:
        # All anchors don't intersect a crowd
        no_crowd_bool = np.ones([anchors.shape[0]], dtype=bool)

    # Compute overlaps [num_anchors, num_gt_boxes]
    overlaps = utils.compute_overlaps(anchors, gt_boxes)

    # Match anchors to GT Boxes
    # If an anchor overlaps a GT box with IoU >= 0.7 then it's positive.
    # If an anchor overlaps a GT box with IoU < 0.3 then it's negative.
    # Neutral anchors are those that don't match the conditions above,
    # and they don't influence the loss function.
    # However, don't keep any GT box unmatched (rare, but happens). Instead,
    # match it to the closest anchor (even if its max IoU is < 0.3).
    #
    # 1. Set negative anchors first. They get overwritten below if a GT box is
    # matched to them. Skip boxes in crowd areas.
    anchor_iou_argmax = np.argmax(overlaps, axis=1)
    anchor_iou_max = overlaps[np.arange(overlaps.shape[0]), anchor_iou_argmax]
    rpn_match[(anchor_iou_max < 0.3) & (no_crowd_bool)] = -1
    # 2. Set an anchor for each GT box (regardless of IoU value).
    # If multiple anchors have the same IoU match all of them
    gt_iou_argmax = np.argwhere(overlaps == np.max(overlaps, axis=0))[:, 0]
    rpn_match[gt_iou_argmax] = 1
    # 3. Set anchors with high overlap as positive.
    rpn_match[anchor_iou_max >= 0.7] = 1

    # Subsample to balance positive and negative anchors
    # Don't let positives be more than half the anchors
    ids = np.where(rpn_match == 1)[0]
    extra = len(ids) - (config.RPN_TRAIN_ANCHORS_PER_IMAGE // 2)
    if extra > 0:
        # Reset the extra ones to neutral
        ids = np.random.choice(ids, extra, replace=False)
        rpn_match[ids] = 0
    # Same for negative proposals
    ids = np.where(rpn_match == -1)[0]
    extra = len(ids) - (config.RPN_TRAIN_ANCHORS_PER_IMAGE -
                        np.sum(rpn_match == 1))
    if extra > 0:
        # Rest the extra ones to neutral
        ids = np.random.choice(ids, extra, replace=False)
        rpn_match[ids] = 0

    # For positive anchors, compute shift and scale needed to transform them
    # to match the corresponding GT boxes.
    ids = np.where(rpn_match == 1)[0]
    ix = 0  # index into rpn_bbox
    # TODO: use box_refinement() rather than duplicating the code here
    for i, a in zip(ids, anchors[ids]):
        # Closest gt box (it might have IoU < 0.7)
        gt = gt_boxes[anchor_iou_argmax[i]]

        # Convert coordinates to center plus width/height.
        # GT Box
        gt_h = gt[2] - gt[0]
        gt_w = gt[3] - gt[1]
        gt_center_y = gt[0] + 0.5 * gt_h
        gt_center_x = gt[1] + 0.5 * gt_w
        # Anchor
        a_h = a[2] - a[0]
        a_w = a[3] - a[1]
        a_center_y = a[0] + 0.5 * a_h
        a_center_x = a[1] + 0.5 * a_w

        # Compute the bbox refinement that the RPN should predict.
        rpn_bbox[ix] = [
            (gt_center_y - a_center_y) / a_h,
            (gt_center_x - a_center_x) / a_w,
            np.log(gt_h / a_h),
            np.log(gt_w / a_w),
        ]
        # Normalize
        rpn_bbox[ix] /= config.RPN_BBOX_STD_DEV
        ix += 1

    return rpn_match, rpn_bbox


def generate_random_rois(image_shape, count, gt_class_ids, gt_boxes):
    """Generates ROI proposals similar to what a region proposal network
    would generate.

    image_shape: [Height, Width, Depth]
    count: Number of ROIs to generate
    gt_class_ids: [N] Integer ground truth class IDs
    gt_boxes: [N, (y1, x1, y2, x2)] Ground truth boxes in pixels.

    Returns: [count, (y1, x1, y2, x2)] ROI boxes in pixels.
    """
    # placeholder
    rois = np.zeros((count, 4), dtype=np.int32)

    # Generate random ROIs around GT boxes (90% of count)
    rois_per_box = int(0.9 * count / gt_boxes.shape[0])
    for i in range(gt_boxes.shape[0]):
        gt_y1, gt_x1, gt_y2, gt_x2 = gt_boxes[i]
        h = gt_y2 - gt_y1
        w = gt_x2 - gt_x1
        # random boundaries
        r_y1 = max(gt_y1 - h, 0)
        r_y2 = min(gt_y2 + h, image_shape[0])
        r_x1 = max(gt_x1 - w, 0)
        r_x2 = min(gt_x2 + w, image_shape[1])

        # To avoid generating boxes with zero area, we generate double what
        # we need and filter out the extra. If we get fewer valid boxes
        # than we need, we loop and try again.
        while True:
            y1y2 = np.random.randint(r_y1, r_y2, (rois_per_box * 2, 2))
            x1x2 = np.random.randint(r_x1, r_x2, (rois_per_box * 2, 2))
            # Filter out zero area boxes
            threshold = 1
            y1y2 = y1y2[np.abs(y1y2[:, 0] -
                               y1y2[:, 1]) >= threshold][:rois_per_box]
            x1x2 = x1x2[np.abs(x1x2[:, 0] -
                               x1x2[:, 1]) >= threshold][:rois_per_box]
            if y1y2.shape[0] == rois_per_box and x1x2.shape[0] == rois_per_box:
                break

        # Sort on axis 1 to ensure x1 <= x2 and y1 <= y2 and then reshape
        # into x1, y1, x2, y2 order
        x1, x2 = np.split(np.sort(x1x2, axis=1), 2, axis=1)
        y1, y2 = np.split(np.sort(y1y2, axis=1), 2, axis=1)
        box_rois = np.hstack([y1, x1, y2, x2])
        rois[rois_per_box * i:rois_per_box * (i + 1)] = box_rois

    # Generate random ROIs anywhere in the image (10% of count)
    remaining_count = count - (rois_per_box * gt_boxes.shape[0])
    # To avoid generating boxes with zero area, we generate double what
    # we need and filter out the extra. If we get fewer valid boxes
    # than we need, we loop and try again.
    while True:
        y1y2 = np.random.randint(0, image_shape[0], (remaining_count * 2, 2))
        x1x2 = np.random.randint(0, image_shape[1], (remaining_count * 2, 2))
        # Filter out zero area boxes
        threshold = 1
        y1y2 = y1y2[np.abs(y1y2[:, 0] -
                           y1y2[:, 1]) >= threshold][:remaining_count]
        x1x2 = x1x2[np.abs(x1x2[:, 0] -
                           x1x2[:, 1]) >= threshold][:remaining_count]
        if y1y2.shape[0] == remaining_count and x1x2.shape[
                0] == remaining_count:
            break

    # Sort on axis 1 to ensure x1 <= x2 and y1 <= y2 and then reshape
    # into x1, y1, x2, y2 order
    x1, x2 = np.split(np.sort(x1x2, axis=1), 2, axis=1)
    y1, y2 = np.split(np.sort(y1y2, axis=1), 2, axis=1)
    global_rois = np.hstack([y1, x1, y2, x2])
    rois[-remaining_count:] = global_rois
    return rois


def data_generator(dataset,
                   config,
                   shuffle=True,
                   augment=False,
                   augmentation=None,
                   random_rois=0,
                   batch_size=1,
                   detection_targets=False,
                   no_augmentation_sources=None):
    """A generator that returns images and corresponding target class ids,
    bounding box deltas, and masks.

    dataset: The Dataset object to pick data from
    config: The model config object
    shuffle: If True, shuffles the samples before every epoch
    augment: (deprecated. Use augmentation instead). If true, apply random
        image augmentation. Currently, only horizontal flipping is offered.
    augmentation: Optional. An imgaug (https://github.com/aleju/imgaug) augmentation.
        For example, passing imgaug.augmenters.Fliplr(0.5) flips images
        right/left 50% of the time.
    random_rois: If > 0 then generate proposals to be used to train the
                 network classifier and mask heads. Useful if training
                 the Mask RCNN part without the RPN.
    batch_size: How many images to return in each call
    detection_targets: If True, generate detection targets (class IDs, bbox
        deltas, and masks). Typically for debugging or visualizations because
        in trainig detection targets are generated by DetectionTargetLayer.
    no_augmentation_sources: Optional. List of sources to exclude for
        augmentation. A source is string that identifies a dataset and is
        defined in the Dataset class.

    Returns a Python generator. Upon calling next() on it, the
    generator returns two lists, inputs and outputs. The contents
    of the lists differs depending on the received arguments:
    inputs list:
    - images: [batch, H, W, C]
    - image_meta: [batch, (meta data)] Image details. See compose_image_meta()
    - rpn_match: [batch, N] Integer (1=positive anchor, -1=negative, 0=neutral)
    - rpn_bbox: [batch, N, (dy, dx, log(dh), log(dw))] Anchor bbox deltas.
    - gt_class_ids: [batch, MAX_GT_INSTANCES] Integer class IDs
    - gt_boxes: [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)]
    - gt_masks: [batch, height, width, MAX_GT_INSTANCES]. The height and width
                are those of the image unless use_mini_mask is True, in which
                case they are defined in MINI_MASK_SHAPE.

    outputs list: Usually empty in regular training. But if detection_targets
        is True then the outputs list contains target class_ids, bbox deltas,
        and masks.
    """
    b = 0  # batch item index
    image_index = -1
    image_ids = np.copy(dataset.image_ids)
    error_count = 0
    no_augmentation_sources = no_augmentation_sources or []

    # Anchors
    # [anchor_count, (y1, x1, y2, x2)]
    backbone_shapes = compute_backbone_shapes(config, config.IMAGE_SHAPE)
    anchors = utils.generate_pyramid_anchors(config.RPN_ANCHOR_SCALES,
                                             config.RPN_ANCHOR_RATIOS,
                                             backbone_shapes,
                                             config.BACKBONE_STRIDES,
                                             config.RPN_ANCHOR_STRIDE)

    # Keras requires a generator to run indefinitely.
    while True:
        try:
            # Increment index to pick next image. Shuffle if at the start of an epoch.
            image_index = (image_index + 1) % len(image_ids)
            if shuffle and image_index == 0:
                np.random.shuffle(image_ids)

            # Get GT bounding boxes and masks for image.
            image_id = image_ids[image_index]

            # If the image source is not to be augmented pass None as augmentation
            if dataset.image_info[image_id][
                    'source'] in no_augmentation_sources:
                print("$$$$$$$$$$$$$HERE")
                image, image_meta, gt_class_ids, gt_boxes, gt_masks, gt_start_rows, gt_start_cols, gt_end_rows, gt_end_cols  = \
                load_image_gt(dataset, config, image_id, augment=augment,
                              augmentation=None,
                              use_mini_mask=config.USE_MINI_MASK)
            else:
                image, image_meta, gt_class_ids, gt_boxes, gt_masks, gt_start_rows, gt_start_cols, gt_end_rows, gt_end_cols = \
                    load_image_gt(dataset, config, image_id, augment=augment,
                                augmentation=augmentation,
                                use_mini_mask=config.USE_MINI_MASK)

            #print(str(image))
            #print(str(image_meta))
            # Skip images that have no instances. This can happen in cases
            # where we train on a subset of classes and the image doesn't
            # have any of the classes we care about.
            if not np.any(gt_class_ids > 0):
                continue

            # RPN Targets
            rpn_match, rpn_bbox = build_rpn_targets(image.shape, anchors,
                                                    gt_class_ids, gt_boxes,
                                                    config)

            # Mask R-CNN Targets
            if random_rois:
                rpn_rois = generate_random_rois(image.shape, random_rois,
                                                gt_class_ids, gt_boxes)
                if detection_targets:
                    rois, mrcnn_class_ids, mrcnn_bbox, mrcnn_mask =\
                        build_detection_targets(
                            rpn_rois, gt_class_ids, gt_boxes, gt_masks, config)

            # Init batch arrays
            if b == 0:
                batch_image_meta = np.zeros((batch_size, ) + image_meta.shape,
                                            dtype=image_meta.dtype)
                batch_rpn_match = np.zeros([batch_size, anchors.shape[0], 1],
                                           dtype=rpn_match.dtype)
                batch_rpn_bbox = np.zeros(
                    [batch_size, config.RPN_TRAIN_ANCHORS_PER_IMAGE, 4],
                    dtype=rpn_bbox.dtype)
                batch_images = np.zeros((batch_size, ) + image.shape,
                                        dtype=np.float32)
                batch_gt_class_ids = np.zeros(
                    (batch_size, config.MAX_GT_INSTANCES), dtype=np.int32)
                batch_gt_boxes = np.zeros(
                    (batch_size, config.MAX_GT_INSTANCES, 4), dtype=np.int32)
                batch_gt_masks = np.zeros(
                    (batch_size, gt_masks.shape[0], gt_masks.shape[1],
                     config.MAX_GT_INSTANCES),
                    dtype=gt_masks.dtype)
                batch_gt_start_rows = np.zeros(
                    (batch_size, config.MAX_GT_INSTANCES), dtype=np.int32)
                batch_gt_start_cols = np.zeros(
                    (batch_size, config.MAX_GT_INSTANCES), dtype=np.int32)
                batch_gt_end_rows = np.zeros(
                    (batch_size, config.MAX_GT_INSTANCES), dtype=np.int32)
                batch_gt_end_cols = np.zeros(
                    (batch_size, config.MAX_GT_INSTANCES), dtype=np.int32)

                if random_rois:
                    batch_rpn_rois = np.zeros(
                        (batch_size, rpn_rois.shape[0], 4),
                        dtype=rpn_rois.dtype)
                    if detection_targets:
                        batch_rois = np.zeros((batch_size, ) + rois.shape,
                                              dtype=rois.dtype)
                        batch_mrcnn_class_ids = np.zeros(
                            (batch_size, ) + mrcnn_class_ids.shape,
                            dtype=mrcnn_class_ids.dtype)
                        batch_mrcnn_bbox = np.zeros(
                            (batch_size, ) + mrcnn_bbox.shape,
                            dtype=mrcnn_bbox.dtype)
                        batch_mrcnn_mask = np.zeros(
                            (batch_size, ) + mrcnn_mask.shape,
                            dtype=mrcnn_mask.dtype)

            # If more instances than fits in the array, sub-sample from them.
            if gt_boxes.shape[0] > config.MAX_GT_INSTANCES:
                ids = np.random.choice(np.arange(gt_boxes.shape[0]),
                                       config.MAX_GT_INSTANCES,
                                       replace=False)
                gt_class_ids = gt_class_ids[ids]
                gt_boxes = gt_boxes[ids]
                gt_start_rows = gt_start_rows[ids]
                gt_start_cols = gt_start_cols[ids]
                gt_end_rows = gt_end_rows[ids]
                gt_end_cols = gt_end_cols[ids]
                gt_masks = gt_masks[:, :, ids]
            #print(image_meta)
            # print(gt_boxes[:,0])
            # print(gt_start_rows)
            # print(gt_boxes[:,1])
            # print(gt_start_cols)
            # print(gt_boxes[:,2])
            # print(gt_end_rows)
            # print(gt_boxes[:,3])
            # print(gt_end_cols)
            # Add to batch
            batch_image_meta[b] = image_meta
            batch_rpn_match[b] = rpn_match[:, np.newaxis]
            batch_rpn_bbox[b] = rpn_bbox
            batch_images[b] = mold_image(image.astype(np.float32), config)
            batch_gt_class_ids[b, :gt_class_ids.shape[0]] = gt_class_ids
            batch_gt_start_rows[b, :gt_start_rows.shape[0]] = gt_start_rows
            batch_gt_start_cols[b, :gt_start_cols.shape[0]] = gt_start_cols
            batch_gt_end_rows[b, :gt_end_rows.shape[0]] = gt_end_rows
            batch_gt_end_cols[b, :gt_end_cols.shape[0]] = gt_end_cols
            batch_gt_boxes[b, :gt_boxes.shape[0]] = gt_boxes
            batch_gt_masks[b, :, :, :gt_masks.shape[-1]] = gt_masks
            if random_rois:
                batch_rpn_rois[b] = rpn_rois
                if detection_targets:
                    batch_rois[b] = rois
                    batch_mrcnn_class_ids[b] = mrcnn_class_ids
                    batch_mrcnn_bbox[b] = mrcnn_bbox
                    batch_mrcnn_mask[b] = mrcnn_mask
            b += 1

            # Batch full?
            #print(batch_gt_start_rows)
            if b >= batch_size:
                inputs = [
                    batch_images, batch_image_meta, batch_rpn_match,
                    batch_rpn_bbox, batch_gt_class_ids, batch_gt_boxes,
                    batch_gt_start_rows, batch_gt_start_cols,
                    batch_gt_end_rows, batch_gt_end_cols, batch_gt_masks
                ]
                outputs = []

                if random_rois:
                    inputs.extend([batch_rpn_rois])
                    if detection_targets:
                        inputs.extend([batch_rois])
                        # Keras requires that output and targets have the same number of dimensions
                        batch_mrcnn_class_ids = np.expand_dims(
                            batch_mrcnn_class_ids, -1)
                        outputs.extend([
                            batch_mrcnn_class_ids, batch_mrcnn_bbox,
                            batch_mrcnn_mask
                        ])

                yield inputs, outputs

                # start a new batch
                b = 0
        except (GeneratorExit, KeyboardInterrupt):
            raise
        except:
            # Log it and skip the image
            logging.exception("Error processing image {}".format(
                dataset.image_info[image_id]))
            error_count += 1
            if error_count > 5:
                raise


def generate_sample(dataset,
                    config,
                    shuffle=True,
                    augment=False,
                    augmentation=None,
                    random_rois=0,
                    batch_size=1,
                    detection_targets=False,
                    no_augmentation_sources=None):
    """A generator that returns images and corresponding target class ids,
    bounding box deltas, and masks.

    dataset: The Dataset object to pick data from
    config: The model config object
    shuffle: If True, shuffles the samples before every epoch
    augment: (deprecated. Use augmentation instead). If true, apply random
        image augmentation. Currently, only horizontal flipping is offered.
    augmentation: Optional. An imgaug (https://github.com/aleju/imgaug) augmentation.
        For example, passing imgaug.augmenters.Fliplr(0.5) flips images
        right/left 50% of the time.
    random_rois: If > 0 then generate proposals to be used to train the
                 network classifier and mask heads. Useful if training
                 the Mask RCNN part without the RPN.
    batch_size: How many images to return in each call
    detection_targets: If True, generate detection targets (class IDs, bbox
        deltas, and masks). Typically for debugging or visualizations because
        in trainig detection targets are generated by DetectionTargetLayer.
    no_augmentation_sources: Optional. List of sources to exclude for
        augmentation. A source is string that identifies a dataset and is
        defined in the Dataset class.

    Returns a Python generator. Upon calling next() on it, the
    generator returns two lists, inputs and outputs. The contents
    of the lists differs depending on the received arguments:
    inputs list:
    - images: [batch, H, W, C]
    - image_meta: [batch, (meta data)] Image details. See compose_image_meta()
    - rpn_match: [batch, N] Integer (1=positive anchor, -1=negative, 0=neutral)
    - rpn_bbox: [batch, N, (dy, dx, log(dh), log(dw))] Anchor bbox deltas.
    - gt_class_ids: [batch, MAX_GT_INSTANCES] Integer class IDs
    - gt_boxes: [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)]
    - gt_masks: [batch, height, width, MAX_GT_INSTANCES]. The height and width
                are those of the image unless use_mini_mask is True, in which
                case they are defined in MINI_MASK_SHAPE.

    outputs list: Usually empty in regular training. But if detection_targets
        is True then the outputs list contains target class_ids, bbox deltas,
        and masks.
    """
    b = 0  # batch item index
    image_index = -1
    image_ids = np.copy(dataset.image_ids)
    error_count = 0
    no_augmentation_sources = no_augmentation_sources or []

    # Anchors
    # [anchor_count, (y1, x1, y2, x2)]
    backbone_shapes = compute_backbone_shapes(config, config.IMAGE_SHAPE)
    anchors = utils.generate_pyramid_anchors(config.RPN_ANCHOR_SCALES,
                                             config.RPN_ANCHOR_RATIOS,
                                             backbone_shapes,
                                             config.BACKBONE_STRIDES,
                                             config.RPN_ANCHOR_STRIDE)

    # Keras requires a generator to run indefinitely.
    #while True:
    try:
        # Increment index to pick next image. Shuffle if at the start of an epoch.
        image_index = (image_index + 1) % len(image_ids)
        if shuffle and image_index == 0:
            np.random.shuffle(image_ids)

        # Get GT bounding boxes and masks for image.
        image_id = image_ids[image_index]

        # If the image source is not to be augmented pass None as augmentation
        if dataset.image_info[image_id]['source'] in no_augmentation_sources:
            print("$$$$$$$$$$$$$HERE")
            image, image_meta, gt_class_ids, gt_boxes, gt_masks, gt_start_rows, gt_start_cols, gt_end_rows, gt_end_cols  = \
            load_image_gt(dataset, config, image_id, augment=augment,
                            augmentation=None,
                            use_mini_mask=config.USE_MINI_MASK)
        else:
            image, image_meta, gt_class_ids, gt_boxes, gt_masks, gt_start_rows, gt_start_cols, gt_end_rows, gt_end_cols = \
                load_image_gt(dataset, config, image_id, augment=augment,
                            augmentation=augmentation,
                            use_mini_mask=config.USE_MINI_MASK)

        #print(str(image))
        #print(str(image_meta))
        # Skip images that have no instances. This can happen in cases
        # where we train on a subset of classes and the image doesn't
        # have any of the classes we care about.
        #if not np.any(gt_class_ids > 0):
        #    continue

        # RPN Targets
        rpn_match, rpn_bbox = build_rpn_targets(image.shape, anchors,
                                                gt_class_ids, gt_boxes, config)

        # Mask R-CNN Targets
        if random_rois:
            rpn_rois = generate_random_rois(image.shape, random_rois,
                                            gt_class_ids, gt_boxes)
            if detection_targets:
                rois, mrcnn_class_ids, mrcnn_bbox, mrcnn_mask =\
                    build_detection_targets(
                        rpn_rois, gt_class_ids, gt_boxes, gt_masks, config)
        #print(image.shape)
        # Init batch arrays
        if b == 0:
            batch_image_meta = np.zeros((batch_size, ) + image_meta.shape,
                                        dtype=image_meta.dtype)
            batch_rpn_match = np.zeros([batch_size, anchors.shape[0], 1],
                                       dtype=rpn_match.dtype)
            batch_rpn_bbox = np.zeros(
                [batch_size, config.RPN_TRAIN_ANCHORS_PER_IMAGE, 4],
                dtype=rpn_bbox.dtype)
            batch_images = np.zeros((batch_size, ) + image.shape,
                                    dtype=np.float32)
            batch_gt_class_ids = np.zeros(
                (batch_size, config.MAX_GT_INSTANCES), dtype=np.int32)
            batch_gt_boxes = np.zeros((batch_size, config.MAX_GT_INSTANCES, 4),
                                      dtype=np.int32)
            batch_gt_masks = np.zeros(
                (batch_size, gt_masks.shape[0], gt_masks.shape[1],
                 config.MAX_GT_INSTANCES),
                dtype=gt_masks.dtype)
            batch_gt_start_rows = np.zeros(
                (batch_size, config.MAX_GT_INSTANCES), dtype=np.int32)
            batch_gt_start_cols = np.zeros(
                (batch_size, config.MAX_GT_INSTANCES), dtype=np.int32)
            batch_gt_end_rows = np.zeros((batch_size, config.MAX_GT_INSTANCES),
                                         dtype=np.int32)
            batch_gt_end_cols = np.zeros((batch_size, config.MAX_GT_INSTANCES),
                                         dtype=np.int32)

            if random_rois:
                batch_rpn_rois = np.zeros((batch_size, rpn_rois.shape[0], 4),
                                          dtype=rpn_rois.dtype)
                if detection_targets:
                    batch_rois = np.zeros((batch_size, ) + rois.shape,
                                          dtype=rois.dtype)
                    batch_mrcnn_class_ids = np.zeros(
                        (batch_size, ) + mrcnn_class_ids.shape,
                        dtype=mrcnn_class_ids.dtype)
                    batch_mrcnn_bbox = np.zeros(
                        (batch_size, ) + mrcnn_bbox.shape,
                        dtype=mrcnn_bbox.dtype)
                    batch_mrcnn_mask = np.zeros(
                        (batch_size, ) + mrcnn_mask.shape,
                        dtype=mrcnn_mask.dtype)

        # If more instances than fits in the array, sub-sample from them.
        if gt_boxes.shape[0] > config.MAX_GT_INSTANCES:
            ids = np.random.choice(np.arange(gt_boxes.shape[0]),
                                   config.MAX_GT_INSTANCES,
                                   replace=False)
            gt_class_ids = gt_class_ids[ids]
            gt_boxes = gt_boxes[ids]
            gt_start_rows = gt_start_rows[ids]
            gt_start_cols = gt_start_cols[ids]
            gt_end_rows = gt_end_rows[ids]
            gt_end_cols = gt_end_cols[ids]
            gt_masks = gt_masks[:, :, ids]
        #print(image_meta)
        print(gt_boxes[:, 0])
        print(gt_start_rows)
        print(gt_boxes[:, 1])
        print(gt_start_cols)
        print(gt_boxes[:, 2])
        print(gt_end_rows)
        print(gt_boxes[:, 3])
        print(gt_end_cols)
        # Add to batch
        batch_image_meta[b] = image_meta
        batch_rpn_match[b] = rpn_match[:, np.newaxis]
        batch_rpn_bbox[b] = rpn_bbox
        batch_images[b] = mold_image(image.astype(np.float32), config)
        batch_gt_class_ids[b, :gt_class_ids.shape[0]] = gt_class_ids
        batch_gt_start_rows[b, :gt_start_rows.shape[0]] = gt_start_rows
        batch_gt_start_cols[b, :gt_start_cols.shape[0]] = gt_start_cols
        batch_gt_end_rows[b, :gt_end_rows.shape[0]] = gt_end_rows
        batch_gt_end_cols[b, :gt_end_cols.shape[0]] = gt_end_cols
        batch_gt_boxes[b, :gt_boxes.shape[0]] = gt_boxes
        batch_gt_masks[b, :, :, :gt_masks.shape[-1]] = gt_masks
        if random_rois:
            batch_rpn_rois[b] = rpn_rois
            if detection_targets:
                batch_rois[b] = rois
                batch_mrcnn_class_ids[b] = mrcnn_class_ids
                batch_mrcnn_bbox[b] = mrcnn_bbox
                batch_mrcnn_mask[b] = mrcnn_mask
        b += 1

        # Batch full?
        #print(batch_gt_start_rows)
        if b >= batch_size:
            inputs = [
                batch_images, batch_image_meta, batch_rpn_match,
                batch_rpn_bbox, batch_gt_class_ids, batch_gt_boxes,
                batch_gt_start_rows, batch_gt_start_cols, batch_gt_end_rows,
                batch_gt_end_cols, batch_gt_masks
            ]
            outputs = []

            if random_rois:
                inputs.extend([batch_rpn_rois])
                if detection_targets:
                    inputs.extend([batch_rois])
                    # Keras requires that output and targets have the same number of dimensions
                    batch_mrcnn_class_ids = np.expand_dims(
                        batch_mrcnn_class_ids, -1)
                    outputs.extend([
                        batch_mrcnn_class_ids, batch_mrcnn_bbox,
                        batch_mrcnn_mask
                    ])

            return inputs, outputs

            # start a new batch
            b = 0
    except (GeneratorExit, KeyboardInterrupt):
        raise
    except:
        # Log it and skip the image
        logging.exception("Error processing image {}".format(
            dataset.image_info[image_id]))
        error_count += 1
        if error_count > 5:
            raise


def prepare_graph_features_row(vertex_features,
                               n_neighbours=40,
                               n_dimensions=4,
                               n_filters=42,
                               n_propagate=18):
    blocks = []
    all_features = vertex_features
    for i in range(4):
        gex = GlobalExchange(name='row_graph_gex_%d' % i)(vertex_features)

        dense0 = KL.Dense(128,
                          activation='tanh',
                          name='row_graph_dense_%d-0' % i)(gex)
        dense1 = KL.Dense(128,
                          activation='tanh',
                          name='row_graph_dense_%d-1' % i)(dense0)
        dense2 = KL.Dense(128,
                          activation='tanh',
                          name='row_graph_dense_%d-2' % i)(dense1)

        gravnet = GravNet(n_neighbours,
                          n_dimensions,
                          n_filters,
                          n_propagate,
                          name='row_graph_gravnet_%d' % i)(dense2)

        all_features = KL.concatenate([all_features, gravnet], axis=-1)

    output_dense_0 = KL.Dense(256,
                              activation='relu',
                              name='row_graph_output_0')(all_features)
    output_dense_1 = KL.Dense(32, activation='relu',
                              name='row_graph_output_1')(output_dense_0)
    return output_dense_1


def prepare_graph_features_col(vertex_features,
                               n_neighbours=40,
                               n_dimensions=4,
                               n_filters=42,
                               n_propagate=18):
    blocks = []
    all_features = vertex_features
    for i in range(4):
        gex = GlobalExchange(name='col_graph_gex_%d' % i)(vertex_features)

        dense0 = KL.Dense(128,
                          activation='tanh',
                          name='col_graph_dense_%d-0' % i)(gex)
        dense1 = KL.Dense(128,
                          activation='tanh',
                          name='col_graph_dense_%d-1' % i)(dense0)
        dense2 = KL.Dense(128,
                          activation='tanh',
                          name='col_graph_dense_%d-2' % i)(dense1)

        gravnet = GravNet(n_neighbours,
                          n_dimensions,
                          n_filters,
                          n_propagate,
                          name='col_graph_gravnet_%d' % i)(dense2)

        all_features = KL.concatenate([all_features, gravnet], axis=-1)

    output_dense_0 = KL.Dense(256,
                              activation='relu',
                              name='col_graph_output_0')(all_features)
    output_dense_1 = KL.Dense(128,
                              activation='relu',
                              name='col_graph_output_1')(output_dense_0)
    return output_dense_1


def build_row_adj_classifier(graph_features):
    net = KL.Dense(256, activation='relu',
                   name='graph_row_dense_1')(graph_features)
    net = KL.Dense(256, activation='relu', name='graph_row_dense_2')(net)
    logits = KL.Dense(2, activation='relu', name='graph_row_dense_logits')(net)
    return logits


def build_col_adj_classifier(graph_features):
    net = KL.Dense(256, activation='relu',
                   name='graph_col_dense_1')(graph_features)
    net = KL.Dense(256, activation='relu', name='graph_col_dense_2')(net)
    logits = KL.Dense(2, activation='relu', name='graph_col_dense_logits')(net)
    return logits


############################################################
#  MaskRCNN Class
############################################################


class MaskRCNN():
    """Encapsulates the Mask RCNN model functionality.

    The actual Keras model is in the keras_model property.
    """
    def __init__(self, mode, config, model_dir):
        """
        mode: Either "training" or "inference"
        config: A Sub-class of the Config class
        model_dir: Directory to save training logs and trained weights
        """
        assert mode in ['training', 'inference']
        self.mode = mode
        self.config = config
        self.model_dir = model_dir
        self.set_log_dir()
        self.keras_model = self.build(mode=mode, config=config)

    def build(self, mode, config):
        """Build Mask R-CNN architecture.
            input_shape: The shape of the input image.
            mode: Either "training" or "inference". The inputs and
                outputs of the model differ accordingly.
        """
        assert mode in ['training', 'inference']

        # Image size must be dividable by 2 multiple times
        h, w = config.IMAGE_SHAPE[:2]
        if h / 2**6 != int(h / 2**6) or w / 2**6 != int(w / 2**6):
            raise Exception(
                "Image size must be dividable by 2 at least 6 times "
                "to avoid fractions when downscaling and upscaling."
                "For example, use 256, 320, 384, 448, 512, ... etc. ")

        # Inputs
        input_image = KL.Input(shape=[None, None, config.IMAGE_SHAPE[2]],
                               name="input_image")
        input_image_meta = KL.Input(shape=[config.IMAGE_META_SIZE],
                                    name="input_image_meta")
        if mode == "training":
            # RPN GT
            input_rpn_match = KL.Input(shape=[None, 1],
                                       name="input_rpn_match",
                                       dtype=tf.int32)
            input_rpn_bbox = KL.Input(shape=[None, 4],
                                      name="input_rpn_bbox",
                                      dtype=tf.float32)

            # Detection GT (class IDs, bounding boxes, and masks)
            # 1. GT Class IDs (zero padded)
            input_gt_class_ids = KL.Input(shape=[None],
                                          name="input_gt_class_ids",
                                          dtype=tf.int32)
            # 2. GT Boxes in pixels (zero padded)
            # [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)] in image coordinates
            input_gt_boxes = KL.Input(shape=[None, 4],
                                      name="input_gt_boxes",
                                      dtype=tf.float32)
            # Normalize coordinates
            gt_boxes = KL.Lambda(lambda x: norm_boxes_graph(
                x,
                K.shape(input_image)[1:3]))(input_gt_boxes)
            #structural information
            input_gt_start_rows = KL.Input(shape=[None],
                                           name="input_gt_start_rows",
                                           dtype=tf.int32)
            input_gt_start_cols = KL.Input(shape=[None],
                                           name="input_gt_start_cols",
                                           dtype=tf.int32)
            input_gt_end_rows = KL.Input(shape=[None],
                                         name="input_gt_end_rows",
                                         dtype=tf.int32)
            input_gt_end_cols = KL.Input(shape=[None],
                                         name="input_gt_end_cols",
                                         dtype=tf.int32)
            # 3. GT Masks (zero padded)
            # [batch, height, width, MAX_GT_INSTANCES]
            if config.USE_MINI_MASK:
                input_gt_masks = KL.Input(shape=[
                    config.MINI_MASK_SHAPE[0], config.MINI_MASK_SHAPE[1], None
                ],
                                          name="input_gt_masks",
                                          dtype=bool)
            else:
                input_gt_masks = KL.Input(
                    shape=[config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1], None],
                    name="input_gt_masks",
                    dtype=bool)
        elif mode == "inference":
            # Anchors in normalized coordinates
            input_anchors = KL.Input(shape=[None, 4], name="input_anchors")

        # Build the shared convolutional layers.
        # Bottom-up Layers
        # Returns a list of the last layers of each stage, 5 in total.
        # Don't create the thead (stage 5), so we pick the 4th item in the list.
        if callable(config.BACKBONE):
            _, C2, C3, C4, C5 = config.BACKBONE(input_image,
                                                stage5=True,
                                                train_bn=config.TRAIN_BN)
        else:
            _, C2, C3, C4, C5 = resnet_graph(input_image,
                                             config.BACKBONE,
                                             stage5=True,
                                             train_bn=config.TRAIN_BN)

        # Top-down Layers
        # TODO: add assert to varify feature map sizes match what's in config
        P5TD = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1),
                         name='td_fpn_c5p5td')(C5)
        P4TD = KL.UpSampling2D(size=(2, 2), name="fpn_p5upsampled")(P5TD)
        P3TD = KL.UpSampling2D(size=(2, 2), name="fpn_p4upsampled")(P4TD)
        P2TD = KL.UpSampling2D(size=(2, 2), name="fpn_p3upsampled")(P3TD)

        #P5TD_gated = KL.multiply([P5TD, KL.Reshape((1, 1, config.TOP_DOWN_PYRAMID_SIZE))(KL.Dense(config.TOP_DOWN_PYRAMID_SIZE, name="gate_up_td_fpn_c5p5td")(KL.Dense(16, name="gate_down_td_fpn_c5p5td")(KL.GlobalAveragePooling2D()(P5TD))))])
        #P4TD_gated = KL.multiply([P4TD, KL.Reshape((1, 1, config.TOP_DOWN_PYRAMID_SIZE))(KL.Dense(config.TOP_DOWN_PYRAMID_SIZE, name="gate_up_fpn_p5upsampled")(KL.Dense(16, name="gate_down_fpn_p5upsampled")(KL.GlobalAveragePooling2D()(P4TD))))])
        #P3TD_gated = KL.multiply([P3TD, KL.Reshape((1, 1, config.TOP_DOWN_PYRAMID_SIZE))(KL.Dense(config.TOP_DOWN_PYRAMID_SIZE, name="gate_up_fpn_p4upsampled")(KL.Dense(16, name="gate_down_fpn_p4upsampled")(KL.GlobalAveragePooling2D()(P3TD))))])
        #P2TD_gated = KL.multiply([P2TD, KL.Reshape((1, 1, config.TOP_DOWN_PYRAMID_SIZE))(KL.Dense(config.TOP_DOWN_PYRAMID_SIZE, name="gate_up_fpn_p3upsampled")(KL.Dense(16, name="gate_down_fpn_p3upsampled")(KL.GlobalAveragePooling2D()(P2TD))))])

        P2BU = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1),
                         name='bu_fpn_c2p2bu')(C2)
        P3BU = KL.MaxPooling2D(pool_size=(2, 2), name='bu_fpn_p2bup3bu')(
            KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), strides=1)(P2BU))
        P4BU = KL.MaxPooling2D(pool_size=(2, 2), name='bu_fpn_p3bup4bu')(
            KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), strides=1)(P3BU))
        P5BU = KL.MaxPooling2D(pool_size=(2, 2), name='bu_fpn_p4bup5bu')(
            KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), strides=1)(P4BU))

        #P5BU_gated = KL.multiply([P5BU, KL.Reshape((1, 1, config.TOP_DOWN_PYRAMID_SIZE))(KL.Dense(config.TOP_DOWN_PYRAMID_SIZE, name="gate_up_bu_fpn_p4bup5bu")(KL.Dense(16, name="gate_down_bu_fpn_p4bup5bu")(KL.GlobalAveragePooling2D()(P5BU))))])
        #P4BU_gated = KL.multiply([P4BU, KL.Reshape((1, 1, config.TOP_DOWN_PYRAMID_SIZE))(KL.Dense(config.TOP_DOWN_PYRAMID_SIZE, name="gate_up_bu_fpn_p3bup4bu")(KL.Dense(16, name="gate_down_bu_fpn_p3bup4bu")(KL.GlobalAveragePooling2D()(P4BU))))])
        #P3BU_gated = KL.multiply([P3BU, KL.Reshape((1, 1, config.TOP_DOWN_PYRAMID_SIZE))(KL.Dense(config.TOP_DOWN_PYRAMID_SIZE, name="gate_up_bu_fpn_p2bup3bu")(KL.Dense(16, name="gate_down_bu_fpn_p2bup3bu")(KL.GlobalAveragePooling2D()(P3BU))))])
        #P2BU_gated = KL.multiply([P2BU, KL.Reshape((1, 1, config.TOP_DOWN_PYRAMID_SIZE))(KL.Dense(config.TOP_DOWN_PYRAMID_SIZE, name="gate_up_bu_fpn_c2p2bu")(KL.Dense(16, name="gate_down_bu_fpn_c2p2bu")(KL.GlobalAveragePooling2D()(P2BU))))])

        #P5 = KL.Add(name="fpn_p5add")([
        #    P5TD_gated,
        #    P5BU_gated])
        #P4 = KL.Add(name="fpn_p4add")([
        #    P4TD_gated,
        #    P4BU_gated])
        #P3 = KL.Add(name="fpn_p3add")([
        #    P3TD_gated,
        #    P3BU_gated])
        #P2 = KL.Add(name="fpn_p2add")([
        #    P2TD_gated,
        #    P2BU_gated])

        P5 = KL.Add(name="fpn_p5add")([P5TD, P5BU])
        P4 = KL.Add(name="fpn_p4add")([
            P4TD,
            KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1),
                      name='fpn_c4p4')(C4), P4BU
        ])
        P3 = KL.Add(name="fpn_p3add")([
            P3TD,
            KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1),
                      name='fpn_c3p3')(C3), P3BU
        ])
        P2 = KL.Add(name="fpn_p2add")([P2TD, P2BU])
        #print(tf.shape(P5))
        #print(tf.shape(P4))
        #print(tf.shape(P3))
        #print(tf.shape(P2))

        #P5 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c5p5')(C5)
        #P4 = KL.Add(name="fpn_p4add")([
        #    KL.UpSampling2D(size=(2, 2), name="fpn_p5upsampled")(P5),
        #    KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c4p4')(C4)])
        #P3 = KL.Add(name="fpn_p3add")([
        #    KL.UpSampling2D(size=(2, 2), name="fpn_p4upsampled")(P4),
        #    KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c3p3')(C3)])
        #P2 = KL.Add(name="fpn_p2add")([
        #    KL.UpSampling2D(size=(2, 2), name="fpn_p3upsampled")(P3),
        #    KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c2p2')(C2)])
        # Attach 3x3 conv to all P layers to get the final feature maps.
        P2 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3),
                       padding="SAME",
                       name="fpn_p2")(P2)
        P3 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3),
                       padding="SAME",
                       name="fpn_p3")(P3)
        P4 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3),
                       padding="SAME",
                       name="fpn_p4")(P4)
        P5 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3),
                       padding="SAME",
                       name="fpn_p5")(P5)
        # P6 is used for the 5th anchor scale in RPN. Generated by
        # subsampling from P5 with stride of 2.
        P6 = KL.MaxPooling2D(pool_size=(1, 1), strides=2, name="fpn_p6")(P5)

        # Note that P6 is used in RPN, but not in the classifier heads.
        rpn_feature_maps = [P2, P3, P4, P5, P6]
        mrcnn_feature_maps = [P2, P3, P4, P5]

        # Anchors
        if mode == "training":
            anchors = self.get_anchors(config.IMAGE_SHAPE)
            # Duplicate across the batch dimension because Keras requires it
            # TODO: can this be optimized to avoid duplicating the anchors?
            anchors = np.broadcast_to(anchors,
                                      (config.BATCH_SIZE, ) + anchors.shape)
            # A hack to get around Keras's bad support for constants
            anchors = KL.Lambda(lambda x: tf.Variable(anchors),
                                name="anchors")(input_image)
        else:
            anchors = input_anchors

        # RPN Model
        rpn = build_rpn_model(config.RPN_ANCHOR_STRIDE,
                              len(config.RPN_ANCHOR_RATIOS),
                              config.TOP_DOWN_PYRAMID_SIZE)
        # Loop through pyramid layers
        layer_outputs = []  # list of lists
        for p in rpn_feature_maps:
            layer_outputs.append(rpn([p]))
        # Concatenate layer outputs
        # Convert from list of lists of level outputs to list of lists
        # of outputs across levels.
        # e.g. [[a1, b1, c1], [a2, b2, c2]] => [[a1, a2], [b1, b2], [c1, c2]]
        output_names = ["rpn_class_logits", "rpn_class", "rpn_bbox"]
        outputs = list(zip(*layer_outputs))
        outputs = [
            KL.Concatenate(axis=1, name=n)(list(o))
            for o, n in zip(outputs, output_names)
        ]

        rpn_class_logits, rpn_class, rpn_bbox = outputs

        # Generate proposals
        # Proposals are [batch, N, (y1, x1, y2, x2)] in normalized coordinates
        # and zero padded.
        proposal_count = config.POST_NMS_ROIS_TRAINING if mode == "training"\
            else config.POST_NMS_ROIS_INFERENCE
        rpn_rois = ProposalLayer(proposal_count=proposal_count,
                                 nms_threshold=config.RPN_NMS_THRESHOLD,
                                 name="ROI",
                                 config=config)([rpn_class, rpn_bbox, anchors])

        if mode == "training":
            # Class ID mask to mark class IDs supported by the dataset the image
            # came from.
            active_class_ids = KL.Lambda(lambda x: parse_image_meta_graph(x)[
                "active_class_ids"])(input_image_meta)

            if not config.USE_RPN_ROIS:
                # Ignore predicted ROIs and use ROIs provided as an input.
                input_rois = KL.Input(shape=[config.POST_NMS_ROIS_TRAINING, 4],
                                      name="input_roi",
                                      dtype=np.int32)
                # Normalize coordinates
                target_rois = KL.Lambda(lambda x: norm_boxes_graph(
                    x,
                    K.shape(input_image)[1:3]))(input_rois)
            else:
                target_rois = rpn_rois

            # Generate detection targets
            # Subsamples proposals and generates target outputs for training
            # Note that proposal class IDs, gt_boxes, and gt_masks are zero
            # padded. Equally, returned rois and targets are zero padded.
            rois, target_class_ids, target_bbox, target_mask =\
                DetectionTargetLayer(config, name="proposal_targets")([
                    target_rois, input_gt_class_ids, gt_boxes, input_gt_masks])

            # Network Heads
            # TODO: verify that this handles zero padded ROIs
            mrcnn_class_logits, mrcnn_class, mrcnn_bbox =\
                fpn_classifier_graph(rois, mrcnn_feature_maps, input_image_meta,
                                     config.POOL_SIZE, config.NUM_CLASSES,
                                     train_bn=config.TRAIN_BN,
                                     fc_layers_size=config.FPN_CLASSIF_FC_LAYERS_SIZE)

            mrcnn_mask = build_fpn_mask_graph(rois,
                                              mrcnn_feature_maps,
                                              input_image_meta,
                                              config.MASK_POOL_SIZE,
                                              config.NUM_CLASSES,
                                              train_bn=config.TRAIN_BN)

            detection_boxes, detection_boxes_denorm, roi_gt_boxes, stuct_info, adj_rows, adj_cols, vertex_features = StructureDetectionLayer(
                config, name="mrcnn_detection")([
                    rois, mrcnn_class, mrcnn_bbox, input_image_meta,
                    input_gt_class_ids, gt_boxes, input_gt_start_rows,
                    input_gt_start_cols, input_gt_end_rows, input_gt_end_cols,
                    P2
                ])
            vertex_features = KL.concatenate(
                [vertex_features, detection_boxes_denorm])
            aggregate_vertex_features_row = prepare_graph_features_row(
                vertex_features,
                n_neighbours=config.GRAPH_NEIGHBORS,
                n_dimensions=8,
                n_filters=64,
                n_propagate=32)
            aggregate_vertex_features_col = prepare_graph_features_col(
                vertex_features,
                n_neighbours=config.GRAPH_NEIGHBORS,
                n_dimensions=8,
                n_filters=64,
                n_propagate=32)
            graph_features_row = KL.Lambda(lambda x: KL.concatenate([
                K.tile(K.expand_dims(x, 1),
                       [1, config.TRAIN_ROIS_PER_IMAGE, 1, 1]),
                K.tile(K.expand_dims(x, 2),
                       [1, 1, config.TRAIN_ROIS_PER_IMAGE, 1])
            ]))(aggregate_vertex_features_row)
            graph_features_t_row = KL.Lambda(lambda x: K.permute_dimensions(
                x, (0, 2, 1, 3)))(graph_features_row)
            graph_features_row = KL.Lambda(
                lambda x: KL.concatenate([x[0], x[1]]))(
                    [graph_features_row, graph_features_t_row])
            graph_features_col = KL.Lambda(lambda x: KL.concatenate([
                K.tile(K.expand_dims(x, 1),
                       [1, config.TRAIN_ROIS_PER_IMAGE, 1, 1]),
                K.tile(K.expand_dims(x, 2),
                       [1, 1, config.TRAIN_ROIS_PER_IMAGE, 1])
            ]))(aggregate_vertex_features_col)
            graph_features_t_col = KL.Lambda(lambda x: K.permute_dimensions(
                x, (0, 2, 1, 3)))(graph_features_col)
            graph_features_col = KL.Lambda(
                lambda x: KL.concatenate([x[0], x[1]]))(
                    [graph_features_col, graph_features_t_col])
            #graph_features = KL.concatenate([K.tile(K.expand_dims(aggregate_vertex_features, 1), [1, config.TRAIN_ROIS_PER_IMAGE, 1, 1]), K.tile(K.expand_dims(aggregate_vertex_features, 2), [1, 1, config.TRAIN_ROIS_PER_IMAGE, 1])])
            row_adj_logits = build_row_adj_classifier(graph_features_row)
            col_adj_logits = build_col_adj_classifier(graph_features_col)
            output_rois = KL.Lambda(lambda x: x * 1, name="output_rois")(rois)

            # Losses
            rpn_class_loss = KL.Lambda(lambda x: rpn_class_loss_graph(*x),
                                       name="rpn_class_loss")(
                                           [input_rpn_match, rpn_class_logits])
            rpn_bbox_loss = KL.Lambda(
                lambda x: rpn_bbox_loss_graph(config, *x),
                name="rpn_bbox_loss")(
                    [input_rpn_bbox, input_rpn_match, rpn_bbox])
            class_loss = KL.Lambda(lambda x: mrcnn_class_loss_graph(*x),
                                   name="mrcnn_class_loss")([
                                       target_class_ids, mrcnn_class_logits,
                                       active_class_ids
                                   ])
            bbox_loss = KL.Lambda(lambda x: mrcnn_bbox_loss_graph(*x),
                                  name="mrcnn_bbox_loss")([
                                      target_bbox, target_class_ids, mrcnn_bbox
                                  ])
            mask_loss = KL.Lambda(lambda x: mrcnn_mask_loss_graph(*x),
                                  name="mrcnn_mask_loss")([
                                      target_mask, target_class_ids, mrcnn_mask
                                  ])
            roi_alignment_loss = KL.Lambda(
                lambda x: rpn_bbox_structural_loss_graph(*x),
                name="roi_alignment_loss")(
                    [detection_boxes, roi_gt_boxes, stuct_info])
            row_adj_loss = KL.Lambda(lambda x: adjacency_loss(*x),
                                     name="row_adj_loss")(
                                         [adj_rows, row_adj_logits])
            col_adj_loss = KL.Lambda(lambda x: adjacency_loss(*x),
                                     name="col_adj_loss")(
                                         [adj_cols, col_adj_logits])

            # Model
            inputs = [
                input_image, input_image_meta, input_rpn_match, input_rpn_bbox,
                input_gt_class_ids, input_gt_boxes, input_gt_start_rows,
                input_gt_start_cols, input_gt_end_rows, input_gt_end_cols,
                input_gt_masks
            ]
            if not config.USE_RPN_ROIS:
                inputs.append(input_rois)
            outputs = [
                rpn_class_logits, rpn_class, rpn_bbox, mrcnn_class_logits,
                mrcnn_class, mrcnn_bbox, mrcnn_mask, rpn_rois, output_rois,
                detection_boxes, rpn_class_loss, rpn_bbox_loss, class_loss,
                bbox_loss, mask_loss, roi_alignment_loss, row_adj_logits,
                col_adj_logits, row_adj_loss, col_adj_loss
            ]
            model = KM.Model(inputs, outputs, name='mask_rcnn')
        else:
            # Network Heads
            # Proposal classifier and BBox regressor heads
            mrcnn_class_logits, mrcnn_class, mrcnn_bbox =\
                fpn_classifier_graph(rpn_rois, mrcnn_feature_maps, input_image_meta,
                                     config.POOL_SIZE, config.NUM_CLASSES,
                                     train_bn=config.TRAIN_BN,
                                     fc_layers_size=config.FPN_CLASSIF_FC_LAYERS_SIZE)

            # Detections
            # output is [batch, num_detections, (y1, x1, y2, x2, class_id, score)] in
            # normalized coordinates
            detections, pred_boxes_denorm, vertex_features = DetectionLayer(
                config, name="mrcnn_detection")(
                    [rpn_rois, mrcnn_class, mrcnn_bbox, input_image_meta, P2])
            vertex_features = KL.concatenate(
                [vertex_features, pred_boxes_denorm], axis=-1)

            aggregate_vertex_features_row = prepare_graph_features_row(
                vertex_features,
                n_neighbours=config.GRAPH_NEIGHBORS,
                n_dimensions=8,
                n_filters=64,
                n_propagate=32)
            aggregate_vertex_features_col = prepare_graph_features_col(
                vertex_features,
                n_neighbours=config.GRAPH_NEIGHBORS,
                n_dimensions=8,
                n_filters=64,
                n_propagate=32)
            graph_features_row = KL.Lambda(lambda x: KL.concatenate([
                K.tile(K.expand_dims(x, 1),
                       [1, config.DETECTION_MAX_INSTANCES, 1, 1]),
                K.tile(K.expand_dims(x, 2),
                       [1, 1, config.DETECTION_MAX_INSTANCES, 1])
            ]))(aggregate_vertex_features_row)
            graph_features_t_row = KL.Lambda(lambda x: K.permute_dimensions(
                x, (0, 2, 1, 3)))(graph_features_row)
            graph_features_row = KL.Lambda(
                lambda x: KL.concatenate([x[0], x[1]]))(
                    [graph_features_row, graph_features_t_row])
            graph_features_col = KL.Lambda(lambda x: KL.concatenate([
                K.tile(K.expand_dims(x, 1),
                       [1, config.DETECTION_MAX_INSTANCES, 1, 1]),
                K.tile(K.expand_dims(x, 2),
                       [1, 1, config.DETECTION_MAX_INSTANCES, 1])
            ]))(aggregate_vertex_features_col)
            graph_features_t_col = KL.Lambda(lambda x: K.permute_dimensions(
                x, (0, 2, 1, 3)))(graph_features_col)
            graph_features_col = KL.Lambda(
                lambda x: KL.concatenate([x[0], x[1]]))(
                    [graph_features_row, graph_features_t_col])
            #graph_features = KL.concatenate([K.tile(K.expand_dims(aggregate_vertex_features, 1), [1, config.TRAIN_ROIS_PER_IMAGE, 1, 1]), K.tile(K.expand_dims(aggregate_vertex_features, 2), [1, 1, config.TRAIN_ROIS_PER_IMAGE, 1])])
            row_adj_logits = build_row_adj_classifier(graph_features_row)
            col_adj_logits = build_col_adj_classifier(graph_features_col)

            row_adj = KL.Lambda(lambda x: K.argmax(x),
                                name="row_adj")(row_adj_logits)
            col_adj = KL.Lambda(lambda x: K.argmax(x),
                                name="col_adj")(col_adj_logits)
            # Create masks for detections
            detection_boxes = KL.Lambda(lambda x: x[..., :4])(detections)
            mrcnn_mask = build_fpn_mask_graph(detection_boxes,
                                              mrcnn_feature_maps,
                                              input_image_meta,
                                              config.MASK_POOL_SIZE,
                                              config.NUM_CLASSES,
                                              train_bn=config.TRAIN_BN)

            model = KM.Model([input_image, input_image_meta, input_anchors], [
                detections, mrcnn_class, mrcnn_bbox, mrcnn_mask, rpn_rois,
                rpn_class, rpn_bbox, row_adj, col_adj
            ],
                             name='mask_rcnn')

        # Add multi-GPU support.
        if config.GPU_COUNT > 1:
            from mrcnn.parallel_model import ParallelModel
            model = ParallelModel(model, config.GPU_COUNT)

        return model

    def find_last(self):
        """Finds the last checkpoint file of the last trained model in the
        model directory.
        Returns:
            The path of the last checkpoint file
        """
        # Get directory names. Each directory corresponds to a model
        dir_names = next(os.walk(self.model_dir))[1]
        key = self.config.NAME.lower()
        dir_names = filter(lambda f: f.startswith(key), dir_names)
        dir_names = sorted(dir_names)
        if not dir_names:
            import errno
            raise FileNotFoundError(
                errno.ENOENT, "Could not find model directory under {}".format(
                    self.model_dir))
        # Pick last directory
        dir_name = os.path.join(self.model_dir, dir_names[-1])
        # Find the last checkpoint
        checkpoints = next(os.walk(dir_name))[2]
        checkpoints = filter(lambda f: f.startswith("mask_rcnn"), checkpoints)
        checkpoints = sorted(checkpoints)
        if not checkpoints:
            import errno
            raise FileNotFoundError(
                errno.ENOENT,
                "Could not find weight files in {}".format(dir_name))
        checkpoint = os.path.join(dir_name, checkpoints[-1])
        return checkpoint

    def load_weights(self, filepath, by_name=False, exclude=None):
        """Modified version of the corresponding Keras function with
        the addition of multi-GPU support and the ability to exclude
        some layers from loading.
        exclude: list of layer names to exclude
        """
        import h5py
        # Conditional import to support versions of Keras before 2.2
        # TODO: remove in about 6 months (end of 2018)
        try:
            from keras.engine import saving
        except ImportError:
            # Keras before 2.2 used the 'topology' namespace.
            from keras.engine import topology as saving

        if exclude:
            by_name = True

        if h5py is None:
            raise ImportError('`load_weights` requires h5py.')
        f = h5py.File(filepath, mode='r')
        if 'layer_names' not in f.attrs and 'model_weights' in f:
            f = f['model_weights']

        # In multi-GPU training, we wrap the model. Get layers
        # of the inner model because they have the weights.
        keras_model = self.keras_model
        layers = keras_model.inner_model.layers if hasattr(keras_model, "inner_model")\
            else keras_model.layers

        # Exclude some layers
        if exclude:
            layers = filter(lambda l: l.name not in exclude, layers)

        if by_name:
            saving.load_weights_from_hdf5_group_by_name(f, layers)
        else:
            saving.load_weights_from_hdf5_group(f, layers)
        if hasattr(f, 'close'):
            f.close()

        # Update the log directory
        self.set_log_dir(filepath)

    def get_imagenet_weights(self):
        """Downloads ImageNet trained weights from Keras.
        Returns path to weights file.
        """
        from keras.utils.data_utils import get_file
        TF_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/'\
                                 'releases/download/v0.2/'\
                                 'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
        weights_path = get_file(
            'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
            TF_WEIGHTS_PATH_NO_TOP,
            cache_subdir='models',
            md5_hash='a268eb855778b3df3c7506639542a6af')
        return weights_path

    def compile(self, learning_rate, momentum):
        """Gets the model ready for training. Adds losses, regularization, and
        metrics. Then calls the Keras compile() function.
        """
        # Optimizer object
        optimizer = keras.optimizers.SGD(
            lr=learning_rate,
            momentum=momentum,
            clipnorm=self.config.GRADIENT_CLIP_NORM)
        # Add Losses
        # First, clear previously set losses to avoid duplication
        self.keras_model._losses = []
        self.keras_model._per_input_losses = {}
        loss_names = [
            "rpn_class_loss", "rpn_bbox_loss", "mrcnn_class_loss",
            "mrcnn_bbox_loss", "mrcnn_mask_loss", "roi_alignment_loss",
            "row_adj_loss", "col_adj_loss"
        ]
        for name in loss_names:
            layer = self.keras_model.get_layer(name)
            if layer.output in self.keras_model.losses:
                continue
            loss = (tf.reduce_mean(layer.output) *
                    self.config.LOSS_WEIGHTS.get(name, 1.))
            self.keras_model.add_loss(loss)

        # Add L2 Regularization
        # Skip gamma and beta weights of batch normalization layers.
        reg_losses = [
            keras.regularizers.l2(self.config.WEIGHT_DECAY)(w) /
            tf.cast(tf.size(w), tf.float32)
            for w in self.keras_model.trainable_weights
            if 'gamma' not in w.name and 'beta' not in w.name
        ]
        self.keras_model.add_loss(tf.add_n(reg_losses))

        # Compile
        self.keras_model.compile(optimizer=optimizer,
                                 loss=[None] * len(self.keras_model.outputs))

        # Add metrics for losses
        for name in loss_names:
            if name in self.keras_model.metrics_names:
                continue
            layer = self.keras_model.get_layer(name)
            self.keras_model.metrics_names.append(name)
            loss = (tf.reduce_mean(layer.output) *
                    self.config.LOSS_WEIGHTS.get(name, 1.))
            self.keras_model.metrics_tensors.append(loss)

    def set_trainable(self,
                      layer_regex,
                      keras_model=None,
                      indent=0,
                      verbose=1):
        """Sets model layers as trainable if their names match
        the given regular expression.
        """
        # Print message on the first call (but not on recursive calls)
        if verbose > 0 and keras_model is None:
            log("Selecting layers to train")

        keras_model = keras_model or self.keras_model

        # In multi-GPU training, we wrap the model. Get layers
        # of the inner model because they have the weights.
        layers = keras_model.inner_model.layers if hasattr(keras_model, "inner_model")\
            else keras_model.layers

        for layer in layers:
            # Is the layer a model?
            if layer.__class__.__name__ == 'Model':
                print("In model: ", layer.name)
                self.set_trainable(layer_regex,
                                   keras_model=layer,
                                   indent=indent + 4)
                continue

            if not layer.weights:
                continue
            # Is it trainable?
            trainable = bool(re.fullmatch(layer_regex, layer.name))
            # Update layer. If layer is a container, update inner layer.
            if layer.__class__.__name__ == 'TimeDistributed':
                layer.layer.trainable = trainable
            else:
                layer.trainable = trainable
            # Print trainable layer names
            if trainable and verbose > 0:
                log("{}{:20}   ({})".format(" " * indent, layer.name,
                                            layer.__class__.__name__))

    def set_log_dir(self, model_path=None):
        """Sets the model log directory and epoch counter.

        model_path: If None, or a format different from what this code uses
            then set a new log directory and start epochs from 0. Otherwise,
            extract the log directory and the epoch counter from the file
            name.
        """
        # Set date and epoch counter as if starting a new model
        self.epoch = 0
        now = datetime.datetime.now()

        # If we have a model path with date and epochs use them
        if model_path:
            # Continue from we left of. Get epoch and date from the file name
            # A sample model path might look like:
            # \path\to\logs\coco20171029T2315\mask_rcnn_coco_0001.h5 (Windows)
            # /path/to/logs/coco20171029T2315/mask_rcnn_coco_0001.h5 (Linux)
            regex = r".*[/\\][\w-]+(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})[/\\]mask\_rcnn\_[\w-]+(\d{4})\.h5"
            m = re.match(regex, model_path)
            if m:
                now = datetime.datetime(int(m.group(1)), int(m.group(2)),
                                        int(m.group(3)), int(m.group(4)),
                                        int(m.group(5)))
                # Epoch number in file is 1-based, and in Keras code it's 0-based.
                # So, adjust for that then increment by one to start from the next epoch
                self.epoch = int(m.group(6)) - 1 + 1
                print('Re-starting from epoch %d' % self.epoch)

        # Directory for training logs
        self.log_dir = os.path.join(
            self.model_dir, "{}{:%Y%m%dT%H%M}".format(self.config.NAME.lower(),
                                                      now))

        # Path to save after each epoch. Include placeholders that get filled by Keras.
        self.checkpoint_path = os.path.join(
            self.log_dir,
            "mask_rcnn_{}_*epoch*.h5".format(self.config.NAME.lower()))
        self.checkpoint_path = self.checkpoint_path.replace(
            "*epoch*", "{epoch:04d}")

    def train(self,
              train_dataset,
              val_dataset,
              learning_rate,
              epochs,
              layers,
              augmentation=None,
              custom_callbacks=None,
              no_augmentation_sources=None):
        """Train the model.
        train_dataset, val_dataset: Training and validation Dataset objects.
        learning_rate: The learning rate to train with
        epochs: Number of training epochs. Note that previous training epochs
                are considered to be done alreay, so this actually determines
                the epochs to train in total rather than in this particaular
                call.
        layers: Allows selecting wich layers to train. It can be:
            - A regular expression to match layer names to train
            - One of these predefined values:
              heads: The RPN, classifier and mask heads of the network
              all: All the layers
              3+: Train Resnet stage 3 and up
              4+: Train Resnet stage 4 and up
              5+: Train Resnet stage 5 and up
        augmentation: Optional. An imgaug (https://github.com/aleju/imgaug)
            augmentation. For example, passing imgaug.augmenters.Fliplr(0.5)
            flips images right/left 50% of the time. You can pass complex
            augmentations as well. This augmentation applies 50% of the
            time, and when it does it flips images right/left half the time
            and adds a Gaussian blur with a random sigma in range 0 to 5.

                augmentation = imgaug.augmenters.Sometimes(0.5, [
                    imgaug.augmenters.Fliplr(0.5),
                    imgaug.augmenters.GaussianBlur(sigma=(0.0, 5.0))
                ])
	    custom_callbacks: Optional. Add custom callbacks to be called
	        with the keras fit_generator method. Must be list of type keras.callbacks.
        no_augmentation_sources: Optional. List of sources to exclude for
            augmentation. A source is string that identifies a dataset and is
            defined in the Dataset class.
        """
        assert self.mode == "training", "Create model in training mode."

        # Pre-defined layer regular expressions
        layer_regex = {
            # all layers but the backbone
            "heads":
            r"(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)|(gate\_.*)|(graph\_.*)|(row\_.*)|(col\_.*)",
            # From a specific Resnet stage and up
            "3+":
            r"(res3.*)|(bn3.*)|(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)(td\_.*)|(bu\_.*)|(gate\_.*)|(graph\_.*)|(row\_.*)|(col\_.*)",
            "4+":
            r"(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)(td\_.*)|(bu\_.*)|(gate\_.*)|(graph\_.*)|(row\_.*)|(col\_.*)",
            "5+":
            r"(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)(td\_.*)|(bu\_.*)|(gate\_.*)|(graph\_.*)|(row\_.*)|(col\_.*)",
            # All layers
            "all": ".*",
        }
        if layers in layer_regex.keys():
            layers = layer_regex[layers]

        # Data generators
        train_generator = data_generator(
            train_dataset,
            self.config,
            shuffle=True,
            augmentation=augmentation,
            batch_size=self.config.BATCH_SIZE,
            no_augmentation_sources=no_augmentation_sources)
        #x,y = generate_sample(train_dataset, self.config, shuffle=True,
        #                                 augmentation=augmentation,
        #                                 batch_size=self.config.BATCH_SIZE,
        #                                 no_augmentation_sources=no_augmentation_sources)

        val_generator = data_generator(val_dataset,
                                       self.config,
                                       shuffle=True,
                                       batch_size=self.config.BATCH_SIZE)

        # Create log_dir if it does not exist
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        # Callbacks
        callbacks = [
            keras.callbacks.TensorBoard(log_dir=self.log_dir,
                                        histogram_freq=0,
                                        write_graph=True,
                                        write_images=False),
            keras.callbacks.ModelCheckpoint(self.checkpoint_path,
                                            verbose=0,
                                            save_weights_only=True),
        ]

        # Add custom callbacks to the list
        if custom_callbacks:
            callbacks += custom_callbacks

        # Train
        log("\nStarting at epoch {}. LR={}\n".format(self.epoch,
                                                     learning_rate))
        log("Checkpoint Path: {}".format(self.checkpoint_path))
        self.set_trainable(layers)
        self.compile(learning_rate, self.config.LEARNING_MOMENTUM)

        # Work-around for Windows: Keras fails on Windows when using
        # multiprocessing workers. See discussion here:
        # https://github.com/matterport/Mask_RCNN/issues/13#issuecomment-353124009
        if os.name is 'nt':
            workers = 0
        else:
            workers = 0  #25#multiprocessing.cpu_count()
        #outputs = [self.keras_model.get_layer("roi_alignment_loss").output]
        #functor = K.function([self.keras_model.inputs, K.learning_phase()], outputs)
        #layer_outs = functor(*x)
        #print(layer_outs)
        #preds = self.keras_model.predict(x)

        self.keras_model.fit_generator(
            train_generator,
            initial_epoch=self.epoch,
            epochs=epochs,
            steps_per_epoch=self.config.STEPS_PER_EPOCH,
            callbacks=callbacks,
            validation_data=val_generator,
            validation_steps=self.config.VALIDATION_STEPS,
            max_queue_size=100,  #25
            workers=workers,
            use_multiprocessing=False,
        )
        #preds = self.keras_model.predict(x)
        #print(preds[16])
        #print(preds[17])
        #print(preds[18])
        #print(preds[19])
        #print(preds[19].shape)
        #print(preds[20])
        #print(preds[20].shape)
        #print(preds[21])
        #print(preds[21].shape)
        #print(preds[22])
        #print(preds[23])
        self.epoch = max(self.epoch, epochs)

    def mold_inputs(self, images):
        """Takes a list of images and modifies them to the format expected
        as an input to the neural network.
        images: List of image matrices [height,width,depth]. Images can have
            different sizes.

        Returns 3 Numpy matrices:
        molded_images: [N, h, w, 3]. Images resized and normalized.
        image_metas: [N, length of meta data]. Details about each image.
        windows: [N, (y1, x1, y2, x2)]. The portion of the image that has the
            original image (padding excluded).
        """
        molded_images = []
        image_metas = []
        windows = []
        for image in images:
            # Resize image
            # TODO: move resizing to mold_image()
            molded_image, window, scale, padding, crop = utils.resize_image(
                image,
                min_dim=self.config.IMAGE_MIN_DIM,
                min_scale=self.config.IMAGE_MIN_SCALE,
                max_dim=self.config.IMAGE_MAX_DIM,
                mode=self.config.IMAGE_RESIZE_MODE)
            molded_image = mold_image(molded_image, self.config)
            # Build image_meta
            image_meta = compose_image_meta(
                0, image.shape, molded_image.shape, window, scale,
                np.zeros([self.config.NUM_CLASSES], dtype=np.int32))
            # Append
            molded_images.append(molded_image)
            windows.append(window)
            image_metas.append(image_meta)
        # Pack into arrays
        molded_images = np.stack(molded_images)
        image_metas = np.stack(image_metas)
        windows = np.stack(windows)
        return molded_images, image_metas, windows

    def unmold_detections(self, detections, row_adj, col_adj, mrcnn_mask,
                          original_image_shape, image_shape, window):
        """Reformats the detections of one image from the format of the neural
        network output to a format suitable for use in the rest of the
        application.

        detections: [N, (y1, x1, y2, x2, class_id, score)] in normalized coordinates
        mrcnn_mask: [N, height, width, num_classes]
        original_image_shape: [H, W, C] Original image shape before resizing
        image_shape: [H, W, C] Shape of the image after resizing and padding
        window: [y1, x1, y2, x2] Pixel coordinates of box in the image where the real
                image is excluding the padding.

        Returns:
        boxes: [N, (y1, x1, y2, x2)] Bounding boxes in pixels
        class_ids: [N] Integer class IDs for each bounding box
        scores: [N] Float probability scores of the class_id
        masks: [height, width, num_instances] Instance masks
        """
        # How many detections do we have?
        # Detections array is padded with zeros. Find the first class_id == 0.
        zero_ix = np.where(detections[:, 4] == 0)[0]
        N = zero_ix[0] if zero_ix.shape[0] > 0 else detections.shape[0]

        # Extract boxes, class_ids, scores, and class-specific masks
        boxes = detections[:N, :4]
        row_adj = row_adj[:N, :N]
        col_adj = col_adj[:N, :N]
        class_ids = detections[:N, 4].astype(np.int32)
        scores = detections[:N, 5]
        masks = mrcnn_mask[np.arange(N), :, :, class_ids]

        # Translate normalized coordinates in the resized image to pixel
        # coordinates in the original image before resizing
        window = utils.norm_boxes(window, image_shape[:2])
        wy1, wx1, wy2, wx2 = window
        shift = np.array([wy1, wx1, wy1, wx1])
        wh = wy2 - wy1  # window height
        ww = wx2 - wx1  # window width
        scale = np.array([wh, ww, wh, ww])
        # Convert boxes to normalized coordinates on the window
        boxes = np.divide(boxes - shift, scale)
        # Convert boxes to pixel coordinates on the original image
        boxes = utils.denorm_boxes(boxes, original_image_shape[:2])

        # Filter out detections with zero area. Happens in early training when
        # network weights are still random
        exclude_ix = np.where((boxes[:, 2] - boxes[:, 0]) *
                              (boxes[:, 3] - boxes[:, 1]) <= 0)[0]
        if exclude_ix.shape[0] > 0:
            row_adj = np.delete(row_adj, exclude_ix, axis=0)
            row_adj = np.delete(row_adj, exclude_ix, axis=1)
            col_adj = np.delete(col_adj, exclude_ix, axis=0)
            col_adj = np.delete(col_adj, exclude_ix, axis=1)
            boxes = np.delete(boxes, exclude_ix, axis=0)
            class_ids = np.delete(class_ids, exclude_ix, axis=0)
            scores = np.delete(scores, exclude_ix, axis=0)
            masks = np.delete(masks, exclude_ix, axis=0)
            N = class_ids.shape[0]

        # Resize masks to original image size and set boundary threshold.
        full_masks = []
        for i in range(N):
            # Convert neural network mask to full size mask
            full_mask = utils.unmold_mask(masks[i], boxes[i],
                                          original_image_shape)
            full_masks.append(full_mask)
        full_masks = np.stack(full_masks, axis=-1)\
            if full_masks else np.empty(original_image_shape[:2] + (0,))

        return boxes, row_adj, col_adj, class_ids, scores, full_masks

    def detect(self, images, verbose=0):
        """Runs the detection pipeline.

        images: List of images, potentially of different sizes.

        Returns a list of dicts, one dict per image. The dict contains:
        rois: [N, (y1, x1, y2, x2)] detection bounding boxes
        class_ids: [N] int class IDs
        scores: [N] float probability scores for the class IDs
        masks: [H, W, N] instance binary masks
        """
        assert self.mode == "inference", "Create model in inference mode."
        assert len(
            images
        ) == self.config.BATCH_SIZE, "len(images) must be equal to BATCH_SIZE"

        if verbose:
            log("Processing {} images".format(len(images)))
            for image in images:
                log("image", image)

        # Mold inputs to format expected by the neural network
        molded_images, image_metas, windows = self.mold_inputs(images)

        # Validate image sizes
        # All images in a batch MUST be of the same size
        image_shape = molded_images[0].shape
        for g in molded_images[1:]:
            assert g.shape == image_shape,\
                "After resizing, all images must have the same size. Check IMAGE_RESIZE_MODE and image sizes."

        # Anchors
        anchors = self.get_anchors(image_shape)
        # Duplicate across the batch dimension because Keras requires it
        # TODO: can this be optimized to avoid duplicating the anchors?
        anchors = np.broadcast_to(anchors,
                                  (self.config.BATCH_SIZE, ) + anchors.shape)

        if verbose:
            log("molded_images", molded_images)
            log("image_metas", image_metas)
            log("anchors", anchors)
        # Run object detection
        detections, _, _, mrcnn_mask, _, _, _, row_adj, col_adj =\
            self.keras_model.predict([molded_images, image_metas, anchors], verbose=0)
        # Process detections
        results = []
        for i, image in enumerate(images):
            final_rois, final_row_adj, final_col_adj, final_class_ids, final_scores, final_masks =\
                self.unmold_detections(detections[i], row_adj[i], col_adj[i], mrcnn_mask[i],
                                       image.shape, molded_images[i].shape,
                                       windows[i])
            results.append({
                "rois": final_rois,
                "class_ids": final_class_ids,
                "scores": final_scores,
                "masks": final_masks,
                "row_adj": final_row_adj,
                "col_adj": final_col_adj
            })
        return results

    def detect_molded(self, molded_images, image_metas, verbose=0):
        """Runs the detection pipeline, but expect inputs that are
        molded already. Used mostly for debugging and inspecting
        the model.

        molded_images: List of images loaded using load_image_gt()
        image_metas: image meta data, also returned by load_image_gt()

        Returns a list of dicts, one dict per image. The dict contains:
        rois: [N, (y1, x1, y2, x2)] detection bounding boxes
        class_ids: [N] int class IDs
        scores: [N] float probability scores for the class IDs
        masks: [H, W, N] instance binary masks
        """
        assert self.mode == "inference", "Create model in inference mode."
        assert len(molded_images) == self.config.BATCH_SIZE,\
            "Number of images must be equal to BATCH_SIZE"

        if verbose:
            log("Processing {} images".format(len(molded_images)))
            for image in molded_images:
                log("image", image)

        # Validate image sizes
        # All images in a batch MUST be of the same size
        image_shape = molded_images[0].shape
        for g in molded_images[1:]:
            assert g.shape == image_shape, "Images must have the same size"

        # Anchors
        anchors = self.get_anchors(image_shape)
        # Duplicate across the batch dimension because Keras requires it
        # TODO: can this be optimized to avoid duplicating the anchors?
        anchors = np.broadcast_to(anchors,
                                  (self.config.BATCH_SIZE, ) + anchors.shape)

        if verbose:
            log("molded_images", molded_images)
            log("image_metas", image_metas)
            log("anchors", anchors)
        # Run object detection
        detections, _, _, mrcnn_mask, _, _, _ =\
            self.keras_model.predict([molded_images, image_metas, anchors], verbose=0)
        # Process detections
        results = []
        for i, image in enumerate(molded_images):
            window = [0, 0, image.shape[0], image.shape[1]]
            final_rois, final_class_ids, final_scores, final_masks =\
                self.unmold_detections(detections[i], mrcnn_mask[i],
                                       image.shape, molded_images[i].shape,
                                       window)
            results.append({
                "rois": final_rois,
                "class_ids": final_class_ids,
                "scores": final_scores,
                "masks": final_masks,
            })
        return results

    def get_anchors(self, image_shape):
        """Returns anchor pyramid for the given image size."""
        backbone_shapes = compute_backbone_shapes(self.config, image_shape)
        # Cache anchors and reuse if image shape is the same
        if not hasattr(self, "_anchor_cache"):
            self._anchor_cache = {}
        if not tuple(image_shape) in self._anchor_cache:
            # Generate Anchors
            a = utils.generate_pyramid_anchors(self.config.RPN_ANCHOR_SCALES,
                                               self.config.RPN_ANCHOR_RATIOS,
                                               backbone_shapes,
                                               self.config.BACKBONE_STRIDES,
                                               self.config.RPN_ANCHOR_STRIDE)
            # Keep a copy of the latest anchors in pixel coordinates because
            # it's used in inspect_model notebooks.
            # TODO: Remove this after the notebook are refactored to not use it
            self.anchors = a
            # Normalize coordinates
            self._anchor_cache[tuple(image_shape)] = utils.norm_boxes(
                a, image_shape[:2])
        return self._anchor_cache[tuple(image_shape)]

    def ancestor(self, tensor, name, checked=None):
        """Finds the ancestor of a TF tensor in the computation graph.
        tensor: TensorFlow symbolic tensor.
        name: Name of ancestor tensor to find
        checked: For internal use. A list of tensors that were already
                 searched to avoid loops in traversing the graph.
        """
        checked = checked if checked is not None else []
        # Put a limit on how deep we go to avoid very long loops
        if len(checked) > 500:
            return None
        # Convert name to a regex and allow matching a number prefix
        # because Keras adds them automatically
        if isinstance(name, str):
            name = re.compile(name.replace("/", r"(\_\d+)*/"))

        parents = tensor.op.inputs
        for p in parents:
            if p in checked:
                continue
            if bool(re.fullmatch(name, p.name)):
                return p
            checked.append(p)
            a = self.ancestor(p, name, checked)
            if a is not None:
                return a
        return None

    def find_trainable_layer(self, layer):
        """If a layer is encapsulated by another layer, this function
        digs through the encapsulation and returns the layer that holds
        the weights.
        """
        if layer.__class__.__name__ == 'TimeDistributed':
            return self.find_trainable_layer(layer.layer)
        return layer

    def get_trainable_layers(self):
        """Returns a list of layers that have weights."""
        layers = []
        # Loop through all layers
        for l in self.keras_model.layers:
            # If layer is a wrapper, find inner trainable layer
            l = self.find_trainable_layer(l)
            # Include layer if it has weights
            if l.get_weights():
                layers.append(l)
        return layers

    def run_graph(self, images, outputs, image_metas=None):
        """Runs a sub-set of the computation graph that computes the given
        outputs.

        image_metas: If provided, the images are assumed to be already
            molded (i.e. resized, padded, and normalized)

        outputs: List of tuples (name, tensor) to compute. The tensors are
            symbolic TensorFlow tensors and the names are for easy tracking.

        Returns an ordered dict of results. Keys are the names received in the
        input and values are Numpy arrays.
        """
        model = self.keras_model

        # Organize desired outputs into an ordered dict
        outputs = OrderedDict(outputs)
        for o in outputs.values():
            assert o is not None

        # Build a Keras function to run parts of the computation graph
        inputs = model.inputs
        if model.uses_learning_phase and not isinstance(
                K.learning_phase(), int):
            inputs += [K.learning_phase()]
        kf = K.function(model.inputs, list(outputs.values()))

        # Prepare inputs
        if image_metas is None:
            molded_images, image_metas, _ = self.mold_inputs(images)
        else:
            molded_images = images
        image_shape = molded_images[0].shape
        # Anchors
        anchors = self.get_anchors(image_shape)
        # Duplicate across the batch dimension because Keras requires it
        # TODO: can this be optimized to avoid duplicating the anchors?
        anchors = np.broadcast_to(anchors,
                                  (self.config.BATCH_SIZE, ) + anchors.shape)
        model_in = [molded_images, image_metas, anchors]

        # Run inference
        if model.uses_learning_phase and not isinstance(
                K.learning_phase(), int):
            model_in.append(0.)
        outputs_np = kf(model_in)

        # Pack the generated Numpy arrays into a a dict and log the results.
        outputs_np = OrderedDict([(k, v)
                                  for k, v in zip(outputs.keys(), outputs_np)])
        for k, v in outputs_np.items():
            log(k, v)
        return outputs_np


############################################################
#  Data Formatting
############################################################


def compose_image_meta(image_id, original_image_shape, image_shape, window,
                       scale, active_class_ids):
    """Takes attributes of an image and puts them in one 1D array.

    image_id: An int ID of the image. Useful for debugging.
    original_image_shape: [H, W, C] before resizing or padding.
    image_shape: [H, W, C] after resizing and padding
    window: (y1, x1, y2, x2) in pixels. The area of the image where the real
            image is (excluding the padding)
    scale: The scaling factor applied to the original image (float32)
    active_class_ids: List of class_ids available in the dataset from which
        the image came. Useful if training on images from multiple datasets
        where not all classes are present in all datasets.
    """
    meta = np.array(
        [image_id] +  # size=1
        list(original_image_shape) +  # size=3
        list(image_shape) +  # size=3
        list(window) +  # size=4 (y1, x1, y2, x2) in image cooredinates
        [scale] +  # size=1
        list(active_class_ids)  # size=num_classes
    )
    return meta


def parse_image_meta(meta):
    """Parses an array that contains image attributes to its components.
    See compose_image_meta() for more details.

    meta: [batch, meta length] where meta length depends on NUM_CLASSES

    Returns a dict of the parsed values.
    """
    image_id = meta[:, 0]
    original_image_shape = meta[:, 1:4]
    image_shape = meta[:, 4:7]
    window = meta[:, 7:11]  # (y1, x1, y2, x2) window of image in in pixels
    scale = meta[:, 11]
    active_class_ids = meta[:, 12:]
    return {
        "image_id": image_id.astype(np.int32),
        "original_image_shape": original_image_shape.astype(np.int32),
        "image_shape": image_shape.astype(np.int32),
        "window": window.astype(np.int32),
        "scale": scale.astype(np.float32),
        "active_class_ids": active_class_ids.astype(np.int32),
    }


def parse_image_meta_graph(meta):
    """Parses a tensor that contains image attributes to its components.
    See compose_image_meta() for more details.

    meta: [batch, meta length] where meta length depends on NUM_CLASSES

    Returns a dict of the parsed tensors.
    """
    image_id = meta[:, 0]
    original_image_shape = meta[:, 1:4]
    image_shape = meta[:, 4:7]
    window = meta[:, 7:11]  # (y1, x1, y2, x2) window of image in in pixels
    scale = meta[:, 11]
    active_class_ids = meta[:, 12:]
    return {
        "image_id": image_id,
        "original_image_shape": original_image_shape,
        "image_shape": image_shape,
        "window": window,
        "scale": scale,
        "active_class_ids": active_class_ids,
    }


def mold_image(images, config):
    """Expects an RGB image (or array of images) and subtracts
    the mean pixel and converts it to float. Expects image
    colors in RGB order.
    """
    return images.astype(np.float32) - config.MEAN_PIXEL


def unmold_image(normalized_images, config):
    """Takes a image normalized with mold() and returns the original."""
    return (normalized_images + config.MEAN_PIXEL).astype(np.uint8)


############################################################
#  Miscellenous Graph Functions
############################################################


def trim_zeros_graph(boxes, name='trim_zeros'):
    """Often boxes are represented with matrices of shape [N, 4] and
    are padded with zeros. This removes zero boxes.

    boxes: [N, 4] matrix of boxes.
    non_zeros: [N] a 1D boolean mask identifying the rows to keep
    """
    non_zeros = tf.cast(tf.reduce_sum(tf.abs(boxes), axis=1), tf.bool)
    boxes = tf.boolean_mask(boxes, non_zeros, name=name)
    return boxes, non_zeros


def batch_pack_graph(x, counts, num_rows):
    """Picks different number of values from each row
    in x depending on the values in counts.
    """
    outputs = []
    for i in range(num_rows):
        outputs.append(x[i, :counts[i]])
    return tf.concat(outputs, axis=0)


def norm_boxes_graph(boxes, shape):
    """Converts boxes from pixel coordinates to normalized coordinates.
    boxes: [..., (y1, x1, y2, x2)] in pixel coordinates
    shape: [..., (height, width)] in pixels

    Note: In pixel coordinates (y2, x2) is outside the box. But in normalized
    coordinates it's inside the box.

    Returns:
        [..., (y1, x1, y2, x2)] in normalized coordinates
    """
    h, w = tf.split(tf.cast(shape, tf.float32), 2)
    scale = tf.concat([h, w, h, w], axis=-1) - tf.constant(1.0)
    shift = tf.constant([0., 0., 1., 1.])
    return tf.divide(boxes - shift, scale)


def denorm_boxes_graph(boxes, shape):
    """Converts boxes from normalized coordinates to pixel coordinates.
    boxes: [..., (y1, x1, y2, x2)] in normalized coordinates
    shape: [..., (height, width)] in pixels

    Note: In pixel coordinates (y2, x2) is outside the box. But in normalized
    coordinates it's inside the box.

    Returns:
        [..., (y1, x1, y2, x2)] in pixel coordinates
    """
    h, w = tf.split(tf.cast(shape, tf.float32), 2)
    scale = tf.concat([h, w, h, w], axis=-1) - tf.constant(1.0)
    shift = tf.constant([0., 0., 1., 1.])
    return tf.cast(tf.round(tf.multiply(boxes, scale) + shift), tf.int32)
