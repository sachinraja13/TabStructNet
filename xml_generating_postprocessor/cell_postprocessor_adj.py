import cv2
from intervaltree.intervaltree import IntervalTree
import numpy as np
import pickle
from PIL import ImageFilter, Image
import os
import os.path
# import xml.etree.cElementTree as ET
import xml.etree.ElementTree as ET
import operator
import pytesseract
import re
from numpy.core.fromnumeric import sort
import xlwt
import statistics

text_read_path = 'result_text/'
row_pkl_read_path = 'result_row_pkl/'
col_pkl_read_path = 'result_col_pkl/'
image_read_path = 'gt_without_box/'
image_write_path = 'processed_jpg/'
image_aligned_write_path = 'processed_aligned_jpg/'
coordinates_write_path = 'processed_txt/'
xml_output_path = 'processed_xml/'
excel_output_path = 'processed_excel/'

PREPARE_CSV = False
ADD_CONNTENT_IN_XML = False
EXECUTE_MERGE = False

try:
    os.mkdir(image_write_path)
    os.mkdir(coordinates_write_path)
    os.mkdir(xml_output_path)
    os.mkdir(excel_output_path)
except:
    print("Directories already exist")


def read_text_file(read_text_path, min_score=0.8):
    table_cells = []
    skipped_indices = []
    #x_mids = []
    #y_mids = []
    cell_id = 0
    i = -1
    with open(read_text_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            i += 1
            line = line.strip()
            data = line.split()
            caption = data[0]
            score = data[1]
            if float(score) < min_score:
                skipped_indices.append(i)
                continue
            x1 = int(data[2])
            y1 = int(data[3])
            x2 = int(data[4])
            y2 = int(data[5])
            x_mid = (x1 + x2) / 2
            y_mid = (y1 + y2) / 2
            table_cells.append((x1, y1, x2, y2))
            #x_mids.append(x1)
            #y_mids.append(y_mid)
            cell_id = cell_id + 1
    return table_cells, skipped_indices


def remove_overlaps(cells,
                    overlapping_area_pct_threshold=0.25,
                    containment_area_pct_threshold=0.8,
                    removed_indices=[]):
    removed_flag = False
    x_interval_tree = IntervalTree()
    y_interval_tree = IntervalTree()
    for i in range(len(cells)):
        bbox = cells[i]
        x1 = bbox[0]
        y1 = bbox[1]
        x2 = bbox[2]
        y2 = bbox[3]
        y_interval_tree[y1:y2] = i
        x_interval_tree[x1:x2] = i
    for i in range(len(cells)):
        cell = cells[i]
        if i in removed_indices:
            continue
        x1, y1, x2, y2 = cell
        y_overlapping_cells = set(
            [j.data for j in y_interval_tree[y1:y2] if j.data != i])
        x_overlapping_cells = set(
            [j.data for j in x_interval_tree[x1:x2] if j.data != i])
        overlapping_cells = x_overlapping_cells & y_overlapping_cells
        overlapping_count = 0
        for overlapping_cell_index in overlapping_cells:
            if overlapping_cell_index in removed_indices:
                continue
            overlapping_cell = cells[overlapping_cell_index]
            ox1, oy1, ox2, oy2 = overlapping_cell
            cell_area = (y2 - y1) * (x2 - x1)
            overlapping_cell_area = (oy2 - oy1) * (ox2 - ox1)
            overlapping_area = max(
                (min(oy2, y2) - max(oy1, y1)) * (min(ox2, x2) - max(ox1, x1)),
                0)
            overlapping_pct = overlapping_area / min(cell_area,
                                                     overlapping_cell_area)
            if overlapping_pct >= overlapping_area_pct_threshold and overlapping_pct <= containment_area_pct_threshold:
                overlapping_count = overlapping_count + 1
            if overlapping_pct >= containment_area_pct_threshold:
                if cell_area < overlapping_cell_area:
                    removed_indices.append(i)
                else:
                    removed_indices.append(overlapping_cell_index)
                removed_flag = True
        if overlapping_count >= 2 and i not in removed_indices:
            removed_indices.append(i)
            removed_flag = True
    return removed_indices, removed_flag


def recursively_remove_overlaps(cells,
                                overlapping_area_pct_threshold=0.25,
                                containment_area_pct_threshold=0.8,
                                removed_indices=[]):
    removed_flag = True
    while removed_flag == True:
        removed_indices, removed_flag = remove_overlaps(
            cells,
            overlapping_area_pct_threshold=overlapping_area_pct_threshold,
            containment_area_pct_threshold=containment_area_pct_threshold,
            removed_indices=removed_indices)
    return removed_indices


def remove_columnwise_unaligned_cells(cells,
                                      containment_region_pct_threshold=0.7,
                                      removed_indices=[]):
    removed_flag = False
    x_interval_tree = IntervalTree()
    for i in range(len(cells)):
        if i in removed_indices:
            continue
        bbox = cells[i]
        x1 = bbox[0]
        x2 = bbox[2]
        x_interval_tree[x1:x2] = i
    for i in range(len(cells)):
        cell = cells[i]
        if i in removed_indices:
            continue
        x1, y1, x2, y2 = cell
        overlapping_cells = set(
            [j.data for j in x_interval_tree[x1:x2] if j.data != i])
        containment_count = 0
        denominator_containment_count = 0
        for overlapping_cell_index in overlapping_cells:
            if overlapping_cell_index in removed_indices:
                continue
            overlapping_cell = cells[overlapping_cell_index]
            ox1, oy1, ox2, oy2 = overlapping_cell
            containment = float(min(x2, ox2) - max(x1, ox1)) / float(
                min(x2 - x1, ox2 - ox1))
            containment = max(0, containment)
            if containment >= containment_region_pct_threshold:
                containment_count = containment_count + 1
            if containment >= 0.2:
                denominator_containment_count = denominator_containment_count + 1
        if denominator_containment_count >= 2 and containment_count < int(
                0.34 *
            (denominator_containment_count + 1)) and i not in removed_indices:
            removed_indices.append(i)
            removed_flag = True
    return removed_indices, removed_flag


def remove_cells_min_height_criteria(cells,
                                     threshold_pct=0.5,
                                     remove_indices=[]):
    heights = []
    for cell in cells:
        x1, y1, x2, y2 = cell
        heights.append(y2 - y1)
    height_threshold = int(
        max(statistics.mean(heights), statistics.median(heights)) *
        threshold_pct)
    for i in range(len(cells)):
        x1, y1, x2, y2 = cells[i]
        if y2 - y1 < height_threshold:
            remove_indices.append(i)
    return remove_indices


def recursively_remove_columnwise_unaligned_cells(
        cells, containment_region_pct_threshold=0.7, removed_indices=[]):
    removed_flag = True
    while removed_flag == True:
        removed_indices, removed_flag = remove_columnwise_unaligned_cells(
            cells,
            containment_region_pct_threshold=containment_region_pct_threshold,
            removed_indices=removed_indices)
    return removed_indices


def remove_rowwise_unaligned_cells(cells,
                                   containment_region_pct_threshold=0.7,
                                   removed_indices=[]):
    removed_flag = False
    y_interval_tree = IntervalTree()
    for i in range(len(cells)):
        if i in removed_indices:
            continue
        bbox = cells[i]
        y1 = bbox[1]
        y2 = bbox[3]
        y_interval_tree[y1:y2] = i
    for i in range(len(cells)):
        cell = cells[i]
        if i in removed_indices:
            continue
        x1, y1, x2, y2 = cell
        overlapping_cells = set(
            [j.data for j in y_interval_tree[y1:y2] if j.data != i])
        containment_count = 0
        denominator_containment_count = 0
        for overlapping_cell_index in overlapping_cells:
            if overlapping_cell_index in removed_indices:
                continue
            overlapping_cell = cells[overlapping_cell_index]
            ox1, oy1, ox2, oy2 = overlapping_cell
            containment = float(min(y2, oy2) - max(y1, oy1)) / float(
                min(y2 - y1, oy2 - oy1))
            containment = max(0, containment)
            if containment >= containment_region_pct_threshold:
                containment_count = containment_count + 1
            if containment >= 0.2:
                denominator_containment_count = denominator_containment_count + 1
        if denominator_containment_count >= 2 and containment_count < int(
                0.34 *
            (denominator_containment_count + 1)) and i not in removed_indices:
            removed_indices.append(i)
            removed_flag = True
    return removed_indices, removed_flag


def recursively_remove_rowwise_unaligned_cells(
        cells, containment_region_pct_threshold=0.7, removed_indices=[]):
    removed_flag = True
    while removed_flag == True:
        removed_indices, removed_flag = remove_rowwise_unaligned_cells(
            cells,
            containment_region_pct_threshold=containment_region_pct_threshold,
            removed_indices=removed_indices)
    return removed_indices


def remove_extra_indices(indices_list):
    def create_indices_dict(indices_list):
        indices_dict = {}
        for assignment in indices_list:
            for index in assignment:
                if index not in indices_dict:
                    indices_dict[index] = []
                other_assignment_set = set(
                    [x for x in assignment if x != index])
                indices_dict[index].append(other_assignment_set)
        return indices_dict

    def remove_extra(indices_list, remove_index):
        indices = []
        for assignment in indices_list:
            new_assignment = []
            for index in assignment:
                if index == remove_index:
                    continue
                elif index > remove_index:
                    new_assignment.append(index - 1)
                else:
                    new_assignment.append(index)
            indices.append(new_assignment)
        return indices

    # print(indices_list)
    while True > 0:
        indices_dict = create_indices_dict(indices_list)
        remove_indices = []
        for i in indices_dict:
            redundant_indices = list(set.intersection(*indices_dict[i]))
            # print(i, indices_dict[i], redundant_indices)
            if len(redundant_indices) > 0:
                remove_indices.append(i)
        remove_indices = list(set(remove_indices))
        remove_indices = sorted(remove_indices)
        # print(indices_list, remove_indices)
        if len(remove_indices) > 0:
            indices_list = remove_extra(indices_list, remove_indices[0])
        else:
            break
    return indices_list


def get_column_structure_indices(adj, coordinates):
    def get_x_overlap_and_containment(cell1, cell2):
        overlap = float(min(cell1[2], cell2[2]) -
                        max(cell1[0], cell2[0])) / float(
                            max(cell1[2] - cell1[0], cell2[2] - cell2[0]))
        containment = float(min(cell1[2], cell2[2]) -
                            max(cell1[0], cell2[0])) / float(
                                min(cell1[2] - cell1[0], cell2[2] - cell2[0]))
        return overlap, containment

    coordinates = np.asarray(coordinates)
    sorted_indices_end_x = np.argsort(coordinates.view('i8,i8,i8,i8'),
                                      order=['f2', 'f0', 'f1'],
                                      axis=0)[:, 0]

    column_indexes = []
    for i in range(len(coordinates)):
        column_indexes.append(set())
    cur_col_index = -1

    x_interval_tree = IntervalTree()
    for index in sorted_indices_end_x:
        x1, y1, x2, y2 = coordinates[index]
        x_interval_tree[x1:x2] = index
    for i in sorted_indices_end_x:
        #include cell itself in it's overlaps
        x1, y1, x2, y2 = coordinates[i]
        x_overlapping_cells = set([j.data for j in x_interval_tree[x1:x2]])
        condition_meeting_overlapping_cells = []
        for j in x_overlapping_cells:
            overlap_i_j, containment_i_j = get_x_overlap_and_containment(
                coordinates[i], coordinates[j])
            adj_i_j = max(adj[i, j], adj[j, i])
            if adj_i_j == 1 and (overlap_i_j >= 0.5 or containment_i_j >= 0.7):
                condition_meeting_overlapping_cells.append(j)

            elif adj_i_j == 0 and (overlap_i_j >= 0.7
                                   or containment_i_j >= 0.8):
                condition_meeting_overlapping_cells.append(j)
        column_indexes_np = np.array(column_indexes)

        if len(column_indexes[i]) >= 1:
            continue
        num_common_assigned_indices = len(
            set.intersection(
                *column_indexes_np[condition_meeting_overlapping_cells]))
        # print(*column_indexes_np[condition_meeting_overlapping_cells])
        # print(i, condition_meeting_overlapping_cells,
        #       len(condition_meeting_overlapping_cells),
        #       num_common_assigned_indices)
        if num_common_assigned_indices > 0:
            continue
        cur_col_index += 1
        # print(cur_col_index)
        for j in condition_meeting_overlapping_cells:
            column_indexes[j].add(cur_col_index)
    column_indexes = remove_extra_indices(column_indexes)
    start_column_indices = []
    end_column_indices = []
    skipped = []
    for i in range(len(column_indexes)):
        if len(column_indexes[i]) == 0:
            start_column_indices.append(-1)
            end_column_indices.append(-1)
            skipped.append(i)
            continue
        start_column_indices.append(min(column_indexes[i]))
        end_column_indices.append(max(column_indexes[i]))
    # print(column_indexes)
    print("Num columns identified: " +
          str(max(max(x) for x in column_indexes if len(x) > 0) + 1))
    print("Num skipped: " + str(len(skipped)))
    return start_column_indices, end_column_indices


def get_row_structure_indices(adj, coordinates):
    def get_y_overlap_and_containment(cell1, cell2):
        overlap = float(min(cell1[3], cell2[3]) -
                        max(cell1[1], cell2[1])) / float(
                            max(cell1[3] - cell1[1], cell2[3] - cell2[1]))
        containment = float(min(cell1[3], cell2[3]) -
                            max(cell1[1], cell2[1])) / float(
                                min(cell1[3] - cell1[1], cell2[3] - cell2[1]))
        return overlap, containment

    coordinates = np.asarray(coordinates)
    sorted_indices_end_y = np.argsort(coordinates.view('i8,i8,i8,i8'),
                                      order=['f3', 'f1', 'f0'],
                                      axis=0)[:, 0]

    row_indexes = []
    for i in range(len(coordinates)):
        row_indexes.append(set())
    cur_row_index = -1

    y_interval_tree = IntervalTree()
    for index in sorted_indices_end_y:
        x1, y1, x2, y2 = coordinates[index]
        y_interval_tree[y1:y2] = index

    for i in sorted_indices_end_y:
        #include cell itself in it's overlaps
        x1, y1, x2, y2 = coordinates[i]
        y_overlapping_cells = set([j.data for j in y_interval_tree[y1:y2]])
        condition_meeting_overlapping_cells = []
        for j in y_overlapping_cells:
            overlap_i_j, containment_i_j = get_y_overlap_and_containment(
                coordinates[i], coordinates[j])
            adj_i_j = max(adj[i, j], adj[j, i])
            if adj_i_j == 1 and (overlap_i_j >= 0.5
                                 or containment_i_j >= 0.75):
                condition_meeting_overlapping_cells.append(j)

            elif adj_i_j == 0 and (overlap_i_j >= 0.7
                                   or containment_i_j >= 0.85):
                condition_meeting_overlapping_cells.append(j)
        row_indexes_np = np.array(row_indexes)

        if len(row_indexes[i]) >= 1:
            continue
        num_common_assigned_indices = len(
            set.intersection(
                *row_indexes_np[condition_meeting_overlapping_cells]))
        # print(*row_indexes_np[condition_meeting_overlapping_cells])
        # print(i, condition_meeting_overlapping_cells,
        #       len(condition_meeting_overlapping_cells),
        #       num_common_assigned_indices)
        if num_common_assigned_indices > 0:
            continue
        cur_row_index += 1
        # print(cur_row_index)
        for j in condition_meeting_overlapping_cells:
            row_indexes[j].add(cur_row_index)

    start_row_indices = []
    end_row_indices = []
    skipped = []
    row_indexes = remove_extra_indices(row_indexes)
    # print(row_indexes)
    for i in range(len(row_indexes)):
        if len(row_indexes[i]) == 0:
            start_row_indices.append(-1)
            end_row_indices.append(-1)
            skipped.append(i)
            continue
        start_row_indices.append(min(row_indexes[i]))
        end_row_indices.append(max(row_indexes[i]))
    print("Num rows identified: " +
          str(max(max(x) for x in row_indexes if len(x) > 0) + 1))
    print("Num skipped: " + str(len(skipped)))
    return start_row_indices, end_row_indices


def get_aligned_column_coordinates(cells_coordinates, start_cols, end_cols):
    column_starts = {}
    column_ends = {}
    for i in range(len(start_cols)):
        col_index = start_cols[i]
        cell = cells_coordinates[i]
        if col_index not in column_starts:
            column_starts[col_index] = []
        column_starts[col_index].append(cell[0])
    for i in range(len(end_cols)):
        col_index = end_cols[i]
        cell = cells_coordinates[i]
        if col_index not in column_ends:
            column_ends[col_index] = []
        column_ends[col_index].append(cell[2])
    min_col_starts = {}
    max_col_ends = {}
    for col_index in column_starts:
        min_col_starts[col_index] = int(
            statistics.median(column_starts[col_index]))
    for col_index in column_ends:
        max_col_ends[col_index] = int(statistics.median(
            column_ends[col_index]))
    col_starts = {}
    col_ends = {}
    for i in sorted(list(min_col_starts.keys())):
        if i == 0:
            col_starts[i] = 0
            col_ends[max(max_col_ends)] = max_col_ends[max(max_col_ends)]
            continue
        gap = min_col_starts[i] - max_col_ends[i - 1]
        col_starts[i] = min_col_starts[i] - int(float(gap) / 2.0)
        col_ends[i - 1] = max_col_ends[i - 1] + int(float(gap) / 2.0)
    return col_starts, col_ends


def get_aligned_row_coordinates(cells_coordinates, start_rows, end_rows):
    row_starts = {}
    row_ends = {}
    for i in range(len(start_rows)):
        row_index = start_rows[i]
        cell = cells_coordinates[i]
        if row_index not in row_starts:
            row_starts[row_index] = []
        row_starts[row_index].append(cell[1])
    for i in range(len(end_rows)):
        row_index = end_rows[i]
        cell = cells_coordinates[i]
        if row_index not in row_ends:
            row_ends[row_index] = []
        row_ends[row_index].append(cell[3])
    min_row_starts = {}
    max_row_ends = {}
    for row_index in row_starts:
        min_row_starts[row_index] = min(row_starts[row_index])
    for row_index in row_ends:
        max_row_ends[row_index] = max(row_ends[row_index])
    row_starts = {}
    row_ends = {}
    for i in sorted(list(min_row_starts.keys())):
        if i == 0:
            row_starts[i] = 0
            row_ends[max(max_row_ends)] = max_row_ends[max(max_row_ends)]
            continue
        gap = min_row_starts[i] - max_row_ends[i - 1]
        row_starts[i] = min_row_starts[i] - int(float(gap) / 2.0)
        row_ends[i - 1] = max_row_ends[i - 1] + int(float(gap) / 2.0)
    return row_starts, row_ends


def get_final_table_details(table_details, row_starts, col_starts, row_ends,
                            col_ends):
    final_table_details = []
    located_regions = {}
    for i in range(len(table_details)):
        _, start_row, start_col, end_row, end_col, x1, y1, x2, y2 = table_details[
            i]
        new_cell_coords = [
            int(col_starts[start_col]),
            int(row_starts[start_row]),
            int(col_ends[end_col]),
            int(row_ends[end_row])
        ]
        num_regions_in_cell = 0
        for row in range(start_row, end_row + 1):
            for col in range(start_col, end_col + 1):
                num_regions_in_cell += 1
                if (col, row, col, row) not in located_regions:
                    located_regions[(col, row, col, row)] = []
                located_regions[(col, row, col, row)].append(
                    (i, num_regions_in_cell))

    regions_processed = {}
    for row in row_starts:
        for col in col_starts:
            region = (col, row, col, row)
            if region in regions_processed:
                continue
            if region in located_regions:
                best_cell = min(located_regions[region], key=lambda x: x[1])
                i = best_cell[0]
                _, start_row, start_col, end_row, end_col, x1, y1, x2, y2 = table_details[
                    i]
                for row_id in range(start_row, end_row + 1):
                    for col_id in range(start_col, end_col + 1):
                        regions_processed[(col_id, row_id, col_id,
                                           row_id)] = True
            else:
                start_row = row
                start_col = col
                end_row = row
                end_col = col
                regions_processed[region] = True
            new_cell_coords = [
                int(col_starts[start_col]),
                int(row_starts[start_row]),
                int(col_ends[end_col]),
                int(row_ends[end_row])
            ]
            final_table_details.append([
                False, start_row, start_col, end_row, end_col,
                new_cell_coords[0], new_cell_coords[1], new_cell_coords[2],
                new_cell_coords[3]
            ])
    return final_table_details


def get_final_table_details_without_merge(row_starts, col_starts, row_ends,
                                          col_ends):
    final_table_details = []

    regions_processed = {}
    for row in row_starts:
        for col in col_starts:
            region = (col, row, col, row)
            if region in regions_processed:
                continue
            start_row = row
            start_col = col
            end_row = row
            end_col = col
            regions_processed[region] = True
            new_cell_coords = [
                int(col_starts[start_col]),
                int(row_starts[start_row]),
                int(col_ends[end_col]),
                int(row_ends[end_row])
            ]
            final_table_details.append([
                False, start_row, start_col, end_row, end_col,
                new_cell_coords[0], new_cell_coords[1], new_cell_coords[2],
                new_cell_coords[3]
            ])
    return final_table_details


def add_cells_to_img(img, final_cells, skipped_indices):
    for i in range(len(final_cells)):
        if i not in skipped_indices:
            x1 = final_cells[i][5]
            y1 = final_cells[i][6]
            x2 = final_cells[i][7]
            y2 = final_cells[i][8]
            cv2.rectangle(img, (x1 - 1, y1 - 1), (x2 - 1, y2 - 1), (255, 0, 0),
                          3)
    return img


def add_aligned_cells_to_img(img, col_starts, row_starts, col_ends, row_ends):
    for i in col_starts:
        for j in row_starts:
            try:
                x1 = int(col_starts[i])
                y1 = int(row_starts[j])
                x2 = int(col_ends[i])
                y2 = int(row_ends[j])
                cv2.rectangle(img, (x1 - 1, y1 - 1), (x2 - 1, y2 - 1),
                              (255, 0, 0), 3)
            except:
                continue
    return img


def create_root(image_path, file_prefix, width, height, depth):
    root = ET.Element("prediction")
    ET.SubElement(root, "folder").text = "images"
    ET.SubElement(root, "filename").text = "{}".format(file_prefix)
    ET.SubElement(root, "path").text = image_path + "{}".format(file_prefix)
    source = ET.SubElement(root, "source")
    ET.SubElement(source, "database").text = "Unknown"
    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(size, "depth").text = str(depth)
    ET.SubElement(root, "segmentated").text = "0"
    return root


def create_cell_annotation(root,
                           table_details,
                           table_information,
                           img,
                           add_content=False):
    obj = ET.SubElement(root, "object")
    ET.SubElement(obj, "name").text = "table"
    ET.SubElement(obj, "pose").text = "Unspecified"
    ET.SubElement(obj, "truncated").text = str(0)
    ET.SubElement(obj, "difficult").text = str(0)
    bbox = ET.SubElement(obj, "bndbox")
    ET.SubElement(bbox, "xmin").text = str(table_information[0])

    ET.SubElement(bbox, "ymin").text = str(table_information[1])
    ET.SubElement(bbox, "xmax").text = str(table_information[2])
    ET.SubElement(bbox, "ymax").text = str(table_information[3])
    cells = ET.SubElement(obj, "cells")
    for j in range(len(table_details)):
        cell_detail = table_details[j]
        cell = ET.SubElement(cells, "tablecell")
        ET.SubElement(cell, "dont_care").text = str(cell_detail[0])
        ET.SubElement(cell, "end_col").text = str(cell_detail[4])
        ET.SubElement(cell, "end_row").text = str(cell_detail[3])
        ET.SubElement(cell, "start_col").text = str(cell_detail[2])
        ET.SubElement(cell, "start_row").text = str(cell_detail[1])
        if add_content:
            cell_img = Image.fromarray(
                img[max(cell_detail[6] - 10, 0):min(cell_detail[8] +
                                                    10, img.shape[0]),
                    max(cell_detail[5] - 10, 0):min(cell_detail[7] +
                                                    10, img.shape[1]), :])
            cell_content = str(
                pytesseract.image_to_string(cell_img, lang='eng')).strip()
            processed_cell_content = re.sub(r'[^a-zA-Z0-9 ]', r'',
                                            cell_content)
            ET.SubElement(
                cell, "processed_content").text = str(processed_cell_content)
            ET.SubElement(cell, "content").text = str(cell_content)
        ET.SubElement(cell, "x0").text = str(cell_detail[5])
        ET.SubElement(cell, "x1").text = str(cell_detail[7])
        ET.SubElement(cell, "y0").text = str(cell_detail[6])
        ET.SubElement(cell, "y1").text = str(cell_detail[8])
        ET.SubElement(cell, "index").text = str(j)
    return root


def create_output_excel(excel_write_path,
                        table_details,
                        table_information,
                        img,
                        add_content=False):
    wb = xlwt.Workbook()
    sheet = wb.add_sheet("digitized_table")
    rows_processed = {}
    cols_processed = {}
    for j in range(len(table_details)):
        cell_detail = table_details[j]
        end_col = cell_detail[4]
        end_row = cell_detail[3]
        start_col = cell_detail[2]
        start_row = cell_detail[1]
        already_processed = False
        # for i in (start_col, end_col + 1):
        #     if i in cols_processed:
        #         already_processed = True
        # for i in (start_row, end_row + 1):
        #     if i in rows_processed:
        #         already_processed = True
        # if already_processed:
        #     continue
        if add_content:
            cell_img = Image.fromarray(
                img[max(cell_detail[6] - 10, 0):min(cell_detail[8] +
                                                    10, img.shape[0]),
                    max(cell_detail[5] - 10, 0):min(cell_detail[7] +
                                                    10, img.shape[1]), :])
            cell_content = str(
                pytesseract.image_to_string(cell_img, lang='eng')).strip()
            processed_cell_content = re.sub(r'[^a-zA-Z0-9 ]', r'',
                                            cell_content)
            try:
                sheet.write(start_row, start_col, str(cell_content))
            except:
                pass
            # if start_row != end_row or start_col != end_col:
            #     sheet.merge(start_row, end_row, start_col, end_col)
            # for i in (start_col, end_col + 1):
            #     cols_processed[i] = True
            # for i in (start_row, end_row + 1):
            #     rows_processed[i] = True
    wb.save(excel_write_path)


######################################################
def main():
    index = 0
    for text_file in os.listdir(text_read_path):
        if text_file.endswith('txt'):
            removed_indices = []
            file_prefix = text_file.replace(".txt", "")
            try:
                if os.path.exists(xml_output_path + file_prefix +
                                  ".xml") and os.path.exists(image_write_path +
                                                             file_prefix +
                                                             ".jpg"):
                    print("File " + str(file_prefix) +
                          " already exists. Proceeding to next.")
                else:
                    print("File " + str(file_prefix) +
                          " does not exist. Processing.")
            except:
                pass
            # if file_prefix != '0709.2961v1.1':
            #     continue
            img_read_path = image_read_path + file_prefix + ".jpg"
            img_write_path = image_write_path + file_prefix + ".jpg"
            # aligned_img_write_path = image_aligned_write_path + file_prefix + ".jpg"
            excel_write_path = excel_output_path + file_prefix + ".xls"
            read_text_path = text_read_path + text_file
            table_cells, skipped_indices = read_text_file(read_text_path)
            removed_indices = remove_cells_min_height_criteria(
                table_cells, threshold_pct=0.5, remove_indices=removed_indices)
            removed_indices = recursively_remove_overlaps(
                table_cells,
                overlapping_area_pct_threshold=0.2,
                containment_area_pct_threshold=0.75,
                removed_indices=removed_indices)
            print("Cells removed due to overlap : " + str(removed_indices))
            removed_indices = recursively_remove_columnwise_unaligned_cells(
                table_cells,
                containment_region_pct_threshold=0.7,
                removed_indices=removed_indices)
            print("Cells removed due to column-wise misalignment : " +
                  str(removed_indices))
            removed_indices = recursively_remove_rowwise_unaligned_cells(
                table_cells,
                containment_region_pct_threshold=0.7,
                removed_indices=removed_indices)
            print("After removing cells due to row-wise misalignment : " +
                  str(removed_indices))
            row_adj = pickle.load(
                open(row_pkl_read_path + file_prefix + ".pkl", 'rb'))
            col_adj = pickle.load(
                open(col_pkl_read_path + file_prefix + ".pkl", 'rb'))
            row_adj = np.delete(row_adj, skipped_indices, axis=0)
            row_adj = np.delete(row_adj, skipped_indices, axis=1)
            col_adj = np.delete(col_adj, skipped_indices, axis=0)
            col_adj = np.delete(col_adj, skipped_indices, axis=1)
            table_cells = np.delete(table_cells, removed_indices, axis=0)
            row_adj = np.delete(row_adj, removed_indices, axis=0)
            row_adj = np.delete(row_adj, removed_indices, axis=1)
            col_adj = np.delete(col_adj, removed_indices, axis=0)
            col_adj = np.delete(col_adj, removed_indices, axis=1)
            x_starts = np.asarray(table_cells)[:, 0]
            x_ends = np.asarray(table_cells)[:, 2]
            print("*******")

            print(str(index + 1) + ":\t" + file_prefix)
            img = cv2.imread(img_read_path)
            # aligned_img = cv2.imread(img_read_path)
            height, width, channel = img.shape
            table_information = [0, 0, width, height]
            table_details = []
            root = create_root(img_read_path, file_prefix, width, height,
                               channel)

            start_col_assignments, end_col_assignments = get_column_structure_indices(
                col_adj, table_cells)
            start_row_assignments, end_row_assignments = get_row_structure_indices(
                row_adj, table_cells)
            skipped_indices = []
            for i in range(len(table_cells)):
                if start_row_assignments[i] == -1 or start_col_assignments[
                        i] == -1:
                    skipped_indices.append(i)
                if start_row_assignments[i] > end_row_assignments[i]:
                    end_row_assignments[i] = start_row_assignments[i]
                if start_col_assignments[i] > end_col_assignments[i]:
                    end_col_assignments[i] = start_col_assignments[i]
                if start_row_assignments[i] == 0:
                    table_cells[i][1] = 0
                if end_row_assignments[i] == max(end_row_assignments):
                    table_cells[i][3] = height
                if start_col_assignments[i] == 0:
                    table_cells[i][0] = 0
                if end_col_assignments[i] == max(end_col_assignments):
                    table_cells[i][2] = width

                table_details.append([
                    False, start_row_assignments[i], start_col_assignments[i],
                    end_row_assignments[i], end_col_assignments[i],
                    table_cells[i][0], table_cells[i][1], table_cells[i][2],
                    table_cells[i][3]
                ])
                # print(
                #     str(table_cells[i]) + "\t" +
                #     str(start_row_assignments[i]) + "\t" +
                #     str(end_row_assignments[i]) + "\t" +
                #     str(start_col_assignments[i]) + "\t" +
                #     str(end_col_assignments[i]))

            col_starts, col_ends = get_aligned_column_coordinates(
                table_cells, start_col_assignments, end_col_assignments)
            row_starts, row_ends = get_aligned_row_coordinates(
                table_cells, start_row_assignments, end_row_assignments)
            if EXECUTE_MERGE:
                table_details = get_final_table_details(
                    table_details, row_starts, col_starts, row_ends, col_ends)
            else:
                table_details = get_final_table_details_without_merge(
                    row_starts, col_starts, row_ends, col_ends)
            # aligned_img = add_aligned_cells_to_img(aligned_img, col_starts,
            #                                        row_starts, col_ends,
            #                                        row_ends)

            root = create_cell_annotation(root,
                                          table_details,
                                          table_information,
                                          img,
                                          add_content=ADD_CONNTENT_IN_XML)
            if PREPARE_CSV:
                create_output_excel(excel_write_path,
                                    table_details,
                                    table_information,
                                    img,
                                    add_content=True)
            tree = ET.ElementTree(root)
            xml_file_path = xml_output_path + file_prefix + ".xml"
            tree.write(xml_file_path)
            img = add_cells_to_img(img, table_details, skipped_indices)
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cv2.imwrite(img_write_path, img)
            # cv2.imwrite(aligned_img_write_path, aligned_img)
            index += 1
            print("Processed Files : " + str(index))


if __name__ == "__main__":
    main()
