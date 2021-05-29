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

text_read_path = 'result_text/'
row_pkl_read_path = 'result_row_pkl/'
col_pkl_read_path = 'result_col_pkl/'
image_read_path = 'gt_without_box/'
image_write_path = 'processed_jpg/'
coordinates_write_path = 'processed_txt/'
xml_output_path = 'processed_xml/'

try:
    os.mkdir(image_write_path)
    os.mkdir(coordinates_write_path)
    os.mkdir(xml_output_path)
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
                    overlapping_area_pct_threshold=0.2,
                    removed_indices=[]):
    import intervaltree
    removed_flag = False
    x_interval_tree = intervaltree.IntervalTree()
    y_interval_tree = intervaltree.IntervalTree()
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
        overlapping_cells = x_overlapping_cells | y_overlapping_cells
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
            if overlapping_pct >= overlapping_area_pct_threshold:
                overlapping_count = overlapping_count + 1
        if overlapping_count > 2 and i not in removed_indices:
            removed_indices.append(i)
            removed_flag = True
    return removed_indices, removed_flag


def recursively_remove_overlaps(cells, overlapping_area_pct_threshold=0.2):
    removed_indices = []
    removed_flag = True
    while removed_flag == True:
        removed_indices, removed_flag = remove_overlaps(
            cells,
            overlapping_area_pct_threshold=overlapping_area_pct_threshold,
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
    sorted_indices_start_x = np.argsort(coordinates.view('i8,i8,i8,i8'),
                                        order=['f0', 'f2', 'f1'],
                                        axis=0)[:, 0]
    column_indexes = []
    for i in range(len(coordinates)):
        column_indexes.append([])
    cur_col_index = -1
    prev_index = sorted_indices_end_x[0]

    for i in sorted_indices_end_x:
        overlap_i_prev, containment_i_prev = get_x_overlap_and_containment(
            coordinates[i], coordinates[prev_index])
        adj_i_prev = max(adj[i, prev_index], adj[prev_index, i])
        if (adj_i_prev == 0 and
            (overlap_i_prev < 0.75 or containment_i_prev < 0.8)) or (
                adj_i_prev == 1 and
                (overlap_i_prev < 0.6
                 or containment_i_prev < 0.7)) or cur_col_index == -1:
            cur_col_index += 1
            for j in sorted_indices_start_x:
                overlap_i_j, containment_i_j = get_x_overlap_and_containment(
                    coordinates[i], coordinates[j])
                adj_i_j = max(adj[i, j], adj[j, i])
                if adj_i_j == 1 and (overlap_i_j >= 0.2
                                     or containment_i_j >= 0.6):
                    column_indexes[j].append(cur_col_index)
                elif adj_i_j == 0 and (overlap_i_j >= 0.25
                                       or containment_i_j >= 0.8):
                    column_indexes[j].append(cur_col_index)
        prev_index = i
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
    sorted_indices_start_y = np.argsort(coordinates.view('i8,i8,i8,i8'),
                                        order=['f1', 'f3', 'f0'],
                                        axis=0)[:, 0]
    row_indexes = []
    for i in range(len(coordinates)):
        row_indexes.append([])
    cur_row_index = -1
    prev_index = sorted_indices_end_y[0]
    for i in sorted_indices_end_y:
        overlap_i_prev, containment_i_prev = get_y_overlap_and_containment(
            coordinates[i], coordinates[prev_index])
        adj_i_prev = max(adj[i, prev_index], adj[prev_index, i])
        if (adj_i_prev == 0 and
            (overlap_i_prev < 0.75 or containment_i_prev < 0.8)) or (
                adj_i_prev == 1 and
                (overlap_i_prev < 0.6
                 or containment_i_prev < 0.7)) or cur_row_index == -1:
            cur_row_index += 1
            # print(i, prev_index, adj_i_prev, overlap_i_prev, containment_i_prev, cur_row_index)
            for j in sorted_indices_start_y:
                overlap_i_j, containment_i_j = get_y_overlap_and_containment(
                    coordinates[i], coordinates[j])
                adj_i_j = max(adj[i, j], adj[j, i])
                if adj_i_j == 1 and (overlap_i_j >= 0.1
                                     or containment_i_j >= 0.3):
                    row_indexes[j].append(cur_row_index)
                elif adj_i_j == 0 and (overlap_i_j >= 0.15
                                       or containment_i_j >= 0.5):
                    row_indexes[j].append(cur_row_index)
        prev_index = i
    start_row_indices = []
    end_row_indices = []
    skipped = []
    row_indexes = remove_extra_indices(row_indexes)
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


def add_cells_to_img(img, final_cells, skipped_indices):
    for i in range(len(final_cells)):
        if i not in skipped_indices:
            x1, y1, x2, y2 = final_cells[i]
            cv2.rectangle(img, (x1 - 1, y1 - 1), (x2 - 1, y2 - 1), (255, 0, 0),
                          3)
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
                           add_content=True):
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
                img[cell_detail[6]:cell_detail[8] + 1,
                    cell_detail[5]:cell_detail[7] + 1, :])
            cell_content = str(
                pytesseract.image_to_string(cell_img, lang='eng')).strip()
            cell_content = re.sub(r'[^a-zA-Z0-9 ]', r'', cell_content)
            ET.SubElement(cell, "content").text = str(cell_content)
        ET.SubElement(cell, "x0").text = str(cell_detail[5])
        ET.SubElement(cell, "x1").text = str(cell_detail[7])
        ET.SubElement(cell, "y0").text = str(cell_detail[6])
        ET.SubElement(cell, "y1").text = str(cell_detail[8])
    return root


######################################################
def main():
    for text_file in os.listdir(text_read_path):
        if text_file.endswith('txt'):
            file_prefix = text_file.replace(".txt", "")
            # if file_prefix != 'failure_1_1806.04265v1.5':
            #     continue
            img_read_path = image_read_path + file_prefix + ".png"
            img_write_path = image_write_path + file_prefix + ".jpg"
            read_text_path = text_read_path + text_file
            table_cells, skipped_indices = read_text_file(read_text_path)
            removed_indices = recursively_remove_overlaps(
                table_cells, overlapping_area_pct_threshold=0.2)
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

            print(read_text_path)
            img = cv2.imread(img_read_path)
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
            root = create_cell_annotation(root, table_details,
                                          table_information, img)
            tree = ET.ElementTree(root)
            xml_file_path = xml_output_path + file_prefix + ".xml"
            tree.write(xml_file_path)
            img = add_cells_to_img(img, table_cells, skipped_indices)
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cv2.imwrite(img_write_path, img)


if __name__ == "__main__":
    main()
