import torch
import pandas as pd

def runs_to_pixel_classes(runs, image_size):
    pixel_classes = torch.zeros([image_size*image_size], dtype=torch.float)

    for run in runs:
        pixel_classes[int(run[0]) : int(run[0]+run[1])] = 1.0

    return pixel_classes.view(-1, image_size)

def bounding_box_to_coordinate_runs(x, y, width, height, image_size):
    runs = []
    row_start = (y-1) * image_size
    for i in range(0, height):
        runs.append((row_start + x - 1, width))
        row_start += image_size

    return runs

def iou_score_from_runs(pred_coord_runs, target_coord_runs):
    # There was no ship expected and no ship found, return a perfect score
    if not pred_coord_runs and not target_coord_runs:
        return 1.0

    # There WAS a ship expected but no ship was found   OR  There WASN'T a ship expected but one was found
    if not pred_coord_runs or not target_coord_runs:
        return 0.0

    area_of_intersect, area_of_union = 0.0, 0.0
    pred_idx, target_idx = 0, 0
    pos = min(pred_coord_runs[pred_idx][0], target_coord_runs[target_idx][0])

    # Count the overlap
    while pred_idx < len(pred_coord_runs) and target_idx < len(target_coord_runs):
        if pred_coord_runs[pred_idx][0] < target_coord_runs[target_idx][0]:
            end = pred_coord_runs[pred_idx][0] + pred_coord_runs[pred_idx][1]

            # Count any of the runs from the other set that are completely within this run
            while target_idx < len(target_coord_runs) \
                    and end > target_coord_runs[target_idx][0] + target_coord_runs[target_idx][1]:
                area_of_intersect += target_coord_runs[target_idx][1]
                target_idx += 1

            # The beginning of a run from the other set overlaps with the end of this one
            if target_idx < len(target_coord_runs) and end > target_coord_runs[target_idx][0]:
                area_of_intersect += end - target_coord_runs[target_idx][0]

            pred_idx += 1
        else:
            end = target_coord_runs[target_idx][0] + target_coord_runs[target_idx][1]

            # Count any of the runs from the other set that are completely within this run
            while pred_idx < len(pred_coord_runs) \
                    and end > pred_coord_runs[pred_idx][0] + pred_coord_runs[pred_idx][1]:
                area_of_intersect += pred_coord_runs[pred_idx][1]
                pred_idx += 1

            # The beginning of a run from the other set overlaps with the end of this one
            if pred_idx < len(pred_coord_runs) and end > pred_coord_runs[pred_idx][0]:
                area_of_intersect += end - pred_coord_runs[pred_idx][0]

            target_idx += 1

    # Count the size of each individually, subtract the size of the overlap to get the union 
    total = sum(n for _, n in pred_coord_runs)
    total += sum(n for _, n in target_coord_runs)
    area_of_union = total - area_of_intersect

    return area_of_intersect / area_of_union
