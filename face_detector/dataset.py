"""
Creates a Pytorch dataset to load the Pascal VOC dataset
"""

import torch
import os
from PIL import Image


class VOCDataset(torch.utils.data.Dataset):
    def __init__(
        self, txt_file, S=7, B=2, C=20, transform=None,
    ):
        self.annotations = []
        with open(txt_file) as f:
            for line in f.readlines():
                self.annotations += [line]
        self.transform = transform
        self.S = S
        self.B = B
        self.C = C

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        label_path = os.path.join('data/labels/', self.annotations[index][12:-5] + '.txt')
        boxes = []
        with open(label_path) as f:
            for label in f.readlines():
                class_label, x, y, width, height = [
                    float(x) if float(x) != int(float(x)) else int(x)
                    for x in label.replace("\n", "").split()
                ]
                boxes.append([class_label, x, y, width, height])
        
        img_path = self.annotations[index]
        image = Image.open(img_path[:-1])
        boxes = torch.tensor(boxes)

        if self.transform:
            image, boxes = self.transform(image, boxes)
        label_matrix = torch.zeros((self.S, self.S, self.C + 5*self.B))
        for box in boxes:
            class_label, x, y, width, height = box.tolist()
            class_label = int(class_label)

            # i,j represents the cell row and cell column
            i, j = int(self.S * y), int(self.S * x)
            x_cell, y_cell = self.S * x - j, self.S * y - i

            """
            Calculating the width and height of cell of bounding box,
            relative to the cell is done by the following, with
            width as the example:
            
            width_pixels = (width*self.image_width)
            cell_pixels = (self.image_width)
            
            Then to find the width relative to the cell is simply:
            width_pixels/cell_pixels, simplification leads to the
            formulas below.
            """
            width_cell, height_cell = (
                width * self.S,
                height * self.S,
            )

            # If no object already found for specific cell i,j
            # Note: This means we restrict to ONE object
            # per cell!
            if label_matrix[i, j, 20] == 0:
                # Set that there exists an object
                label_matrix[i, j, 20] = 1

                # Box coordinates
                box_coordinates = torch.tensor(
                    [x_cell, y_cell, width_cell, height_cell]
                )

                label_matrix[i, j, 21:25] = box_coordinates

                # Set one hot encoding for class_label
                label_matrix[i, j, class_label] = 1
        return image, label_matrix
