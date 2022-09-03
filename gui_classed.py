from tkinter import *
from tkinter import filedialog
from tkinter import messagebox

from PIL import ImageTk, Image
import os
import cv2 as cv

from facet import Facet
from piece import masks_in_scale, image_in_scale, pieces_from_masks
from puzzle import sort_and_filter
from puzzle_piece_detector.inference_callable import Inference

import numpy as np


class Application(Frame):
    def __init__(self, parent):
        Frame.__init__(self, parent)
        self.imageWidth = 800
        self.imageOnCanvas: int = None
        self.button: Button = None
        self.photoImage: PhotoImage = None
        self.weightsFilename: str = None
        self.imageFilename: str = None
        self.viewWindow: Canvas = None
        self.fileMenu: Menu = None
        self.menuBar: Menu = None
        self.pack(fill=BOTH, expand=True)
        self.create_menu()
        self.create_widgets()

    def create_menu(self):
        self.menuBar = Menu(self)
        self.fileMenu = Menu(self.menuBar, tearoff=0)
        self.fileMenu.add_command(label="Image", command=self.pick_image_via_fd)
        self.fileMenu.add_command(label="Weights", command=self.pick_weights_via_fd)
        self.fileMenu.add_separator()
        self.fileMenu.add_command(label="Exit", command=self.quit)
        self.menuBar.add_cascade(label="File", menu=self.fileMenu)
        root.config(menu=self.menuBar)

    def create_widgets(self):
        self.viewWindow = Canvas(self, bg="white")
        self.viewWindow.pack(side=TOP, fill=BOTH, expand=True)
        self.button = Button(self, text="Run segmentation", command=self.run_segmentation_and_update_image)
        self.button.pack()

    def pick_image_via_fd(self):
        filename = filedialog.askopenfile(
            title='Pick Puzzle scan image',
            initialdir='./plots/',
            filetypes=(
                ('Image file', '*.png'),
            ),
        ).name
        self.imageFilename: str = filename
        # self.photoImage: PhotoImage = PhotoImage(file=filename)
        self.update_photo_image(PhotoImage(file=filename))

    def pick_weights_via_fd(self):
        filename = filedialog.askopenfile(
            title='Pick weights',
            initialdir='./weights/',
            filetypes=(
                ('Weights file', '*.h5'),
            ),
        ).name
        self.weightsFilename: str = filename

    def update_photo_image(self, photo_image):
        scale = int(photo_image.width() / self.imageWidth)
        self.photoImage = photo_image.subsample(scale, scale)
        self.imageOnCanvas: int = self.viewWindow.create_image(0, 0, anchor="nw", image=self.photoImage)
        self.viewWindow.itemconfig(self.imageOnCanvas, image=self.photoImage)

    def run_segmentation_and_update_image(self):
        if self.imageFilename is None:
            messagebox.showerror(title="No image picked",
                                 message="Press on the menu top left and pick image with puzzle")
            return
        elif self.weightsFilename is None:
            messagebox.showerror(title="No weights picked",
                                 message="Press on the menu top left and pick weights file")
            return

        weights_path, image_path = self.weightsFilename, self.imageFilename
        inference = Inference(weights_path)
        self.masks = inference.infer_masks_and_watershed(image_path)
        self.update_photo_image(_photo_image(np.invert(np.sum(self.masks, -1, keepdims=True))))
        # rebinding button to
        self.button.config(command=self.run_segmentation_and_update_image,
                           text="Run piece classification and facet segmentation")

    def run_piece_classification_with_facet_segmentation_and_update_image(self):
        print("Piece classification")
        image_path = self.imageFilename
        masks = self.masks
        image = cv.imread(image_path)
        scale = 1
        masks = masks_in_scale(masks, scale)
        image = image_in_scale(image, scale)
        pieces = pieces_from_masks(masks, image)

        n_pieces = len(pieces)
        n_facets = 4
        iou = np.zeros((n_pieces, n_pieces, n_facets, n_facets))
        mgc = np.zeros((n_pieces, n_pieces, n_facets, n_facets))
        cmp = np.zeros((n_pieces, n_pieces, n_facets, n_facets))
        length_for_comparison = 25
        for p1idx, p1 in enumerate(pieces):
            for p2idx, p2 in enumerate(pieces):
                if p1idx < p2idx:
                    for f1idx, f1 in enumerate(p1.facets):
                        if f1.type is Facet.Type.FLAT:
                            continue
                        for f2idx, f2 in enumerate(p2.facets):
                            if f2.type is Facet.Type.FLAT:
                                continue
                            iou[p1idx, p2idx, f1idx, f2idx] = Facet.iou(f1, f2)
                            # mgc[p1idx, p2idx, f1idx, f2idx] = Facet.mgc(f1, f2, length_for_comparison)
                            # cmp[p1idx, p2idx, f1idx, f2idx] = Facet.compatibility(f1, f2, length_for_comparison)

        # sort and filter in descending order

        edges_by_mgc = sort_and_filter(n_pieces, n_facets, 0, mgc, descending=True)
        edges_by_iou = sort_and_filter(n_pieces, n_facets, 0, iou, descending=False)
        edges_by_cmp = sort_and_filter(n_pieces, n_facets, 0, cmp, descending=True)

        # img = np.zeros_like(cropped_image)
        # facet_colors = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 255]]).astype(np.uint8)
        # for fi in range(len(self.facets)):
        #     mask = self.facets[fi].facet_mask
        #     img[mask, :] = facet_colors[fi, :]
        # pass


def _photo_image(image: np.ndarray):
    height, width, _ = image.shape
    data = f'P5 {width} {height} 255 '.encode() + image.astype(np.uint8).tobytes()
    return PhotoImage(width=width, height=height, data=data, format='PPM')


root = Tk()
root.title("Puzzle Solver")

app = Application(root)

root.attributes('-fullscreen', True)
root.bind("<F11>", lambda event: root.attributes("-fullscreen", not root.attributes("-fullscreen")))
root.bind("<Escape>", lambda event: root.attributes("-fullscreen", False))

root.mainloop()
