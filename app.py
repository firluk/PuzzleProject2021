from tkinter import *
from tkinter import filedialog
from tkinter import messagebox

import cv2 as cv
import numpy as np
import skimage
from PIL import ImageTk, Image

from piece import pieces_from_masks
from utils import image_in_scale, masks_in_scale
from puzzle import paint_facets_by_type, paint_facets_distinct
from puzzle_piece_detector.inference_callable import Inference


class Application(Frame):
    def __init__(self, parent):
        Frame.__init__(self, parent)
        self.weightsFilename: str = None
        self.imageFilename: str = None
        self.masks = None
        self.pieces = None

        self.imageWidth: int = 800
        self.imageOnCanvas: int = None
        self.button: Button = None
        self.photoImage: PhotoImage = None
        self.viewWindow: Canvas = None
        self.fileMenu: Menu = None
        self.menuBar: Menu = None
        self.pack(fill=BOTH, expand=True)
        self.create_menu()
        self.create_widgets()

    def create_menu(self):
        self.menuBar = Menu(self)
        self.fileMenu = Menu(self.menuBar, tearoff=0)
        self.fileMenu.add_command(label="Image and Weights", command=self.pick_args)
        self.fileMenu.add_separator()
        self.fileMenu.add_command(label="Exit", command=self.quit)
        self.menuBar.add_cascade(label="File", menu=self.fileMenu)
        root.config(menu=self.menuBar)

    def create_widgets(self):
        self.button = Button(self, anchor="nw", text="Run segmentation", command=self.run_segmentation_and_update_image)
        self.button.pack(side=TOP, fill=BOTH)
        self.viewWindow = Canvas(self, bg="white")
        self.viewWindow.pack(side=TOP, fill=BOTH, expand=True)

    def pick_args(self):
        self.pick_image_via_fd()
        self.pick_weights_via_fd()

    def pick_image_via_fd(self):
        filename = filedialog.askopenfile(
            title='Pick Puzzle scan image',
            initialdir='./plots/',
            filetypes=(
                ('Image file', '*.png'),
            ),
        ).name
        self.imageFilename: str = filename
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

    def update_photo_image(self, photo_image: PhotoImage):
        if self.photoImage is None or photo_image.width() >= self.photoImage.width():
            scale = int(photo_image.height() / int(self.winfo_height() * (2 / 3)))
            self.photoImage = photo_image.subsample(scale, scale)
        else:
            scale = int(int(self.winfo_height() * (2 / 3)) / photo_image.height())
            self.photoImage = photo_image.zoom(scale, scale)
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
        self.masks = inference.infer_masks_and_blur(image_path)
        self.update_photo_image(_photo_image(np.invert(np.sum(self.masks, -1, keepdims=True))))
        # rebinding button to
        self.button.config(command=self.run_piece_classification_with_facet_segmentation_and_update_image,
                           text="Run piece classification and facet segmentation")

    def run_piece_classification_with_facet_segmentation_and_update_image(self):
        print("Piece classification")
        image_path = self.imageFilename
        masks = self.masks
        image = skimage.io.imread(image_path)
        scale = 1
        masks = masks_in_scale(masks, scale)
        image = image_in_scale(image, scale)
        pieces = pieces_from_masks(masks, image)
        self.pieces = pieces

        masks_with_facets = paint_facets_distinct(masks, pieces)

        self.update_photoImage_using_np_array(masks_with_facets)
        # rebinding button to
        self.button.config(command=self.show_facets_by_classification,
                           text="Show facets by classification")

    def show_facets_by_classification(self):
        pieces = self.pieces
        masks = self.masks
        masks_with_facets = paint_facets_by_type(masks, pieces)
        self.update_photoImage_using_np_array(masks_with_facets)
        # TODO add next screens from here

    def update_photoImage_using_np_array(self, masks_with_facets):
        masks_with_facets = cv.resize(masks_with_facets, (self.photoImage.width(), self.photoImage.height()))
        self.photoImage = ImageTk.PhotoImage(Image.fromarray(masks_with_facets))
        self.imageOnCanvas: int = self.viewWindow.create_image(0, 0, anchor="nw", image=self.photoImage)
        self.viewWindow.itemconfig(self.imageOnCanvas, image=self.photoImage)


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
root.iconbitmap('favicon.ico')
root.mainloop()
