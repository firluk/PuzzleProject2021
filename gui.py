import numpy as np
import cv2 as cv
import matplotlib
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog as fd
from tkinter import *
from PIL import Image, ImageTk

matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg
)

# TODO
# - Read file and show image
# - Show segmentation
# - Show detection of facets
# - Show classification of facets
# - Show puzzle assembly

root: tk = tk.Tk()
root.title("Puzzle Solver")


# Model object that will hold the images and such
class Model:
    def __init__(self):
        self.image = None


def pick_file_via_fd():
    filename = fd.askopenfile(
        title='Pick Puzzle scan image',
        initialdir='.',
        filetypes=(
            ('Image file', '*.png'),
        ),
    ).name

    model.filename = filename
    model.photo_image = PhotoImage(file=filename)
    image_canvas.create_image(0, 0, anchor="CENTER", image=model.photo_image)


model = Model()


menubar = Menu(root)
filemenu = Menu(menubar, tearoff=0)
filemenu.add_command(label="Pick Puzzle scan image", command=pick_file_via_fd)
filemenu.add_separator()
filemenu.add_command(label="Close", command=root.destroy)
menubar.add_cascade(label="File", menu=filemenu)

intro_frame = Frame(root)
intro_label = Label(intro_frame)

image_frame = Frame(root, bg="green")
# image_canvas = Canvas(image_frame, width=300, height=400, bg="yellow")

image_canvas = Canvas(image_frame, width=1920, height=1356, bg="yellow")
image_canvas.pack()
# image_canvas.create_image(5, 5, anchor="nw", image=img)
image_frame.pack()
# image_canvas.pack_forget()

# segmentation_frame = Frame(root)
# detection_frame = Frame(root)
# classification_frame = Frame(root)
# assembly_frame = Frame(root)

root.attributes('-fullscreen', True)
root.bind("<F11>", lambda event: root.attributes("-fullscreen", not root.attributes("-fullscreen")))
root.bind("<Escape>", lambda event: root.attributes("-fullscreen", False))
root.config(menu=menubar)
root.mainloop()
