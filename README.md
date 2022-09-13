# Automated Puzzle Solver - Final Project


## Authors
- Valentin Volovik, 326814449
- Vadim Golotsvan, 314148099

## Running on Colab
The Solver is best run on Colab runtime for the ease of setup and usability.

Follow the link to use the prepared Google Drive:
https://drive.google.com/drive/folders/1K5hPNEX1HWtU5S_6PxxxyKHkR55vWONU?usp=sharing
and run the `Puzzle Project - Final Submission.ipynb`

## How to install without Colab
`pipenv install -r requirements.txt` and edit in `main.py`:
`weights_path, image_path = "./weights/mask_rcnn_puzzle.h5", "./plots/full_downscale.jpg"` to provide the image path and
path to the weights file (included in release)  
