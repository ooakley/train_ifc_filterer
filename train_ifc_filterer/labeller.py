"""
This script provides a GUI for labeling a dataset of cell images into "good" (i.e. contains
a single cell at the right size) and "bad" (multiple cells, or too small) images using a Tinder-like
interface.
"""
import os
from tkinter import Label, Tk
import numpy as np
from PIL import Image, ImageTk


class Labeller:
    """Labeller GUI class."""

    def __init__(self, root, data, labels, index):
        """
        Args:
            root -- the Tkinter root window
            data -- the images that still remain to be labelled
            labels -- an array that contains label information
            index -- an index into the array indicating our current position in the dataset
        """

        self._index = index
        self._data = data
        self._labels = labels
        # create the image GUI
        image = Image.fromarray(self._data[index])
        self._photo = ImageTk.PhotoImage(image)
        self._label = Label(root, image=self._photo)
        self._label.bind("<Right>", self.correct)
        self._label.bind("<Left>", self.incorrect)
        self._label.bind("<Up>", self.unsure)
        self._label.config(height=200, width=200)
        self._label.focus_set()
        self._label.pack()

    def correct(self, _):
        """Call method when the image has been labelled as correct."""
        label = 1
        self._labels[self._index] = label
        self._next_image()

    def incorrect(self, _):
        """Call method when the image has been labelled as incorrect."""
        label = 0
        self._labels[self._index] = label
        self._next_image()

    def unsure(self, _):
        """Call method when the image has not been labelled due to uncertainty."""
        label = 2
        self._labels[self._index] = label
        self._next_image()

    def _next_image(self):
        """Move to the next image."""
        self._index += 1
        image = Image.fromarray(self._data[self._index])
        self._photo = ImageTk.PhotoImage(image)
        self._label.config(image=self._photo)


def transform_images(data):
    """PIL can't handle uint16 images so rescale pixels and transform to uint8"""
    data = data.astype('float32')
    val_min, val_max = data.min(), data.max()
    data = (data - val_min)/(val_max - val_min)
    data = (data*255).astype('uint8')
    return data


def main():
    """Main function."""

    # Load the np array that contains the images.
    # if we have started labelling this numpy array, it will load the label array and
    # a temp array with the cell pictures (where entries get deleted after the are
    # labelled, so we can quit anytime and start where we left it)
    # otherwise it will load the original cell array and create a label array
    data = np.load("raw_dataset.npy")
    data = data[:, :, :, 0]

    if os.path.exists("labels.npy"):
        labels = np.load("labels.npy")
    else:
        # Use 10 as a default value if the image hasn't been labelled
        labels = np.ones(data.shape[0])*10

    # Get the pictures to a readable format for the GUI:
    data = transform_images(data)
    index = np.where(labels == 10)[0][0]
    print(np.count_nonzero(labels != 10), " images labelled!")

    # Create the GUI:
    root = Tk()  # this is our blank window
    Labeller(root, data, labels, index)
    root.mainloop()  # infinite loop that runs the GUI until we stop

    # Save the labels:
    np.save("labels.npy", labels)


if __name__ == "__main__":
    main()
