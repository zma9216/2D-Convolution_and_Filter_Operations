from skimage import io, img_as_float, img_as_ubyte
import skimage.filters as fl
import skimage.feature as ft
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as grid
import os

os.chdir(os.path.dirname(__file__))


def read_image(filename, gray):
    """

    Simple function for reading input images.

    :param str filename: Image file name
    :param bool gray: Converts image to grayscale if True
    :return: Outputs the image array as either grayscale or RGB
    :rtype: ndarray
    """
    if gray is True:
        img = io.imread(filename, as_gray=True)
        return img
    else:
        img = io.imread(filename)
        return img


def convolve_2d(image, kernel):
    """

    User-defined function for performing convolution on an image array with given kernel/filter in 2D.

    :param ndarray image: Image array to be convoluted
    :param ndarray kernel: Input matrix (3x3 in this case)
    :return: Outputs the convolved image array
    :rtype: ndarray
    """
    # Flip the kernel vertically and horizontally
    kernel = np.flipud(np.fliplr(kernel))
    # Get dimensions of kernel
    k_row, k_col = kernel.shape
    # Compute h and w of kernel
    h, w = ((k_row - 1) // 2), ((k_col - 1) // 2)
    # Perform zero padding on the image with width of 1
    padded_img = np.pad(image, pad_width=1, mode="constant", constant_values=0.0)

    # Create output array of zeros with same shape and type of padded image
    output_img = np.zeros_like(padded_img)
    # Get dimensions of the output image
    out_row, out_col = output_img.shape
    # Implementation is based on algorithm presented in lecture
    # i and j represent the position of the filter sliding throughout the whole padded image
    # starting on the top left corner of the input image
    for i in range(h, out_row - h):
        for j in range(w, out_col - w):
            # m and n represent the position of pixels in the given image patch and all elements of the filter
            for m in range(-h, h + 1):
                for n in range(-w, w + 1):
                    # Sum of pointwise multiplication of each element of the filter and image patch
                    # is stored to the output array
                    output_img[i, j] += (padded_img[i + m, j + n] * kernel[m + h, n + w])

    # Strip padding from output image
    output_img = np.delete((np.delete(output_img, [0, -1], axis=0)), [0, -1], axis=1)
    return output_img


def part1():
    # Read input image as 64-bit float ndarray
    moon_png = "moon.png"
    img = img_as_float(read_image(moon_png, gray=True))
    # Read input image as uint8 ndarray for later use
    img_uint8 = read_image(moon_png, gray=True)

    # Create 3x3 64-bit float matrices for use as kernels
    # Laplacian filter
    laplacian_k = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]]).astype(np.float64)
    # "Copy All" filter
    copy_all_k = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]).astype(np.float64)
    # "Shift" filter
    shifted_k = np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]]).astype(np.float64)
    # Mean/averaging filter
    mean_k = (np.ones((3, 3)) / 9).astype(np.float64)

    # Output image arrays after 2D convolution is performed
    img_lap = convolve_2d(img, laplacian_k)
    img_copy_all = convolve_2d(img, copy_all_k)
    img_shifted = convolve_2d(img, shifted_k)
    img_mean = convolve_2d(img, mean_k)
    # Create copy of 64-bit float image array
    temp = np.copy(img)
    # Subtract mean filtered image from original image, then converted difference to uint8
    diff = img_as_ubyte(temp - img_mean)
    # Add difference to original uint8 image array
    img_sharpen = img_uint8 + diff

    # Create 2x3 grid of subplots with gridspec to specify grid geometry
    gs = grid.GridSpec(2, 3)
    # Figure size set to 15 x 10 inches - 1500 x 1000 pixels
    fig = plt.figure(figsize=(15, 10))
    # Add original image to left of 2 x 2 grid of other images and labelled as such
    ax1 = fig.add_subplot(gs[0:, 0])
    # Images to be displayed are converted back to uint8 ndarrays
    ax1.imshow(img_as_ubyte(img), cmap="gray")
    ax1.set_title("Original")

    # Add Laplacian filtered image in first subplot of 2 x 2 grid
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(img_as_ubyte(np.abs(img_lap)), cmap="gray")
    ax2.set_title("Laplacian Filter")

    # Add copied image in second subplot of 2 x 2 grid
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(img_as_ubyte(img_copy_all), cmap="gray")
    ax3.set_title("\"Copy All\" Filter")

    # Add shifted image in third subplot of 2 x 2 grid
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.imshow(img_as_ubyte(img_shifted), cmap="gray")
    ax4.set_title("\"Shifted\" Filter")

    # Add sharpened image in fourth subplot of 2 x 2 grid
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.imshow(img_sharpen, cmap="gray")
    ax5.set_title("\"Sharpen\" Filter")

    # Automatically adjust subplot parameters and display all subplots
    plt.tight_layout()
    plt.show()


def part2():
    """
    The median filter was more successful in removing the salt and pepper noise
    """
    # Read input image as 64-bit float ndarray
    noisy_jpg = "noisy.jpg"
    img = img_as_float(read_image(noisy_jpg, gray=True))

    # Perform median filtering on image
    median = fl.median(img)
    # Perform Gaussian filtering on image
    gaussian = fl.gaussian(img)

    # Create figure with 1 x 2 grid of subplots and set figure size to 10 x 10 inches - 1000 x 1000 pixels
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 10))
    # Images to be displayed are converted back to uint8 ndarrays
    axes[0].imshow(img_as_ubyte(median), cmap="gray")
    axes[0].set_title("Median Filter")

    axes[1].imshow(img_as_ubyte(gaussian), cmap="gray")
    axes[1].set_title("Gaussian Filter")

    # Display all filtered images
    plt.tight_layout()
    plt.show()


def part3():
    # Read input image as 64-bit float ndarrays
    cameraman_png = "damage_cameraman.png"
    mask_png = "damage_mask.png"
    damaged_img = img_as_float(read_image(cameraman_png, gray=True))
    mask = img_as_float(read_image(mask_png, gray=True))

    # Create copy of original image
    repaired_img = np.copy(damaged_img)
    # Get length of row and col from mask image
    row, col = mask.shape
    # Set stopping criterion for algorithm
    repairing = True
    # Create counter for number of iterations performed by the algorithm
    counter = 0
    while repairing:
        # Inpainting implementation based on algorithm presented in lecture
        # Create temporary ndarray of image for checking convergence
        temp = repaired_img
        # Perform Gaussian filtering on image
        repaired_img = fl.gaussian(repaired_img)
        # Iterate
        for i in range(row):
            for j in range(col):
                if mask[i, j] == 1:
                    repaired_img[i, j] = damaged_img[i, j]
        # Increment counter by 1 when one iteration is complete
        counter += 1
        # Check for convergence of output image array and temporary array
        if np.allclose(temp, repaired_img) is True:
            # End algorithm when stopping criterion is met
            repairing = False

    # Create figure with 1 x 2 grid of subplots and set figure size to 10 x 10 inches - 1000 x 1000 pixels
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 10))
    # Images to be displayed are converted back to uint8 ndarrays
    axes[0].imshow(img_as_ubyte(damaged_img), cmap="gray")
    axes[0].set_title("Damaged Cameraman")

    axes[1].imshow(img_as_ubyte(repaired_img), cmap="gray")
    # Display number of iterations performed by the algorithm
    axes[1].set_title("Repaired Cameraman \nIterations: %i" % counter)

    # Display damaged image and repaired image
    plt.tight_layout()
    plt.show()


def part4():
    # Read input image as 64-bit float ndarray
    ex2_jpg = "ex2.jpg"
    img = img_as_float(read_image(ex2_jpg, gray=True))

    # Compute vertical derivative of image
    sobel_v = fl.sobel_v(img)
    # Compute horizontal derivative of image
    sobel_h = fl.sobel_h(img)
    # Compute gradient magnitude of image by square rooting the sum of vertical derivative squared and horizontal
    # derivative squared. Solution obtained from link provided by the assignment description -
    # https://homepages.inf.ed.ac.uk/rbf/HIPR2/sobel.htm
    sobel = np.hypot(sobel_h, sobel_v)

    # Create figure with 2 x 2 grid of subplots and set figure size to 20 x 12 inches - 2000 x 1200 pixels
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 12))
    # Images to be displayed are converted back to uint8 ndarrays
    axes[0, 0].imshow(img_as_ubyte(img), cmap="gray")
    axes[0, 0].set_title("Original")

    axes[0, 1].imshow(img_as_ubyte(sobel_v), cmap="gray")
    axes[0, 1].set_title("Sobel_Vertical")

    axes[1, 0].imshow(img_as_ubyte(sobel_h), cmap="gray")
    axes[1, 0].set_title("Sobel_Horizontal")

    axes[1, 1].imshow(img_as_ubyte(sobel), cmap="gray")
    axes[1, 1].set_title("Gradient Magnitude")

    # Display the images
    plt.tight_layout()
    plt.show()


def part5():
    """
    Decreasing low threshold values present more weak edges and more strong edges for increasing high threshold values.
    As the sigma value increases, finer features are detected less since smoothing/blurring increases,
    but there will be less noise.
    """
    # Read input image (dtype of ndarray is uint8)
    ex2_jpg = "ex2.jpg"
    img = read_image(ex2_jpg, gray=True)

    # Perform Gaussian filtering on 64-bit float image
    gaussian = fl.gaussian(img_as_float(img))

    # Fix sigma value to 1 and perform Canny edge detection with specified threshold values
    lt_25 = ft.canny(img, sigma=1.0, low_threshold=25.0, high_threshold=None)
    lt_50 = ft.canny(img, sigma=1.0, low_threshold=50.0, high_threshold=None)
    ht_150 = ft.canny(img, sigma=1.0, low_threshold=None, high_threshold=150.0)
    ht_200 = ft.canny(img, sigma=1.0, low_threshold=None, high_threshold=200.0)

    # Fix low threshold value to 50 and high threshold to 150
    # and perform Canny edge detection with specified sigma values
    sig_1_0 = ft.canny(img, sigma=1.0, low_threshold=50.0, high_threshold=150.0)
    sig_1_5 = ft.canny(img, sigma=1.5, low_threshold=50.0, high_threshold=150.0)
    sig_2_0 = ft.canny(img, sigma=2.0, low_threshold=50.0, high_threshold=150.0)
    sig_2_5 = ft.canny(img, sigma=2.5, low_threshold=50.0, high_threshold=150.0)

    # Create 3x4 grid of subplots with gridspec to specify grid geometry
    gs = grid.GridSpec(3, 4)
    fig = plt.figure(figsize=(25, 13))
    ax1 = fig.add_subplot(gs[0, 1])
    # Display original image
    ax1.imshow(img, cmap="gray")
    ax1.set_title("Original")

    ax2 = fig.add_subplot(gs[0, 2])
    # Display Gaussian filtered image as uint8
    ax2.imshow(img_as_ubyte(gaussian), cmap="gray")
    ax2.set_title("Gaussian Filter")

    # All Canny filtered images are outputted as binary and left as such
    ax3 = fig.add_subplot(gs[1, 0])
    # Display image affected by low threshold = 25
    ax3.imshow(lt_25, cmap="gray")
    ax3.set_title("Low Threshold: 25.0")

    ax4 = fig.add_subplot(gs[1, 1])
    # Display image affected by low threshold = 50
    ax4.imshow(lt_50, cmap="gray")
    ax4.set_title("Low Threshold: 50.0")

    ax5 = fig.add_subplot(gs[1, 2])
    # Display image affected by high threshold = 150
    ax5.imshow(ht_150, cmap="gray")
    ax5.set_title("High Threshold: 150.0")

    ax6 = fig.add_subplot(gs[1, 3])
    # Display image affected by high threshold = 150
    ax6.imshow(ht_200, cmap="gray")
    ax6.set_title("High Threshold: 200.0")

    ax7 = fig.add_subplot(gs[2, 0])
    # Display image affected by sigma = 1
    ax7.imshow(sig_1_0, cmap="gray")
    ax7.set_title("Sigma: 1.0")

    ax8 = fig.add_subplot(gs[2, 1])
    # Display image affected by sigma = 1.5
    ax8.imshow(sig_1_5, cmap="gray")
    ax8.set_title("Sigma: 1.5")

    ax9 = fig.add_subplot(gs[2, 2])
    # Display image affected by sigma = 2.0
    ax9.imshow(sig_2_0, cmap="gray")
    ax9.set_title("Sigma: 2.0")

    ax10 = fig.add_subplot(gs[2, 3])
    # Display image affected by sigma = 2.5
    ax10.imshow(sig_2_5, cmap="gray")
    ax10.set_title("Sigma: 2.5")

    # Display all subplots
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    part1()
    part2()
    part3()
    part4()
    part5()
