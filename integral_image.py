import numpy as np 

def get_intgral_image(input_image_array):
    # initialize the array with all zeros containing extra row and column
    width, height = input_image_array.shape

    integral_image = np.zeros((width+1, height+1))

    for i in range(0, width+1):
        for j in range(0, height+1):
            integral_image[i+1][j+1] = input_image_array[i][j] + integral_image[i+1][j] + integral_image[i][j+1] - integral_image[i][j]

    return integral_image

def sum_of_region(integral_image, top_left_point, bottom_right_point):
    """
    Calculates the sum in the rectangle with top-left point and bottom-right co-ordinates are given
    :param integral_image: Integral image of input image
    :type integral_image: numpy.ndarray
    :param top_left_point: (x, y) tuple containing coordinates of top-left point of the rectangle
    :type top_left_point: (int, int)
    :param bottom_right_point: (x, y) tuple containing coordinates of bottom-right point of the rectangle
    :type bottom_right_point: (int, int)
    :return The Sum of all the pixel intensities of pixels inside the rectangle with given top-left and bottom-right corners
    :rtype int
    """

    if top_left_point == bottom_right_point:
        return integral_image[(top_left_point[1], top_left_point[0])]

    top_right_point = (bottom_right_point[1], top_left_point[0])
    bottom_left_point = (top_left_point[1], bottom_right_point[0])
    return (integral_image[(bottom_right_point[1], bottom_right_point[0])] + integral_image[(top_left_point[1], top_left_point[0])] - integral_image[top_right_point] - integral_image[bottom_left_point])   