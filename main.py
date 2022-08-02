try:
    from numpy import dtype
    from skimage.feature import peak_local_max
    import os
    import json
    import glob
    import argparse
    import cv2
    import numpy as np
    from scipy import signal as sg
    from scipy.ndimage import convolve
    import scipy.ndimage as filters

    from scipy import misc
    from PIL import Image

    import matplotlib.pyplot as plt
except ImportError:
    print("Need to fix the installation")
    raise


def rgb_convolve(image, kernel):
    red = convolve(image[:, :, 0], kernel)
    green = convolve(image[:, :, 1], kernel)
    # blue = convolve2d(image[:, :, 2], [[0, 0, 0], [0, 0, 0], [0, 0, 0]], 'same')
    # return np.stack([red, green, blue], axis=2)
    return red, green


def find_tfl_lights(c_image: np.ndarray, **kwargs):
    """
    Detect candidates for TFL lights. Use c_image, kwargs and you imagination to implement
    :param c_image: The image itself as np.uint8, shape of (H, W, 3)
    :param kwargs: Whatever config you want to pass in here
    :return: 4-tuple of x_red, y_red, x_green, y_green
    """

    kernel = (1 / 9) * np.array([[-1, -1, -1, -1, -1],
                                 [-1, -1, 4, -1, -1],
                                 [-1, 4, 4, 4, -1],
                                 [-1, -1, 4, -1, -1],
                                 [-1, -1, -1, -1, -1]])



    # Getting the kernel to be used in Top-Hat
    # filterSize = (3, 3)
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
    #                                  filterSize)
    # kernel = np.array([[1, 1, 2, 1, 1],
    #                    [1, -2, -2, -2, 1],
    #                    [2, -2, -4, -2, 2],
    #                    [1, -2, -2, -2, 1],
    #                    [1, 1, 2, 1, 1]])
    # Do not delete its important!
    # conv_im1 = rgb_convolve2d(c_image, kernel6)
    # fig, ax = plt.subplots(1, 2)
    # plt.imshow(kernel5, cmap='gray')

    # Convert Image to Red.
    # for i in range(len(c_image)):
    #     for j in range(len(c_image[i])):
    #         c_image[i][j][0] = 255

    # img = Image.fromarray(c_image)
    # img.resize(size=(int(img.size[0] * 0.9), int(img.size[1] * 0.9)))
    # gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    # c_image = plt.imread(c_image)

    conv_im1, conv_im2 = rgb_convolve(c_image, kernel)

    # conv_im2 = black_tophat(conv_im1, size=3)
    # red_image = filters.maximum_filter(conv_im1, size=3)
    # green_image = filters.maximum_filter(conv_im2, size=3)
    # red_image = peak_local_max(conv_im1, min_distance=20)
    # green_image = peak_local_max(conv_im2, min_distance=20)
    # plt.imshow(kernel, cmap="gray" )
    # plt.show()

    conv_im1, conv_im2 = rgb_convolve(c_image, kernel)
    # plt.imshow(conv_im1)
    # plt.title("red")
    # plt.show()
    # conv_im2 = black_tophat(conv_im1, size=3)
    red_image = filters.maximum_filter(conv_im1, size=1)
    green_image = filters.maximum_filter(conv_im2, size=1)


    red_coordinates = np.argwhere(red_image > 0.35)
    green_coordinates = np.argwhere(green_image > 0.4)

    return red_coordinates, green_coordinates


### GIVEN CODE TO TEST YOUR IMPLENTATION AND PLOT THE PICTURES
def show_image_and_gt(image, objs, fig_num=None):
    plt.figure(fig_num).clf()
    plt.imshow(image)
    labels = set()
    if objs is not None:
        for o in objs:
            poly = np.array(o['polygon'])[list(np.arange(len(o['polygon']))) + [0]]
            plt.plot(poly[:, 0], poly[:, 1], 'r', label=o['label'])
            labels.add(o['label'])
        if len(labels) > 1:
            plt.legend()


def test_find_tfl_lights(image_path, json_path=None, fig_num=None):
    """
    Run the attention code
    """
    # image = np.array(Image.open(image_path))
    image = plt.imread(image_path)
    if json_path is None:
        objects = None
    else:
        gt_data = json.load(open(json_path))
        what = ['traffic light']
        objects = [o for o in gt_data['objects'] if o['label'] in what]

    # show_image_and_gt(image, objects, fig_num)

    plt.figure(56)
    plt.clf()
    h = plt.subplot(111)
    plt.imshow(image)
    plt.figure(57)
    plt.clf()
    plt.subplot(111, sharex=h, sharey=h)
    plt.imshow(image)

    # red_x, red_y, green_x, green_y = find_tfl_lights(image)
    red_list, green_list = find_tfl_lights(image)
    for i in red_list:
        plt.plot(i[1], i[0], 'ro', color='r', markersize=2)
    for i in green_list:
        plt.plot(i[1], i[0], 'ro', color='g', markersize=2)
    # plt.plot(green_x, green_y, 'ro', color='g', markersize=4)


def main(argv=None):
    """It's nice to have a standalone tester for the algorithm.
    Consider looping over some images from here, so you can manually exmine the results
    Keep this functionality even after you have all system running, because you sometime want to debug/improve a module
    :param argv: In case you want to programmatically run this"""

    parser = argparse.ArgumentParser("Test TFL attention mechanism")
    parser.add_argument('-i', '--image', type=str, help='Path to an image')
    parser.add_argument("-j", "--json", type=str, help="Path to json GT for comparison")
    parser.add_argument('-d', '--dir', type=str, help='Directory to scan images in')
    args = parser.parse_args(argv)
    # To do: change the directory according to your computer!!!
    default_base = r"tests"

    if args.dir is None:
        args.dir = default_base
    flist = glob.glob(os.path.join(args.dir, '*_leftImg8bit.png'))

    print(flist)

    for image in flist:
        json_fn = image.replace('_leftImg8bit.png', '_gtFine_polygons.json')

        if not os.path.exists(json_fn):
            json_fn = None
        test_find_tfl_lights(image, json_fn)

    if len(flist):
        print("You should now see some images, with the ground truth marked on them. Close all to quit.")
    else:
        print("Bad configuration?? Didn't find any picture to show")
    plt.show(block=True)


if __name__ == '__main__':
    main()
