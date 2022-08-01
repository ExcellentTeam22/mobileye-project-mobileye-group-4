try:
    import os
    import json
    import glob
    import argparse
    import cv2
    import numpy as np
    from scipy import signal as sg
    from scipy.ndimage import maximum_filter
    from scipy.signal import convolve2d
    from scipy import misc
    from PIL import Image

    import matplotlib.pyplot as plt
except ImportError:
    print("Need to fix the installation")
    raise


def rgb_convolve2d(image, kernel):
    red = convolve2d(image[:,:,0], kernel, 'valid', boundary='wrap')
    green = convolve2d(image[:,:,1], kernel, 'valid')
    blue = convolve2d(image[:,:,2], kernel, 'valid')
    return np.stack([red, green, blue], axis=2)

def find_tfl_lights(c_image: np.ndarray, **kwargs):
    """
    Detect candidates for TFL lights. Use c_image, kwargs and you imagination to implement
    :param c_image: The image itself as np.uint8, shape of (H, W, 3)
    :param kwargs: Whatever config you want to pass in here
    :return: 4-tuple of x_red, y_red, x_green, y_green
    """
    # Some kernels for test
    kernel4 = np.array([[1, 0, -1],
                        [0, 0, 0],
                        [-1, 0, 1]])
    kernel1 = np.array([[0, -1, 0],
                        [-1, 5, -1],
                        [0, -1, 0]])
    kernel2 = np.array([[-1, -1, -1],
                        [-1, 8, -1],
                        [-1, -1, -1]])
    kernel5 = np.array([[1, 0, -1],
                        [2, 0, -2],
                        [1, 0, -1]])
    kernel5 = np.array([[-1, 2, -1],
                        [-2, 4, -2],
                        [-1, 2, -1]])
    kernel6 = np.array([[-1, -1, -1, -1, -1],
                        [-1, 2, 2, 2, -1],
                        [-1, 2, 10, 2, -1],
                        [-1, 2, 2, 2, -1],
                        [-1, -1, -1, -1, -1]])
    # Do not delete its important!
    conv_im1 = rgb_convolve2d(c_image, kernel6)
    #fig, ax = plt.subplots(1, 2)
    #plt.imshow(kernel5, cmap='gray')

    # Convert Image to Red.
    for i in range(len(c_image)):
        for j in range(len(c_image[i])):
            c_image[i][j][0] = 255
    gray = cv2.cvtColor(c_image, cv2.COLOR_RGB2GRAY)
    conv_im1 = convolve2d(gray, kernel6)

    conv_im1 = maximum_filter(conv_im1, size=3)
    plt.imshow(conv_im1, cmap="gray")
    # plt.imshow(c_image)

    return [500, 510, 520], [500, 500, 500], [700, 710], [500, 500]


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
    image = np.array(Image.open(image_path))
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

    red_x, red_y, green_x, green_y = find_tfl_lights(image)
    plt.plot(red_x, red_y, 'ro', color='r', markersize=4)
    plt.plot(green_x, green_y, 'ro', color='g', markersize=4)



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