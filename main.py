def get_image_cords(img_path, screen_img_path):
    """
    Function returns the coordinates of image on screen
    Parameters:
        img_path: path of image to be searched on screen
        screen_img_path: path of full_screen image
    Returns the coordinates of search img on full screen
    """
    import numpy, cv2, collections
    Box = collections.namedtuple('Box', 'left top width height')
    full_img = cv2.imread(screen_img_path, cv2.IMREAD_GRAYSCALE)
    search_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    confidence = 0.9
    limit = 1
    step = 1
    region = (0, 0)
    needleHeight, needleWidth = search_img.shape[:2]
    result = cv2.matchTemplate(full_img, search_img, cv2.TM_CCOEFF_NORMED)
    maxresult = result.max()
    print("Match %:",maxresult*100)
    match_indices = numpy.arange(result.size)[(result >= result.max()).flatten()]
    matches = numpy.unravel_index(match_indices[:limit], result.shape)
    matchx = matches[1] * step + region[0]  # vectorized
    matchy = matches[0] * step + region[1]
    for x, y in zip(matchx, matchy):
        # print(x, y, needleWidth, needleHeight)
        return Box(x, y, needleWidth, needleHeight)
