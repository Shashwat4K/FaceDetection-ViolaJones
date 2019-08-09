import integral_image as ii

def enum(**enums):
    return type('Enum', (), enums)

FeatureTypes = {
    'TWO_VERTICAL': (1,2),
    'TWO_HORIZONTAL': (2,1),
    'THREE_VERTICAL': (1,3),
    'THREE_HORIZONTAL': (3,1),
    'FOUR': (2,2)
}

#FeatureType = enum(TWO_VERTICAL=(1,2), TWO_HORIZONTAL=(2,1), THREE_HORIZONTAL=(3,1), THREE_VERTICAL=(1,3), FOUR=(2,2))
#FeatureTypes = list((FeatureType.TWO_VERTICAL, FeatureType.TWO_HORIZONTAL, FeatureType.THREE_VERTICAL, FeatureType.THREE_HORIZONTAL, FeatureType.FOUR))    

class HaarLikeFeature(object):
    
    def __init__(self, feature_type, position, size, threshold, p):
        """
        Creates a new haar-like feature.
        :param feature_type: Type of new feature, see FeatureType enum
        :type feature_type: (int, int)
        :param position: Top left corner where the feature begins (x, y)
        :type position: (int, int)
        :param size: (width, height), Width * Height of the feature
        :type size: (int, int)
        :param threshold: Feature threshold
        :type threshold: float
        :param p: polarity of the feature -1 or 1
        :type p: int
        """
        self.type = feature_type
        self.top_left = position
        self.bottom_right = (position[0] + size[0], position[1] + size[1])
        self.width = size[0]
        self.height = size[1]
        self.threshold = threshold
        self.polarity = p 
        self.weight = 1

    def get_haar_feature_value(self, integral_image):
        """
        Get Haar-Feature value for given integral image array.
        :param int_img: Integral image array
        :type int_img: numpy.ndarray
        :return: Haar-Feature Value for given feature
        :rtype: float
        """    
        val = 0
        if self.type==FeatureTypes['TWO_VERTICAL']:
            first = ii.sum_of_region(integral_image, self.top_left, (self.top_left[0] + self.width, int((self.top_left[1] + self.height) / 2)))
            second = ii.sum_of_region(integral_image, (self.top_left[0], int((self.top_left[1] + self.height)/2)), self.bottom_right)
            val = first - second
        elif self.type==FeatureTypes['TWO_HORIZONTAL']:
            first = ii.sum_of_regions(integral_image, self.top_left, (int((self.top_left[0] + self.width) / 2), self.top_left[1]+self.height))
            second = ii.sum_of_region(integral_image, (int((self.top_left[0] + self.width) / 2), self.top_left[1]), self.bottom_right)
            val = first - second
        elif self.type==FeatureTypes['THREE_HORIZONTAL']:
            first = ii.sum_of_region(integral_image, self.top_left, (int(self.top_left[0] + self.width / 3), self.top_left[1] + self.height))
            second = ii.sum_of_region(integral_image, (int(self.top_left[0] + self.width / 3), self.top_left[1]), (int(self.top_left[0] + 2 * self.width / 3), self.top_left[1] + self.height))
            third = ii.sum_of_region(integral_image, (int(self.top_left[0] + 2 * self.width / 3), self.top_left[1]), self.bottom_right)
            val = first-second+third
        elif self.type==FeatureTypes['THREE_VERTICAL']:
            first = ii.sum_of_region(integral_image, self.top_left, (self.bottom_right[0], int(self.top_left[1] + self.height / 3)))
            second = ii.sum_of_region(integral_image, (self.top_left[0], int(self.top_left[1] + self.height / 3)), (self.bottom_right[0], int(self.top_left[1] + 2 * self.height / 3)))
            third = ii.sum_of_region(integral_image, (self.top_left[0], int(self.top_left[1] + 2 * self.height / 3)), self.bottom_right)
            val = first - second + third

        elif self.type==FeatureTypes['FOUR']:
            first = ii.sum_of_region(integral_image, self.top_left, (int((self.top_left[0] + self.width)/2), int((self.top_left[1]+ self.height)/2)))
            second = ii.sum_of_region(integral_image, (int((self.top_left[0] + self.width)/2), self.top_left[1]), (self.bottom_right[0], int((self.top_left[1] + self.height)/2)))
            third = ii.sum_of_region(integral_image, (self.top_left[0], int((self.top_left[1] + self.height) / 2)), (int((self.top_left[0] + self.width) / 2), self.bottom_right[1]))
            fourth = ii.sum_of_region(integral_image, (int((self.top_left[0] + self.width) / 2), int((self.top_left[1] + self.height) / 2)), self.bottom_right)
            val = first + fourth - second - third

        return val

def get_classification_value(self, integral_image):
    """
    Get the classification value for given integral image. i.e. hj(integral_image)
    :param integral_image: Integral Image array
    :type integral_image: numpy.ndarray
    :return: 1 iff this feature is classified positively, otherwise -1
    :rtype: int
    """        
    haar_feature_value = get_haar_feature_value(integral_image)

    return self.weight * (1 if haar_feature_value < self.polarity * self.threshold else -1)
