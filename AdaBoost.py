import numpy as np 
from haar_like_features import HaarLikeFeature, FeatureTypes
import progressbar
import multiprocessing
import functools

def create_features(image_dimensions, min_feature_width, max_feature_width, min_feature_height, max_feature_height):
    features = list()
    for (key, value) in FeatureTypes.items():
        feature_start_width = max(min_feature_width, value[0])
        for feature_width in range(feature_start_width, max_feature_width, value[0]):
            feature_start_height = max(min_feature_height, value[1])
            for feature_height in range(feature_start_height, max_feature_height, value[1]):
                for i in range(image_dimensions[0] - feature_width):
                    for j in range(image_dimensions[1] - feature_height):
                        features.append(HaarLikeFeature(value, (i,j), feature_width, feature_height, 0, 1))
                        features.append(HaarLikeFeature(value, (i,j), feature_width, feature_height, 0, -1))
    return features

def get_feature_result(feature, image):
    return feature.get_classification_value(image)

def learn(
    positive_integral_images,
    negative_integral_images,
    num_classifiers=-1,
    min_feature_width=1,
    max_feature_width=-1,
    min_feature_height=1,
    max_feature_height=-1
):
    l = len(positive_integral_images) # No. of positive images
    m = len(negative_integral_images) # No. of negative images
    total_images = m+l 
    image_height, image_width = positive_integral_images[0].shape

    max_feature_height = image_height if max_feature_height == -1 else max_feature_height
    max_feature_width = image_width if max_feature_width == -1 else max_feature_width

    positive_weights = np.ones(l) * 1.0 / (2 * l) #  [1/2l, 1/2l, ...]
    negative_weights = np.ones(m) * 1.0 / (2 * m) #  [1/2m, 1/2m, ...]
    weights = np.hstack((positive_weights, negative_weights)) # w(1,i) belongs to [1/2l, 1/2l, ..., 1/2l, 1/2m, 1/2m, ..., 1/2m]
    labels = np.hstack((np.ones(l), np.ones(m)*-1))

    images = positive_integral_images + negative_integral_images

    features = create_features((image_width, image_height), min_feature_width, max_feature_width, min_feature_height, max_feature_height)
    num_features = len(features)
    feature_indices = list(range(len(features)))

    num_classifiers = num_features if num_classifiers == -1 else num_classifiers

    results = np.zeros((num_images, num_features))
    bar = progressbar.ProgressBar()

    pool = multiprocessing.Pool(processes=None)

    for i in bar(range(num_images)):
        results[i, :] = np.array(list(pool.map(functools.partial(get_feature_result, image=images[i]), features)))
    # Classifier selection using AdaBoost method
    classifiers = list()

    bar = progressbar.ProgressBar()
    for i in bar(range(num_classifiers)):
        # Array of Epsilon for each feature
        classification_errors = np.zeros(len(feature_indices))
        # weight normalization so that it looks like probability distribution
        weights = weights * 1. / np.sum(weights)

        for f in range(len(feature_indices)):
            f_index = feature_indices[f]
            error = sum(map(lambda image_index: weights[image_index] if labels[image_index] != results[image_index, f_index] else 0, range(num_images)))
            classification_errors[f] = error
        # Selection of feature having minimum error (epsilon)
        min_error_index = np.argmin(classification_errors)
        best_error = classification_errors[min_error_index]
        best_feature_index = feature_indices[min_error_index]

        # set weights of best feature
        best_feature = features[best_feature_index]
        feature_weight = 0.5 * np.log((1-best_error)/best_error) # Beta function is log
        best_feature.weight = feature_weight

        classfiers.append(best_feature)

        # Update image weights according to best epsilon
        weights = np.array(list(map(lambda image_index: weights[image_index] * np.sqrt((1-best_error)/best_error) if labels[image_index] != results[image_index, best_feature_index] else weights[image_index] * np.sqrt(best_error/(1-best_error)), range(images))))

        # Once a feature is selected for a classifier, it should not be present for another classifier.
        # Therefore remove a feature from the pool once it is selected
        feature_indices.remove(best_feature_index)

    return classfiers    