import numpy as np
import pandas as pd
from Utilities.Paths import Paths

class Data():
    # preparing batch sizes for the classes in the dataset for non-IID settings
    @staticmethod
    def prepare_batch_sizes_per_classes(args, batch_size_options, batch_size_per_classes, random_seed=0):
        np.random.seed(random_seed)

        number_of_classes = np.random.randint(args.number_of_classes_in_dataset)+1

        classes_of_data_object_per_agent = list(
            np.random.choice(args.number_of_classes_in_dataset, number_of_classes))

        for i in classes_of_data_object_per_agent:
            batch_size_per_classes[str(i)] = batch_size_options[np.random.randint(len(batch_size_options))]

    # prepare dataset for learning and predicting
    @staticmethod
    def prepare_dataset(path_to_dataset_from_project_root, batch_sizes_per_classes=
                        {'0': 300, '1': 300, '2': 300, '3': 300, '4': 300, '5': 300, '6': 300, '7': 300, '8': 300, '9': 300},
                        local_epochs=5, standard_deviation_for_noises=0, random_seed=0):
        np.random.seed(random_seed)

        if local_epochs <= 0:
            local_epochs = 1

        dataframe = pd.read_csv(str(Paths.get_project_root()) + path_to_dataset_from_project_root)

        x_test = {}
        y_test = {}
        y_labels = {}
        for i in range(local_epochs):
            # create a dataframe with the specified classes and sizes
            sample = pd.DataFrame()
            for j in batch_sizes_per_classes.keys():
                x = dataframe[dataframe['label'] == int(j)]
                sample = pd.concat([sample, x.sample(batch_sizes_per_classes[j], replace=False)], ignore_index=True)

            # shuffle
            sample = sample.sample(frac = 1)

            # get images, labels and shape of images
            image_shape = (sample.shape[0], 28, 28)
            sample = np.array(sample, dtype='float32')
            images = sample[:, 1:]
            labels = sample[:, 0]

            # reshaping needed for ML
            if image_shape is not None:
                images = images.reshape(image_shape)
            images = np.expand_dims(images, axis=1)
            labels_reshape = np.zeros((sample.shape[0], 10), dtype='float32')
            for b in range(sample.shape[0]):
                labels_reshape[b, int(labels[b])] = 1

            # adding noise
            noise = np.random.normal(0, standard_deviation_for_noises, images.shape).astype(np.float32)
            images_noised = images + noise

            x_test[str(i)] = images_noised
            y_test[str(i)] = labels_reshape
            y_labels[str(i)] = labels
        return x_test, y_test, y_labels
