from Utilities.Argparser import Argparser
from Utilities.Data import Data
import numpy as np
from Utilities.Paths import config
import unittest


class TestData(unittest.TestCase):
    def test_prepare_batch_size_per_classes(self):
        # set variables
        args = Argparser.args_parser()
        batch_size_per_classes = {}
        batch_size_options = config["options"]["batch_size_per_class"]

        Data.prepare_batch_sizes_per_classes(args, batch_size_options, batch_size_per_classes)

        # assert functionality
        for i in batch_size_per_classes.keys():
            self.assertIn(batch_size_per_classes[i], batch_size_options)

        # assert reproducibility
        batch_sizes = {'0': 200, '3': 1000, '7': 500, '9': 300}
        self.assertDictEqual(batch_sizes, batch_size_per_classes)

    def test_prepare_dataset(self):
        # set variables
        path_to_dataset_form_project_root = "\\fashion dataset\\fashion-mnist_test.csv"
        batch_sizes = {'3': 10, '9': 10, '2': 20}
        classes = [2, 3, 9]
        batch_size = 40
        local_epoch = 10
        standard_deviation_for_noises = 0
        x_train, y_train, y_original_labels = Data.prepare_dataset(path_to_dataset_form_project_root, batch_sizes,
                                                                   local_epoch, standard_deviation_for_noises)

        # assert functions
        self.assertEqual(local_epoch, len(x_train))
        for i in range(local_epoch):
            self.assertEqual((batch_size, 1, 28, 28), x_train[str(i)].shape)
            self.assertEqual((batch_size, 10), y_train[str(i)].shape)
        self.assertListEqual(classes, sorted(np.unique(y_original_labels[str(i)])))

        # assert reproducibility
        y_labels = {'0': np.array([9., 2., 9., 9., 2., 9., 2., 3., 9., 2., 2., 2., 3., 2., 9., 2., 9.,
           2., 2., 9., 3., 3., 2., 9., 9., 3., 2., 3., 2., 3., 2., 3., 2., 2.,
           2., 3., 2., 3., 2., 2.], dtype=np.float32),
           '1': np.array([2., 9., 9., 2., 9., 9., 9., 2., 2., 2., 2., 3., 2., 2., 9., 2., 3.,
           2., 2., 2., 3., 2., 3., 2., 2., 9., 2., 3., 2., 9., 2., 2., 2., 3.,
           3., 9., 9., 3., 3., 3.], dtype=np.float32),
           '2': np.array([9., 2., 3., 3., 2., 2., 3., 3., 2., 2., 2., 3., 9., 2., 2., 2., 9.,
           9., 9., 2., 2., 2., 3., 9., 3., 3., 2., 2., 2., 9., 9., 2., 2., 2.,
           3., 9., 3., 2., 2., 9.], dtype=np.float32),
           '3': np.array([3., 9., 9., 2., 2., 2., 9., 2., 9., 2., 9., 3., 2., 2., 3., 2., 3.,
           3., 3., 2., 9., 2., 2., 2., 9., 2., 3., 2., 9., 9., 2., 3., 2., 3.,
           2., 2., 2., 2., 3., 9.], dtype=np.float32),
           '4': np.array([2., 9., 2., 3., 3., 3., 2., 9., 2., 3., 9., 2., 2., 2., 3., 9., 2.,
           3., 3., 2., 2., 3., 2., 9., 3., 9., 2., 9., 2., 2., 2., 2., 2., 9.,
           9., 2., 3., 2., 9., 2.], dtype=np.float32),
           '5': np.array([9., 9., 2., 2., 2., 2., 2., 2., 2., 9., 9., 9., 9., 2., 2., 2., 2.,
           2., 2., 3., 9., 3., 3., 3., 9., 3., 9., 3., 2., 2., 2., 3., 3., 3.,
           3., 2., 2., 9., 2., 2.], dtype=np.float32),
           '6': np.array([2., 2., 2., 2., 2., 9., 2., 9., 2., 2., 9., 3., 3., 2., 3., 2., 2.,
           3., 2., 2., 2., 9., 9., 2., 3., 3., 9., 2., 3., 9., 3., 9., 9., 2.,
           2., 2., 3., 2., 9., 3.], dtype=np.float32),
           '7': np.array([2., 2., 9., 2., 2., 3., 2., 3., 3., 2., 2., 2., 3., 2., 9., 9., 9.,
           3., 2., 3., 3., 2., 2., 3., 2., 9., 9., 3., 9., 2., 2., 2., 2., 9.,
           2., 2., 9., 2., 3., 9.], dtype=np.float32),
           '8': np.array([2., 3., 9., 2., 9., 9., 3., 2., 2., 3., 2., 3., 2., 9., 2., 3., 2.,
           2., 3., 9., 2., 2., 9., 2., 2., 3., 2., 3., 9., 3., 9., 3., 2., 2.,
           9., 2., 2., 2., 2., 9.], dtype=np.float32),
           '9': np.array([3., 3., 2., 3., 9., 3., 2., 2., 2., 2., 2., 9., 2., 2., 2., 2., 2.,
           2., 9., 9., 3., 3., 2., 9., 2., 2., 3., 9., 2., 9., 2., 2., 2., 3.,
           9., 2., 9., 3., 3., 9.], dtype=np.float32)}

        for i in y_labels.keys():
            self.assertEqual(y_labels[i].tolist(), y_original_labels[i].tolist())

if __name__ == "__main__":
    unittest.main()
