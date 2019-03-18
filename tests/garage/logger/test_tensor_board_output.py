import random
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from garage.logger import TabularInput
from garage.logger import TensorBoardOutput
from tests.fixtures import TfTestCase


class TestTensorBoardOutput(TfTestCase):
    def setUp(self):
        self.log_dir = tempfile.TemporaryDirectory()
        self.tensor_board_output = TensorBoardOutput(self.log_dir.name)
        self.tabular = TabularInput()
        self.tabular.clear()

    def tearDown(self):
        self.tensor_board_output.close()

    def test_record(self):
        with tf.Session():
            foo = random.randint(0, 10)
            bar = random.randint(0, 20)
            self.tabular.record('foo', foo)
            self.tabular.record('bar', bar)

            fig = plt.figure()
            ax = fig.gca()
            xs = np.arange(10.0)
            ys = np.random.rand(10)
            ax.scatter(xs, ys)
            self.tabular.record('baz', fig)
            self.tensor_board_output.record(self.tabular)

            histo_shape = np.ones((1000, 10))
            normal = tfp.distributions.Normal(
                loc=np.random.rand() * histo_shape,
                scale=np.random.rand() * histo_shape)
            gamma = tfp.distributions.Gamma(
                concentration=np.random.rand() * histo_shape,
                rate=np.random.rand() * histo_shape)
            poisson = tfp.distributions.Poisson(
                rate=np.random.rand() * histo_shape)
            uniform = tfp.distributions.Uniform(
                high=np.random.rand() * histo_shape)
            self.tensor_board_output.record(normal)
            self.tensor_board_output.record(gamma)
            self.tensor_board_output.record(poisson)
            self.tensor_board_output.record(uniform)

    def test_unknown_input_type(self):
        with self.assertRaises(ValueError):
            foo = np.zeros((3, 10))
            self.tensor_board_output.record(foo)

    def test_unknown_tabular_data(self):
        with self.assertRaises(ValueError):
            foo = np.zeros((3, 10))
            self.tabular.record('foo', foo)
            self.tensor_board_output.record(self.tabular)
