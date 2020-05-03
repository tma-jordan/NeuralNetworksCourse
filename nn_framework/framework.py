# -*- coding: utf-8 -*-
"""framework.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/10ld5nFgC2KzKoIDGNGLkDL7oXs8orXOu
"""

import os
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend("agg")


class ANN(object):
    def __init__(
        self,
        model=None,
        expected_range=(-1, 1),
    ):
        self.layers = model

        self.error_history = []                           #List to capture error history
        self.n_iter_train = int(1e8)                      #Number of iterations in training set
        self.n_iter_evaluate = int(1e6)                   #Number of iterations in evaluation set
        self.viz_interval = int(1e5)                      #Number of iterations at which the error visualisation updates each time
        self.reporting_bin_size = int(1e3)                #Run this number of times and average to spot trends
        self.report_min = -3                              #Minimum amount the report focuses on - so the graph focuses on the right area
        self.report_max = 0                               #Maximum amount the report focuses on - so the graph focuses on the right area
        self.expected_range = expected_range

        self.reports_path = "reports"                     #File settings to save visualisation
        self.report_name = "performance_history.png"      #File settings to save visualisation
        # Ensure that subdirectories exist.
        try:
            os.mkdir("reports")                           #Try using OS package to make a reports directory
        except Exception:
            pass                                          #If the reports directory isn't created, move on

    def train(self, training_set):
        for i_iter in range(self.n_iter_train):
            x = self.normalize(next(training_set()).ravel())
            y = self.forward_prop(x)
            self.error_history.append(1)

            #When we reach an iteration that corresponds with a reporting individual (i.e. remainder is 0)
            if (i_iter + 1) % self.viz_interval == 0:
                #...run report() function
                self.report()


    def evaluate(self, evaluation_set):
        for i_iter in range(self.n_iter_evaluate):
            x = self.normalize(next(evaluation_set()).ravel())
            y = self.forward_prop(x)
            self.error_history.append(1)

            #When we reach an iteration that corresponds with a reporting individual (i.e. remainder is 0)
            if (i_iter + 1) % self.viz_interval == 0:
                #...run report() function
                self.report()


    def forward_prop(self, x):
        # Convert the inputs into a 2D array of the right shape
        y = x.ravel()[np.newaxis, :]
        #Run through each layer as a loop. y is made the output for each layer, which is fed into the next layer.
        for layer in self.layers:
            y = layer.forward_prop(y)
        #The final output vector is returned once all layers are worked through. np.ravel() turns this into a one dimensional array
        return y.ravel()


    def normalize(self, values):
        """
        Transform the input/output values so that they tend to
        fall between -.5 and .5
        """
        min_val = self.expected_range[0]
        max_val = self.expected_range[1]
        scale_factor = max_val - min_val
        offset_factor = min_val
        return (values - offset_factor) / scale_factor - .5

    def denormalize(self, transformed_values):
        min_val = self.expected_range[0]
        max_val = self.expected_range[1]
        scale_factor = max_val - min_val
        offset_factor = min_val
        return (transformed_values + .5) * scale_factor + offset_factor

    def report(self):
        n_bins = int(len(self.error_history) // self.reporting_bin_size)
        smoothed_history = []
        for i_bin in range(n_bins):
            smoothed_history.append(np.mean(self.error_history[
                i_bin * self.reporting_bin_size:
                (i_bin + 1) * self.reporting_bin_size
            ]))
        error_history = np.log10(np.array(smoothed_history) + 1e-10)
        ymin = np.minimum(self.report_min, np.min(error_history))
        ymax = np.maximum(self.report_max, np.max(error_history))
        fig = plt.figure()
        ax = plt.gca()
        ax.plot(error_history)
        ax.set_xlabel(f"x{self.reporting_bin_size} iterations")
        ax.set_ylabel("log error")
        ax.set_ylim(ymin, ymax)
        ax.grid()
        fig.savefig(os.path.join(self.reports_path, self.report_name))
        plt.close()