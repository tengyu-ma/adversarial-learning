#Copyright (C) 2017 Paolo Galeone <nessuno@nerdz.eu>
#
#This Source Code Form is subject to the terms of the Mozilla Public
#License, v. 2.0. If a copy of the MPL was not distributed with this
#file, you can obtain one at http://mozilla.org/MPL/2.0/.
#Exhibit B is not attached; this software is compatible with the
#licenses expressed under Section 1.12 of the MPL v2.
""" Evaluate Detection models """

from .interfaces import Evaluator


class DetectorEvaluator(Evaluator):
    """DetectorEvaluator is the evaluation object for a Detector model"""

    def eval(self,
             checkpoint_path,
             dataset,
             input_type,
             batch_size,
             augmentation_fn=None):
        """Eval the model, restoring weight found in checkpoint_path, using the dataset.
        Args:
            checkpoint_path: path of the trained model checkpoint directory
            dataset: implementation of the Input interface
            input_type: InputType enum
            batch_size: evaluate in batch of size batch_size
            augmentation_fn: if present, applies the augmentation to the input data

        Returns:
            value: scalar value representing the evaluation of the model,
                   on the dataset, fetching values of the specified input_type
        """
        raise ValueError("method not implemented")

    def stats(self, checkpoint_path, dataset, batch_size, augmentation_fn=None):
        """Run the eval method on the model, see eval for arguments
        and return value description.
        Moreover, adds informations about the model and returns the whole information
        in a dictionary.
        Returns:
            dict
        """
        raise ValueError("method not implemented")

    @property
    def model(self):
        """Returns the model to evaluate"""
        raise ValueError("method not implemented")

    @model.setter
    def model(self, model):
        """Set the model to evaluate.
        Args:
            model: implementation of the Model interface
        """
        raise ValueError("method not implemented")
