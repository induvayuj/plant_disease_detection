"""
Unit tests for model components
"""

import unittest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from config import NUM_CLASSES, IMG_HEIGHT, IMG_WIDTH
from model import build_cnn_model, compile_model


class TestModel(unittest.TestCase):
    """Test model architecture"""
    
    def test_model_build(self):
        """Test if model builds successfully"""
        model = build_cnn_model()
        self.assertIsNotNone(model)
    
    def test_model_layers(self):
        """Test model has correct layers"""
        model = build_cnn_model()
        # Should have conv layers, pooling layers, and dense layers
        self.assertGreater(len(model.layers), 10)
    
    def test_model_output_shape(self):
        """Test model output shape"""
        model = build_cnn_model()
        self.assertEqual(model.output_shape[-1], NUM_CLASSES)
    
    def test_model_compile(self):
        """Test model compilation"""
        model = build_cnn_model()
        model = compile_model(model)
        self.assertIsNotNone(model.optimizer)


if __name__ == '__main__':
    unittest.main()
