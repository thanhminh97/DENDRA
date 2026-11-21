"""DENDRA: a language model built from biological neuron mechanisms."""
from .model import DENDRA, HebbianSTDP, DendriticSpikingLayer, CorticalRegion, Neuromodulators
from .fast import FastDendra, make_decoder
from .brain import BrainAgent
from .visual_brain import VisualBrainAgent
