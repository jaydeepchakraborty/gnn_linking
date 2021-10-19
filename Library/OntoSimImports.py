
#imports
import nltk
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

import re
import os

import copy

import time
import datetime
import traceback

import json
import numpy as np

import scipy.spatial.distance

import numpy as np
from collections import OrderedDict

import shutil
import h5py
from pathlib import Path

import math
import statistics

import xml.etree.ElementTree as ET

import fasttext
import fasttext.util
from fasttext import load_model

from torchbiggraph.config import parse_config
from torchbiggraph.converters.importers import TSVEdgelistReader, convert_input_data
from torchbiggraph.train import train
from torchbiggraph.util import SubprocessInitializer, setup_logging