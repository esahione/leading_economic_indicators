# IHS Markit Leading Economic Indicators
# Developed by Eduardo Sahione

import ihs_lei
from pathlib import Path


ROOT_PATH = Path(ihs_lei.__file__).parents[1]
MODULE_PATH = Path(ihs_lei.__file__).parents[0]
ASSETS_PATH = ROOT_PATH / 'assets'

CONNECTION_STR = None
