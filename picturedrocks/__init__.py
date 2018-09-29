from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

name = "picturedrocks"

from . import read
from . import plot
from . import markers
from . import performance
