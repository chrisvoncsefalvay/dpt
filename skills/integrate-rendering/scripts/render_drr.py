"""Project an image file through the installed DPT CLI.

Use ``dpt project`` directly when DPT is installed. Spectral, detector and
transport operations are available in their specialist package modules.
"""

import sys

from dpt.cli import main

if __name__ == "__main__":
    main(["project", *sys.argv[1:]])
