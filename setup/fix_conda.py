import sys
import re

with open(sys.argv[1]) as f:
    content = f.read()
    content = re.sub(r'#!(.+)/miniconda3/bin/python', '#!/usr/bin/env python', content)

with open(sys.argv[1], "w+") as f:
    f.write(content)
