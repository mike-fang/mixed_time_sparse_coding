import numpy as np
from glob import glob

tmp_files = glob('./results/tmp/*')
tmp_files.sort()
print(tmp_files[-1])
