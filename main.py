import sy_class

import sys
import numpy as np


cutoff = sys.argv[1]
freNum = int(sys.argv[2])

## optional
sizeinput = int(sys.argv[3])

cutoff_str = str(cutoff).replace('-','_')
# path = f"C:/Users/Shibo/Desktop/COMAP-sem2/maps/before_astro_cal/maps{cutoff_str}"
path = f"C:/Users/Shibo/Desktop/COMAP-sem2/maps/after_astro_cal/maps{cutoff_str}"
filenames_un = sy_class.get_filename_full(path, 'fits')
onlynames = sy_class.get_filename_full(path, 'fits',1)
filenames = sy_class.sortbyIband(onlynames, filenames_un)

map1 = sy_class.AstroMap(filenames[freNum])

map2 = sy_class.AstroMap(filenames[freNum+1])
# sy_class.plot_map(map1, 'primary')
# sy_class.plot_map(map1, 'primary',sizeinput)
# sy_class.plot_diffmap(map1,map2, 'primary')
# sy_class.plot_diffmap(map1,map2, 'primary',sizeinput, filter_name)