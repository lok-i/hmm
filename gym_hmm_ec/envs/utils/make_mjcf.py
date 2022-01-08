'''
To be implemented in future if the processof xml generation is to be
1) automated
2) procedurally done in run time

refer PyMJCF: https://github.com/deepmind/dm_control/tree/master/dm_control/mjcf

'''

from dm_control import mjcf
import os

filename = "walker2D.xml"

if filename.startswith("/"):
    fullpath = filename
else:
    fullpath = os.path.join("./gym_hmm_ec/envs/assets", filename)

# Parse from path
mjcf_model = mjcf.from_path(fullpath)

# write the changes the fileto assets
mjcf.export_with_assets(mjcf_model,"./gym_hmm_ec/envs/assets/","testWrite.xml")
