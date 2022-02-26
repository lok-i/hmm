# This is a customised version of the implementation available in dm_control,
# as seen here https://github.com/deepmind/dm_control/blob/master/dm_control/suite/utils/parse_amc.py
# ============================================================================

"""Parse and convert amc motion capture data."""

import collections
from gym_hmm_ec.envs.utils.misc_functions import *

import numpy as np
from scipy import interpolate

MOCAP_DT = 1.0/120.0
CONVERSION_LENGTH = 0.056444
_CMU_MOCAP_JOINT_ORDER = (
    'root0', 'root1', 'root2', 'root3', 'root4', 'root5', 'lowerbackrx',
    'lowerbackry', 'lowerbackrz', 'upperbackrx', 'upperbackry', 'upperbackrz',
    'thoraxrx', 'thoraxry', 'thoraxrz', 'lowerneckrx', 'lowerneckry',
    'lowerneckrz', 'upperneckrx', 'upperneckry', 'upperneckrz', 'headrx',
    'headry', 'headrz', 'rclaviclery', 'rclaviclerz', 'rhumerusrx',
    'rhumerusry', 'rhumerusrz', 'rradiusrx', 'rwristry', 'rhandrx', 'rhandrz',
    'rfingersrx', 'rthumbrx', 'rthumbrz', 'lclaviclery', 'lclaviclerz',
    'lhumerusrx', 'lhumerusry', 'lhumerusrz', 'lradiusrx', 'lwristry',
    'lhandrx', 'lhandrz', 'lfingersrx', 'lthumbrx', 'lthumbrz', 'rfemurrx',
    'rfemurry', 'rfemurrz', 'rtibiarx', 'rfootrx', 'rfootrz', 'rtoesrx',
    'lfemurrx', 'lfemurry', 'lfemurrz', 'ltibiarx', 'lfootrx', 'lfootrz',
    'ltoesrx'
)

Converted = collections.namedtuple('Converted',
                                   ['qpos', 'qvel', 'time'])



def convert(file_name, index2joint, timestep):
  """Converts the parsed .amc values into qpos and qvel values and resamples.
  Args:
    file_name: The .amc file to be parsed and converted.
    physics: The corresponding physics instance.
    timestep: Desired output interval between resampled frames.
  Returns:
    A namedtuple with fields:
        `qpos`, a numpy array containing converted positional variables.
        `qvel`, a numpy array containing converted velocity variables.
        `time`, a numpy array containing the corresponding times.
  """
  frame_values = parse(file_name)
#   joint2index = {}
#   for name in physics.named.data.qpos.axes.row.names:
#     joint2index[name] = physics.named.data.qpos.axes.row.convert_key_item(name)
#   index2joint = {}
#   for joint, index in joint2index.items():
#     if isinstance(index, slice):
#       indices = range(index.start, index.stop)
#     else:
#       indices = [index]
#     for ii in indices:
#       index2joint[ii] = joint

  # Convert frame_values to qpos
  amcvals2qpos_transformer = Amcvals2qpos(index2joint, _CMU_MOCAP_JOINT_ORDER)
  qpos_values = []
  for frame_value in frame_values:
    qpos_values.append(amcvals2qpos_transformer(frame_value))
  qpos_values = np.stack(qpos_values)  # Time by nq

  # Interpolate/resample.
  # Note: interpolate quaternions rather than euler angles (slerp).
  # see https://en.wikipedia.org/wiki/Slerp
  qpos_values_resampled = []
  time_vals = np.arange(0, len(frame_values)*MOCAP_DT - 1e-8, MOCAP_DT)
  time_vals_new = np.arange(0, len(frame_values)*MOCAP_DT, timestep)
  while time_vals_new[-1] > time_vals[-1]:
    time_vals_new = time_vals_new[:-1]

  for i in range(qpos_values.shape[1]):
    f = interpolate.splrep(time_vals, qpos_values[:, i])
    qpos_values_resampled.append(interpolate.splev(time_vals_new, f))

  qpos_values_resampled = np.stack(qpos_values_resampled)  # nq by ntime

  qvel_list = []
  for t in range(qpos_values_resampled.shape[1]-1):
    p_tp1 = qpos_values_resampled[:, t + 1]
    p_t = qpos_values_resampled[:, t]
    qvel = [(p_tp1[:3]-p_t[:3])/ timestep,
            mj_quat2vel(
                mj_quatdiff(p_t[3:7], p_tp1[3:7]), timestep),
            (p_tp1[7:]-p_t[7:])/ timestep]
    qvel_list.append(np.concatenate(qvel))

  qvel_values_resampled = np.vstack(qvel_list).T

  return Converted(qpos_values_resampled, qvel_values_resampled, time_vals_new)


def parse(file_name):
  """Parses the amc file format."""
  values = []
  fid = open(file_name, 'r')
  line = fid.readline().strip()
  frame_ind = 1
  first_frame = True
  while True:
    # Parse first frame.
    if first_frame and line[0] == str(frame_ind):
      first_frame = False
      frame_ind += 1
      frame_vals = []
      while True:
        line = fid.readline().strip()
        if not line or line == str(frame_ind):
          values.append(np.array(frame_vals, dtype=np.float))
          break
        tokens = line.split()
        frame_vals.extend(tokens[1:])
    # Parse other frames.
    elif line == str(frame_ind):
      frame_ind += 1
      frame_vals = []
      while True:
        line = fid.readline().strip()
        if not line or line == str(frame_ind):
          values.append(np.array(frame_vals, dtype=np.float))
          break
        tokens = line.split()
        frame_vals.extend(tokens[1:])
    else:
      line = fid.readline().strip()
      if not line:
        break
  return values


class Amcvals2qpos:
  """Callable that converts .amc values for a frame and to MuJoCo qpos format.
  """

  def __init__(self, index2joint, joint_order):
    """Initializes a new Amcvals2qpos instance.
    Args:
      index2joint: List of joint angles in .amc file.
      joint_order: List of joint names in MuJoco MJCF.
    """
    # Root is x,y,z, then quat.
    # need to get indices of qpos that order for amc default order
    self.qpos_root_xyz_ind = [0, 1, 2]
    self.root_xyz_ransform = np.array(
        [[1, 0, 0], [0, 0, -1], [0, 1, 0]]) * CONVERSION_LENGTH
    self.qpos_root_quat_ind = [3, 4, 5, 6]
    amc2qpos_transform = np.zeros((len(index2joint), len(joint_order)))
    for i in range(len(index2joint)):
      for j in range(len(joint_order)):
        if index2joint[i] == joint_order[j]:
          if 'rx' in index2joint[i]:
            amc2qpos_transform[i][j] = 1
          elif 'ry' in index2joint[i]:
            amc2qpos_transform[i][j] = 1
          elif 'rz' in index2joint[i]:
            amc2qpos_transform[i][j] = 1
    self.amc2qpos_transform = amc2qpos_transform

  def __call__(self, amc_val):
    """Converts a `.amc` frame to MuJoCo qpos format."""
    amc_val_rad = np.deg2rad(amc_val)
    qpos = np.dot(self.amc2qpos_transform, amc_val_rad)

    # Root.
    qpos[:3] = np.dot(self.root_xyz_ransform, amc_val[:3])
    qpos_quat = euler2quat(amc_val[3], amc_val[4], amc_val[5])
    qpos_quat = mj_quatprod(euler2quat(90, 0, 0), qpos_quat)

    for i, ind in enumerate(self.qpos_root_quat_ind):
      qpos[ind] = qpos_quat[i]

    return qpos