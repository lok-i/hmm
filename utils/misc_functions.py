import numpy as np
from mujoco_py import functions 


# For testing whether a number is close to zero
_FLOAT_EPS = np.finfo(np.float64).eps
_EPS4 = _FLOAT_EPS * 4.0

def euler2quat(ax, ay, az):
  """Converts euler angles to a quaternion.
  Note: rotation order is zyx
  Args:
    ax: Roll angle (deg)
    ay: Pitch angle (deg).
    az: Yaw angle (deg).
  Returns:
    A numpy array representing the rotation as a quaternion.
  """
  r1 = az
  r2 = ay
  r3 = ax

  c1 = np.cos(np.deg2rad(r1 / 2))
  s1 = np.sin(np.deg2rad(r1 / 2))
  c2 = np.cos(np.deg2rad(r2 / 2))
  s2 = np.sin(np.deg2rad(r2 / 2))
  c3 = np.cos(np.deg2rad(r3 / 2))
  s3 = np.sin(np.deg2rad(r3 / 2))

  q0 = c1 * c2 * c3 + s1 * s2 * s3
  q1 = c1 * c2 * s3 - s1 * s2 * c3
  q2 = c1 * s2 * c3 + s1 * c2 * s3
  q3 = s1 * c2 * c3 - c1 * s2 * s3

  return np.array([q0, q1, q2, q3])

def mj_quatprod(q, r):
  quaternion = np.zeros(4)
  functions.mju_mulQuat(quaternion, np.ascontiguousarray(q),
                    np.ascontiguousarray(r))
  return quaternion

def mj_quat2vel(q, dt):
  vel = np.zeros(3)
  functions.mju_quat2Vel(vel, np.ascontiguousarray(q), dt)
  return vel

def mj_quatneg(q):
  quaternion = np.zeros(4)
  functions.mju_negQuat(quaternion, np.ascontiguousarray(q))
  return quaternion

def mj_quatdiff(source, target):
  return mj_quatprod(mj_quatneg(source), np.ascontiguousarray(target))

def R_axis_angle(axis, angle):

    # Trig factors.
    ca = np.cos(angle)
    sa = np.sin(angle)
    C = 1 - ca

    # Depack the axis.
    x, y, z = axis

    # Multiplications (to remove duplicate calculations).
    xs = x*sa
    ys = y*sa
    zs = z*sa
    xC = x*C
    yC = y*C
    zC = z*C
    xyC = x*yC
    yzC = y*zC
    zxC = z*xC

    # Update the rotation matrix.
    matrix = np.zeros((3,3))
    matrix[0, 0] = x*xC + ca
    matrix[0, 1] = xyC - zs
    matrix[0, 2] = zxC + ys
    matrix[1, 0] = xyC + zs
    matrix[1, 1] = y*yC + ca
    matrix[1, 2] = yzC - xs
    matrix[2, 0] = zxC - ys
    matrix[2, 1] = yzC + xs
    matrix[2, 2] = z*zC + ca

    return matrix

def calc_rotation_vec_a2b(frc_ptb):

    f_mag = np.linalg.norm(np.array(frc_ptb))

    norm_frc_ptb_vec = np.array(frc_ptb)/f_mag

    a_vec = np.array([0.,0.,1.])
    b_vec = np.array(norm_frc_ptb_vec)

    if b_vec[0] == 0 and b_vec[1] == 0.:
        # Vertically upwards arrow
        rotation_mat = np.array(
                                [
                                    [1,0, 0.],
                                    [0,1.,0],
                                    [0.,0,1],
                                ]
                                )

    else:
        # any arrow not parallel to z axis,[0,0,1]
        v_vec = np.cross(a_vec, b_vec)
        v_vec = v_vec/np.linalg.norm(v_vec)

        c_vec = np.arccos(np.dot(a_vec, b_vec))

        rotation_mat = R_axis_angle(v_vec, c_vec)

    return rotation_mat



def quat2mat(quat):
    """ Convert Quaternion to Euler Angles.  See rotation.py for notes """
    quat = np.asarray(quat, dtype=np.float64)
    assert quat.shape[-1] == 4, "Invalid shape quat {}".format(quat)

    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    Nq = np.sum(quat * quat, axis=-1)
    s = 2.0 / Nq
    X, Y, Z = x * s, y * s, z * s
    wX, wY, wZ = w * X, w * Y, w * Z
    xX, xY, xZ = x * X, x * Y, x * Z
    yY, yZ, zZ = y * Y, y * Z, z * Z

    mat = np.empty(quat.shape[:-1] + (3, 3), dtype=np.float64)
    mat[..., 0, 0] = 1.0 - (yY + zZ)
    mat[..., 0, 1] = xY - wZ
    mat[..., 0, 2] = xZ + wY
    mat[..., 1, 0] = xY + wZ
    mat[..., 1, 1] = 1.0 - (xX + zZ)
    mat[..., 1, 2] = yZ - wX
    mat[..., 2, 0] = xZ - wY
    mat[..., 2, 1] = yZ + wX
    mat[..., 2, 2] = 1.0 - (xX + yY)
    return np.where((Nq > _FLOAT_EPS)[..., np.newaxis, np.newaxis], mat, np.eye(3))

def mat2euler(mat):
    """ Convert Rotation Matrix to Euler Angles.  See rotation.py for notes """
    mat = np.asarray(mat, dtype=np.float64)
    assert mat.shape[-2:] == (3, 3), "Invalid shape matrix {}".format(mat)

    cy = np.sqrt(mat[..., 2, 2] * mat[..., 2, 2] + mat[..., 1, 2] * mat[..., 1, 2])
    condition = cy > _EPS4
    euler = np.empty(mat.shape[:-1], dtype=np.float64)
    euler[..., 2] = np.where(condition,
                             -np.arctan2(mat[..., 0, 1], mat[..., 0, 0]),
                             -np.arctan2(-mat[..., 1, 0], mat[..., 1, 1]))
    euler[..., 1] = np.where(condition,
                             -np.arctan2(-mat[..., 0, 2], cy),
                             -np.arctan2(-mat[..., 0, 2], cy))
    euler[..., 0] = np.where(condition,
                             -np.arctan2(mat[..., 1, 2], mat[..., 2, 2]),
                             0.0)
    return euler

def quat2euler(quat):
    """ Convert Quaternion to Euler Angles.  See rotation.py for notes """
    return mat2euler(quat2mat(quat))