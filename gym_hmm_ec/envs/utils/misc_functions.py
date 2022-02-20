import numpy as np

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
