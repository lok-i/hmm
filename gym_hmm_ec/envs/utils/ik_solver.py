

"""Functions for computing inverse kinematics for multiple site on MuJoCo models."""

import collections

from absl import logging
from dm_control.mujoco.wrapper import mjbindings
import numpy as np

mjlib = mjbindings.mjlib


_INVALID_JOINT_NAMES_TYPE = (
    '`joint_names` must be either None, a list, a tuple, or a numpy array; '
    'got {}.')
_REQUIRE_TARGET_POS_OR_QUAT = (
    'At least one of `target_pos` or `target_quat` must be specified.')

IKResult = collections.namedtuple(
    'IKResult', ['qpos', 'err_norm', 'steps', 'success'])


def qpos_from_site_pose(physics,
                        site_names,
                        target_pos=None,
                        target_quat=None,
                        joint_names=None,
                        tol=1e-14,
                        rot_weight=1.0,
                        regularization_threshold=0.1,
                        regularization_strength=3e-2,
                        max_update_norm=2.0,
                        progress_thresh=20.0,
                        max_steps=100,
                        inplace=False):

  dtype = physics.data.qpos.dtype
  n_sites = len(site_names)
  if target_pos is not None and target_quat is not None:
    jac = np.empty((n_sites,6, physics.model.nv), dtype=dtype)
    err = np.empty(n_sites,6, dtype=dtype)
    jac_pos, jac_rot = jac[:,:3], jac[3:]
    err_pos, err_rot = err[:,:3], err[3:]
  else:
    jac = np.empty((n_sites,3, physics.model.nv), dtype=dtype)
    err = np.empty((n_sites,3), dtype=dtype)
    if target_pos is not None:
      jac_pos, jac_rot = jac, None
      err_pos, err_rot = err, None
    # elif target_quat is not None:
    #   jac_pos, jac_rot = None, jac
    #   err_pos, err_rot = None, err
    else:
      raise ValueError(_REQUIRE_TARGET_POS_OR_QUAT)

  update_nv = np.zeros(physics.model.nv, dtype=dtype)

#   if target_quat is not None:
#     site_xquat = np.empty(4, dtype=dtype)
#     neg_site_xquat = np.empty(4, dtype=dtype)
#     err_rot_quat = np.empty(4, dtype=dtype)

  if not inplace:
    physics = physics.copy(share_model=True)

  # Ensure that the Cartesian position of the site is up to date.
  mjlib.mj_fwdPosition(physics.model.ptr, physics.data.ptr)

  # Convert site name to index.
  site_id = []
  site_xpos = np.zeros((n_sites,3))
  site_xmat = np.zeros((n_sites,9))

  for i,site_name in enumerate(site_names): 
    site_id.append( physics.model.name2id(site_name, 'site') )

    # These are views onto the underlying MuJoCo buffers. mj_fwdPosition will
    # update them in place, so we can avoid indexing overhead in the main loop.
    
    site_xpos[i] = physics.named.data.site_xpos[site_name]
    site_xmat[i] = physics.named.data.site_xmat[site_name]

  # This is an index into the rows of `update` and the columns of `jac`
  # that selects DOFs associated with joints that we are allowed to manipulate.
  if joint_names is None:
    dof_indices = slice(None)  # Update all DOFs.
  elif isinstance(joint_names, (list, np.ndarray, tuple)):
    if isinstance(joint_names, tuple):
      joint_names = list(joint_names)
    # Find the indices of the DOFs belonging to each named joint. Note that
    # these are not necessarily the same as the joint IDs, since a single joint
    # may have >1 DOF (e.g. ball joints).
    indexer = physics.named.model.dof_jntid.axes.row
    # `dof_jntid` is an `(nv,)` array indexed by joint name. We use its row
    # indexer to map each joint name to the indices of its corresponding DOFs.
    dof_indices = indexer.convert_key_item(joint_names)
  else:
    raise ValueError(_INVALID_JOINT_NAMES_TYPE.format(type(joint_names)))

  steps = 0
  success = False

  for steps in range(max_steps):

    err_norm = 0.0
    # update site positions
    for i,site_name in enumerate(site_names): 
        site_xpos[i] = physics.named.data.site_xpos[site_name]
        site_xmat[i] = physics.named.data.site_xmat[site_name]

    if target_pos is not None:
      for i in range(n_sites):
          # Translational error.
        #   print(target_pos[i], site_xpos[i])
          err_pos[i,:] = target_pos[i] - site_xpos[i]
          err_norm += np.linalg.norm(err_pos[i])
    
    # if target_quat is not None:
    #   for i in range(n_sites):
    #     # Rotational error.
    #     mjlib.mju_mat2Quat(site_xquat[i], site_xmat[i])
    #     mjlib.mju_negQuat(neg_site_xquat[i], site_xquat[i])
    #     mjlib.mju_mulQuat(err_rot_quat, target_quat, neg_site_xquat)
    #     mjlib.mju_quat2Vel(err_rot, err_rot_quat, 1)
    #     err_norm += np.linalg.norm(err_rot) * rot_weight
    if err_norm < tol:
      logging.debug('Converged after %i steps: err_norm=%3g', steps, err_norm)
      success = True
      break
    else:
      # TODO(b/112141670): Generalize this to other entities besides sites.
      
      for i in range(n_sites):
        if jac_rot == None:
            mjlib.mj_jacSite(physics.model.ptr, physics.data.ptr, jac_pos[i], jac_rot, site_id[i])
      jac_joints = jac[:,:, dof_indices]

      # TODO(b/112141592): This does not take joint limits into consideration.
      reg_strength = (
          regularization_strength if err_norm > regularization_threshold
          else 0.0)
      update_joints = nullspace_method(
          jac_joints, err, regularization_strength=reg_strength)
      
    #   print('step',steps, update_joints,err_norm, tol)
      update_norm = np.linalg.norm(update_joints)

      # Check whether we are still making enough progress, and halt if not.
      progress_criterion = err_norm / update_norm
      if progress_criterion > progress_thresh:
        logging.debug('Step %2i: err_norm / update_norm (%3g) > '
                      'tolerance (%3g). Halting due to insufficient progress',
                      steps, progress_criterion, progress_thresh)
        # print('Terminated due to insufficient pogress in step',steps)
        break

      if update_norm > max_update_norm:
        update_joints *= max_update_norm / update_norm

      # Write the entries for the specified joints into the full `update_nv`
      # vector.
      update_nv[dof_indices] = update_joints

      # Update `physics.qpos`, taking quaternions into account.
      mjlib.mj_integratePos(physics.model.ptr, physics.data.qpos, update_nv, 1)

      # Compute the new Cartesian position of the site.
      mjlib.mj_fwdPosition(physics.model.ptr, physics.data.ptr)

      logging.debug('Step %2i: err_norm=%-10.3g update_norm=%-10.3g',
                    steps, err_norm, update_norm)

  if not success and steps == max_steps - 1:
    logging.warning('Failed to converge after %i steps: err_norm=%3g',
                    steps, err_norm)

  if not inplace:
    # Our temporary copy of physics.data is about to go out of scope, and when
    # it does the underlying mjData pointer will be freed and physics.data.qpos
    # will be a view onto a block of deallocated memory. We therefore need to
    # make a copy of physics.data.qpos while physics.data is still alive.
    qpos = physics.data.qpos.copy()
  else:
    # If we're modifying physics.data in place then it's fine to return a view.
    qpos = physics.data.qpos

  return IKResult(qpos=qpos, err_norm=err_norm, steps=steps, success=success)

def nullspace_method(jac_joints, delta, regularization_strength=0.0):
  """Calculates the joint velocities to achieve a specified end effector deltas.
  Reference:
    Buss, S. R. S. (2004). Introduction to inverse kinematics with jacobian
    transpose, pseudoinverse and damped least squares methods.
    https://www.math.ucsd.edu/~sbuss/ResearchWeb/ikmethods/iksurvey.pdf
  """
  n_sites = delta.shape[0]

  for i in range(n_sites):
      if i == 0:
          A = jac_joints[i].T.dot(jac_joints[i])
          b = jac_joints[i].T.dot(delta[i])
      else:
          A += jac_joints[i].T.dot(jac_joints[i])
          b += jac_joints[i].T.dot(delta[i])

  hess_approx = A
  joint_delta = b
#   hess_approx = jac_joints.T.dot(jac_joints)
#   joint_delta = jac_joints.T.dot(delta)
  
  if regularization_strength > 0:
    # L2 regularization
    hess_approx += np.eye(hess_approx.shape[0]) * regularization_strength
    return np.linalg.solve(hess_approx, joint_delta)
  else:
    return np.linalg.lstsq(hess_approx, joint_delta, rcond=-1)[0]



