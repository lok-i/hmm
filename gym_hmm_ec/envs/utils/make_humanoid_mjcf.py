'''
To be implemented in future if the processof xml generation is to be
1) automated
2) procedurally done in run time

refer PyMJCF: https://github.com/deepmind/dm_control/tree/master/dm_control/mjcf

'''

from dm_control import mjcf
import os

# filename = "humanoid_no_hands_mocap.xml"

# if filename.startswith("/"):
#     fullpath = filename
# else:
#     fullpath = os.path.join("./gym_hmm_ec/envs/assets/models", filename)

# # Parse from path
# mjcf_model = mjcf.from_path(fullpath)

# print(mjcf_model.worldbody.body['torso'].body)


# TODO: Add , marker sites
class Leg(object):

  def __init__( 
              self, 
              name,
              symetric_transform = 1,
              thigh_h_scale =1.,
              thigh_r_scale =1.,
              shin_h_scale = 1.,
              shin_r_scale = 1.,
              foot_l_scale = 1.,
              foot_r_scale = 1.,
              ):

    self.mjcf_model = mjcf.RootElement(model=name)
    # <body name="right_thigh" pos="0 -.1 -.04">
    self.thigh = self.mjcf_model.worldbody.add('body', name='thigh')
    # <joint name="right_hip_x" axis="1 0 0" range="-25 5"   class="big_joint"/>
    self.hip_x = self.thigh.add('joint', name='hip_x', type='hinge',damping=5,stiffness=10, axis=[1, 0, 0], range=[-25, 5], armature=.01, limited=True, solimplimit=[0, .99 ,.01])
    # <joint name="right_hip_z" axis="0 0 1" range="-60 35"  class="big_joint"/>
    self.hip_z = self.thigh.add('joint', name='hip_z', type='hinge',damping=5,stiffness=10, axis=[0, 0, 1], range=[-60, 35], armature=.01, limited=True, solimplimit=[0, .99 ,.01])
    # <joint name="right_hip_y" axis="0 1 0" range="-110 20" class="big_stiff_joint"/>
    self.hip_y = self.thigh.add('joint', name='hip_y', type='hinge',damping=5,stiffness=20, axis=[0, 1, 0], range=[-110, 20], armature=.01, limited=True, solimplimit=[0, .99 ,.01])
    # <geom name="right_thigh" fromto="0 0 0 0 .01 -.34" size=".06"/>
    self.thigh.add('geom', name='thigh', type='capsule',
                       fromto=[0, 0, thigh_h_scale*0, 0, symetric_transform*.01, thigh_h_scale*-.34], size=[thigh_r_scale*0.06])

    #  <body name="right_shin" pos="0 .01 -.403">
    self.shin = self.thigh.add('body', name='shin',pos=[0, symetric_transform*.01, thigh_h_scale*-.403])
    # <joint name="right_knee" pos="0 0 .02" axis="0 -1 0" range="-160 2"/>
    self.knee = self.shin.add('joint', name='knee', pos=[0, 0, .02], type='hinge',damping=0.2,stiffness=1, axis=[0, -1, 0], range=[-160, 2], armature=.01, limited=True, solimplimit=[0, .99 ,.01])
    # <geom name="right_shin" fromto="0 0 0 0 0 -.3"  size=".049"/>
    self.shin.add('geom', name='shin',type='capsule',fromto=[0, 0, 0, 0, 0, shin_h_scale*-.3],size=[shin_r_scale*.049])
    
    self.foot = self.shin.add('body',name='foot',pos=[0, 0, shin_h_scale*-.39])
    # <joint name="right_ankle_y" pos="0 0 .08" axis="0 1 0"   range="-50 50" stiffness="6"/>
    self.ankle_y = self.foot.add('joint', name='ankle_y', pos=[0, 0, .08], type='hinge',damping=0.2,stiffness=6, axis=[0, 1, 0], range=[-50, 50], armature=.01, limited=True, solimplimit=[0, .99 ,.01])
    # <joint name="right_ankle_x" pos="0 0 .04" axis="1 0 .5" range="-50 50" stiffness="3"/>
    self.ankle_x = self.foot.add('joint', name='ankle_x', pos=[0, 0, .04], type='hinge',damping=0.2,stiffness=3, axis=[1, 0, .5], range=[-50, 50], armature=.01, limited=True, solimplimit=[0, .99 ,.01])

    # foot lengths based on end points
    x2 = .14
    x1 = -.07
    alpha = (foot_l_scale - 1) * (x2 - x1) * 0.5
    # <geom name="right_right_foot" fromto="-.07 -.02 0 .14 -.04 0" size=".027"/>    
    self.foot.add('geom', name='right_foot',type='capsule',fromto=[x1 - alpha, symetric_transform*-.02, 0, x2+alpha, symetric_transform*-.04, 0],size=[foot_r_scale*.027])
    # <geom name="left_right_foot" fromto="-.07 0 0 .14  .02 0" size=".027"/>
    self.foot.add('geom', name='left_foot',type='capsule',fromto=[x1 - alpha, symetric_transform*0, 0, x2+alpha,  symetric_transform*.02, 0],size=[foot_r_scale*.027])
 
class Humanoid(object):

  def __init__(self,
               name,
               torso_h_scale = 1.0,
               torso_b_scale = 1.0,
               head_r_scale = 1.0,
               leg_scales =  {          
                              'thigh_h_scale':1.0,
                              'thigh_r_scale':1.0,

                              'shin_h_scale':1.0,
                              'shin_r_scale':1.0,

                              'foot_l_scale':1.0,
                              'foot_r_scale':1.0
                              }
                              
               ):

    self.mjcf_model = mjcf.RootElement(model=name)

    # <material name='grid' reflectance='0.000000' texrepeat='1 1' texture='texplane' texuniform='true'/>
    # <body name="floor" pos="0 0 0" childclass="body">
    #   <geom name="floor" type="plane" conaffinity="1" size="100 100 .2" material="grid"/>
    # </body>
    self.floor= self.mjcf_model.worldbody.add('body', name='floor',pos=[0,0,0])
    self.floor.add('geom',name='floor',type='plane',pos=[0,0,0],conaffinity=1, size=[100, 100, .2])
    
    #add n markers
    for i in range(40):
      self.marker = self.mjcf_model.worldbody.add('body', name='m'+str(i),mocap=True,pos=[0,0,0])
      self.marker.add('geom',name='m'+str(i),type='sphere', size=[0.01],rgba=[0.,1.,0.,1.0])

    #<body name="torso" pos="0 0 1.5" childclass="body">

    # <body mocap='true' name="m12" pos="1 0 0" >
    #   <geom type='sphere' size="0.01" material="mocap_material"/>
    # </body>
    # nominal_torso_h ~ 0.585
    # nominal_leg_length ~ 0.45
    clearence = 0.25 
    initial_torso_height = clearence + torso_h_scale*0.585 + leg_scales['thigh_h_scale']*0.34 + leg_scales['shin_h_scale']*0.39 +leg_scales['foot_r_scale']*.027
    self.torso = self.mjcf_model.worldbody.add('body', name='torso',pos=[0,0,initial_torso_height])#pos= base_pos)#[0, -.1 ,-.04])
    #   <light name="top" pos="0 0 2" mode="trackcom"/>
    #   <camera name="back" pos="-3 0 1" xyaxes="0 -1 0 1 0 2" mode="trackcom"/>
    #   <camera name="side" pos="0 -3 1" xyaxes="1 0 0 0 1 2" mode="trackcom"/>
    #   <freejoint name="root"/>
    self.torso.add('freejoint',name='root')
    #   <!-- <site name="root" class="force-torque"/> -->

    #   <geom name="torso" fromto="0 -.07 0 0 .07 0" size=".07"/>
    y1 = -.07
    y2 =  .07
    alpha = (torso_b_scale - 1) * (y2 - y1) * 0.5
    self.torso.add('geom', name='torso', type='capsule',
                       fromto=[0, y1 - alpha, 0, 0, y2+alpha, 0], size=[torso_h_scale*.07])  
    #   <site name="C7" type="sphere" rgba="1. 0. 0. 1." pos="-0.07 0 0.05" size="0.01"/>
    #   <site name="RSHO" type="sphere" rgba="1. 0. 0. 1." pos="0 -.14 0.025" size="0.01"/>
    #   <site name="LSHO" type="sphere" rgba="1. 0. 0. 1." pos="0 .14 0.025" size="0.01"/>
    #   <site name="CLAV" type="sphere" rgba="1. 0. 0. 1." pos=".07 0 0" size="0.01"/>

    #   <geom name="upper_waist" fromto="-.01 -.06 -.12 -.01 .06 -.12" size=".06"/>
    y1 = -.06
    y2 =  .06
    alpha = (torso_b_scale - 1) * (y2 - y1) * 0.5
    self.torso.add('geom', name='upper_waist', type='capsule',
                       fromto=[-.01, y1 - alpha, torso_h_scale*-.12, -.01, y2 + alpha, torso_h_scale*-.12], size=[torso_h_scale*.06])  
    #   <!-- <site name="torso" class="touch" type="box" pos="0 0 -.05" size=".075 .14 .13"/> -->
    #   <body name="head" pos="0 0 .19">
    self.head = self.torso.add('body', name='head',pos=[0, 0, torso_h_scale*.19])
    #     <geom name="head" type="sphere" size=".09"/>
    self.head.add('geom', name='head', type='sphere', size=[head_r_scale*.09])  
    #     <!-- <site name="head" class="touch" type="sphere" size=".091"/> -->
    #     <camera name="egocentric" pos=".09 0 0" xyaxes="0 -1 0 .1 0 1" fovy="80"/>
    #   </body>

    #   <body name="lower_waist" pos="-.01 0 -.260" quat="1.000 0 -.002 0">
    self.lower_waist = self.torso.add('body', name='lower_waist',pos=[-.01, 0, torso_h_scale*-.260],quat=[1.000, 0, -.002, 0])
    #     <geom name="lower_waist" fromto="0 -.06 0 0 .06 0" size=".06"/>
    y1 = -.06
    y2 =  .06
    alpha = (torso_b_scale - 1) * (y2 - y1) * 0.5
    self.lower_waist.add('geom', name='lower_waist', type='capsule',fromto=[0, y1-alpha, 0, 0, y2 + alpha, 0], size=[torso_h_scale*.06])
    #     <!-- <site name="lower_waist" class="touch" size=".061 .06" zaxis="0 1 0"/> -->
    #     <joint name="abdomen_z" pos="0 0 .065" axis="0 0 1" range="-45 45" class="big_stiff_joint"/>
    self.abdomen_z = self.lower_waist.add('joint', name='abdomen_z',pos=[0, 0, torso_h_scale*.065], type='hinge',damping=5,stiffness=20, axis=[0, 0, 1], range=[-45, 45], armature=.01, limited=True, solimplimit=[0, .99 ,.01])
    #     <joint name="abdomen_y" pos="0 0 .065" axis="0 1 0" range="-75 30" class="big_joint"/>
    self.abdomen_y = self.lower_waist.add('joint', name='abdomen_y',pos=[0, 0, torso_h_scale*.065], type='hinge',damping=5,stiffness=10, axis=[0, 1, 0], range=[-75, 30], armature=.01, limited=True, solimplimit=[0, .99 ,.01])    
    #     <body name="pelvis" pos="0 0 -.165" quat="1.000 0 -.002 0">
    self.pelvis = self.lower_waist.add('body', name='pelvis',pos=[0, 0, torso_h_scale*-.165],quat=[1.000, 0, -.002, 0])
    #       <joint name="abdomen_x" pos="0 0 .1" axis="1 0 0" range="-35 35" class="big_joint"/>
    self.abdomen_x = self.pelvis.add('joint', name='abdomen_x',pos=[0, 0, torso_h_scale*.1], type='hinge',damping=5,stiffness=10, axis=[1, 0, 0], range=[-35, 35], armature=.01, limited=True, solimplimit=[0, .99 ,.01])    
          
    #       <site name="RASI" type="sphere" rgba="1. 0. 0. 1." pos="0.08 -.15 0.05" size="0.01"/>
    #       <site name="LASI" type="sphere" rgba="1. 0. 0. 1." pos="0.08 .15 0.05" size="0.01"/>
    #       <site name="RPSI" type="sphere" rgba="1. 0. 0. 1." pos="-.10 -0.03 0.15" size="0.01"/>
    #       <site name="LPSI" type="sphere" rgba="1. 0. 0. 1." pos="-.10 0.03 0.15" size="0.01"/>

    #       <geom name="butt" fromto="-.02 -.07 0 -.02 .07 0" size=".09"/>
    y1 = -.07
    y2 =  .07
    alpha = (torso_b_scale - 1) * (y2 - y1) * 0.5    
    self.pelvis.add('geom', name='butt', type='capsule',fromto=[-.02, y1-alpha, 0, -.02, y2+alpha, 0], size=[torso_h_scale*.09])


    left_leg_site = self.pelvis.add(
        'site', name='left_leg_site',size=[1e-6]*3, pos=[0, torso_b_scale*.1 ,torso_h_scale*-.04])
    right_leg_site = self.pelvis.add(
        'site', name='right_leg_site',size=[1e-6]*3, pos=[0, torso_b_scale*-.1 ,torso_h_scale*-.04])

    
    self.left_leg = Leg(name='left_leg',
                        symetric_transform = -1.,
                        **leg_scales
                        )
    left_leg_site.attach(self.left_leg.mjcf_model)

    self.right_leg = Leg(name='right_leg',
                        symetric_transform = 1.,
                        **leg_scales
                        )
    right_leg_site.attach(self.right_leg.mjcf_model)

body = Humanoid(name='humanoid',
                torso_h_scale=1.0,
                torso_b_scale=1.0,
                head_r_scale=1.0,
                leg_scales = {          
                        'thigh_h_scale':1.0,
                        'thigh_r_scale':1.0,

                        'shin_h_scale':1.0,
                        'shin_r_scale':1.0,

                        'foot_l_scale':1.0,
                        'foot_r_scale':1.0
}
                )
physics = mjcf.Physics.from_mjcf_model(body.mjcf_model)
mjcf.export_with_assets(body.mjcf_model,"./gym_hmm_ec/envs/assets/models","testWrite.xml")
