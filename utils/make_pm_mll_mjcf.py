from dm_control import mjcf
import numpy as np
import argparse 
import yaml
import os
from utils import misc_functions


class Leg(object):

  def __init__( 
              self, 
              name,
              knee_actuation,
              marker_pos_params,
              thigh_h_scale =1.,
              thigh_r_scale =1.,
              shin_h_scale = 1.,
              shin_r_scale = 1.,

              foot_r_scale = 1.,
              
              ):

    self.mjcf_model = mjcf.RootElement(model=name)
    # <body name="right_thigh" pos="0 -.1 -.04">
    self.thigh = self.mjcf_model.worldbody.add('body', name='thigh')
    # <joint name="right_hip_x" axis="1 0 0" range="-25 5"   class="big_joint"/>
    self.hip_x = self.thigh.add('joint', 
                                  name='hip_x', 
                                  type='hinge',
                                  # damping=5, # TO ASK
                                  # stiffness=10, # TO ASK
                                  axis=[1, 0, 0], 
                                  range=[-25, 5], 
                                  # armature=.01,  # TO ASK
                                  limited=True, 
                                  solimplimit=[0, .99 ,.01]
                                  )
    # <joint name="right_hip_z" axis="0 0 1" range="-60 35"  class="big_joint"/>
    # self.hip_z = self.thigh.add('joint', name='hip_z', type='hinge',damping=5,stiffness=10, axis=[0, 0, 1], range=[-60, 35], armature=.01, limited=True, solimplimit=[0, .99 ,.01])
    # <joint name="right_hip_y" axis="0 1 0" range="-110 20" class="big_stiff_joint"/>

    self.hip_y = self.thigh.add('joint', 
                                  name='hip_y', 
                                  type='hinge',
                                  # damping=5, # TO ASK
                                  # stiffness=20, # TO ASK
                                  axis=[0, 1, 0], 
                                  range=[-110, 20],
                                  # armature=.01, # TO ASK 
                                  limited=True, 
                                  solimplimit=[0, .99 ,.01]
                                  )
    # <geom name="right_thigh" fromto="0 0 0 0 .01 -.34" size=".06"/>
    
    thigh_name = 'thigh'
    thigh_radius = thigh_r_scale*0.06
    thigh_length = thigh_h_scale*.34 - thigh_radius  
    self.thigh.add('geom', 
                    name='thigh', 
                    type='capsule',
                    fromto=[0, 0, thigh_h_scale*0, 0, 0, -thigh_length ], 
                    size=[thigh_radius],
                    mass=1e-6 # negligible mass
                    )
    for t_m in marker_pos_params[thigh_name]:
      
      r = thigh_r_scale*marker_pos_params[thigh_name][t_m]['r_nominal'] if 'r_nominal' in marker_pos_params[thigh_name][t_m].keys() else thigh_radius
      theta = np.radians(marker_pos_params[thigh_name][t_m]['theta'])
      k = marker_pos_params[thigh_name][t_m]['k']
      
      self.thigh.add(
          'site', name=t_m,type="sphere", rgba="1. 0. 0. 1.",size=[0.01], 
          pos=[
              r*np.cos(theta), 
              r*np.sin(theta), 
              k*0.5*thigh_length - 0.5*thigh_length,
              ]
              )                      

    #  <body name="right_shin" pos="0 .01 -.403">
    self.shin = self.thigh.add('body', name='shin',
                                pos=[0, 
                                     0, 
                                     thigh_h_scale*-.403 + thigh_radius]
                                     )
    # <joint name="right_knee" pos="0 0 .02" axis="0 -1 0" range="-160 2"/>
    
    series_spring_stiffness = knee_actuation['series_spring_stiffness'] if 'series_spring_stiffness' in knee_actuation.keys() else 1.
    series_spring_damping = knee_actuation['series_spring_damping'] if 'series_spring_damping' in knee_actuation.keys() else 0.2

    if knee_actuation['joint'] == 'slide':
        self.knee = self.shin.add('joint', name='knee', 
                                pos=[0, 0, .02],
                                type='slide',
                                damping=series_spring_damping,
                                stiffness=series_spring_stiffness, 
                                axis=[0, 0, 1], range=[-160, 2], 
                                armature=.01, limited=True, solimplimit=[0, .99 ,.01],
                                )
    else:
        self.knee = self.shin.add('joint', name='knee', 
                                pos=[0, 0, .02],
                                type='hinge',
                                damping=series_spring_damping,
                                stiffness=series_spring_stiffness, 
                                axis=[0, -1, 0], range=[-160, 2], 
                                armature=.01, limited=True, solimplimit=[0, .99 ,.01])       
    # <geom name="right_shin" fromto="0 0 0 0 0 -.3"  size=".049"/>
    
    shin_name = 'shin'
    shin_radius = shin_r_scale*.049
    shin_length = shin_h_scale*.3 - 2*shin_radius
    
    self.shin.add('geom', 
                    name='shin',
                    type='capsule',
                    fromto=[0, 0, 0, 0, 0, -shin_length ],
                    size=[shin_radius],
                    mass=1e-6 # negligible mass
                    )
                        
    for t_m in marker_pos_params[shin_name]:
      
      r = shin_r_scale*marker_pos_params[shin_name][t_m]['r_nominal'] if 'r_nominal' in marker_pos_params[shin_name][t_m].keys() else shin_radius
      theta = np.radians(marker_pos_params[shin_name][t_m]['theta'])
      k = marker_pos_params[shin_name][t_m]['k']
      
      self.shin.add(
          'site', name=t_m,type="sphere", rgba="1. 0. 0. 1.",size=[0.01], 
          pos=[
              r*np.cos(theta), 
              r*np.sin(theta), 
              k*0.5*shin_length - 0.5*shin_length,
              ]
              )   

    ankle_clearence =  0.0
    foot_radius = foot_r_scale*.03
    self.foot = self.shin.add('body',name='foot',
                              pos=[0, 0, (shin_h_scale*-.3) - (foot_radius+ ankle_clearence) + 2*shin_radius ],
                              )
    #     <geom name="head" type="sphere" size=".09"/>
    self.foot.add('geom', 
                    name='foot',
                    pos=[0, 0, 0], 
                    type='sphere', 
                    size=[foot_radius],
                    mass=1e-1 # negligible mass, less than 0.1, it sinkss
                    )  

    self.foot.add(
        'site', name='foot',type="sphere", rgba="1. 0. 0. 1.",size=[0.01], 
        pos=[0.,0.,0.]
            ) 
    # <motor name="right_hip_x"     gear="40"  joint="right_hip_x"/> <!-- roll -->
    # <motor name="right_hip_z"     gear="40"  joint="right_hip_z"/> <!-- yaw -->
    # <motor name="right_hip_y"     gear="120" joint="right_hip_y"/> <!-- pitch -->
    # <motor name="right_knee"      gear="80"  joint="right_knee"/> <!-- pitch -->
    # <motor name="right_ankle_y"   gear="20"  joint="right_ankle_y"/> <!-- pitch -->
    # <motor name="right_ankle_x"   gear="20"  joint="right_ankle_x"/> <!-- roll -->

    self.mjcf_model.actuator.add("motor",name='hip_x',joint='hip_x')
    self.mjcf_model.actuator.add("motor",name='hip_y',joint='hip_y')

    self.mjcf_model.actuator.add(
                                  'motor',
                                  name='knee',
                                  joint='knee',
                                  )

class Pm_mll(object):

  def __init__(self,
               name='pm_mll',
               ground_clearence = 0.004,
               total_mass = 50.,
               com_radius = 0.1,
               torso_h_scale = 1,
               torso_b_scale = 1,
               knee_actuation = 
               {
                    'joint':'hinge',
                    'series_spring_stiffness': 0.,
                    'series_spring_damping': 0.
               },
               leg_scales =  
                            {
                                'left_leg':                           
                                  {
                                  'thigh_h_scale':1.0,
                                  'thigh_r_scale':1.0,

                                  'shin_h_scale':1.0,
                                  'shin_r_scale':1.0,


                                  'foot_r_scale':1.0
                                  },
                              
                                'right_leg':                           
                                  {
                                  'thigh_h_scale':1.0,
                                  'thigh_r_scale':1.0,

                                  'shin_h_scale':1.0,
                                  'shin_r_scale':1.0,


                                  'foot_r_scale':1.0
                                  }                        
                              
                              },


               marker_pos_params=
                            {


                              'torso':
                                  {
                                    'RASI' : {'theta':40,'k':-1.5,}, #if r is given, r_link is over written
                                    'LASI' : {'theta':40,'k': 1.5},
                                    'RPSI' : {'theta':115,'k':-.3,'r_nominal':0.11},
                                    'LPSI' : {'theta':115,'k':0.3,'r_nominal':0.11},
                                  },

                              'left_leg':
                              {

                                  'thigh': 
                                    {
                                      'LGT': {'theta': 50, 'k': 1.0},
                                      'LT1': {'theta': 0, 'k': 0.3}, 
                                      'LT2': {'theta': 30, 'k': 0}, 
                                      'LT3': {'theta': 0, 'k': -0.3}, 
                                      'LT4': {'theta': -30, 'k': 0}
                                    }, 
                                  'shin': 
                                      {
                                        'LKNL': {'theta': 90, 'k': 1.0}, 
                                        'LKNM': {'theta': 270, 'k': 1.0},
                                        'LS1': {'theta': 0, 'k': 0.3}, 
                                        'LS2': {'theta': 30, 'k': 0}, 
                                        'LS3': {'theta': 0, 'k': -0.3}, 
                                        'LS4': {'theta': -30, 'k': 0},
                                        'LANL': {'theta': 90, 'k': -1.0}, 
                                        'LANM': {'theta': 270, 'k': -1.0},
                                      }, 


                              },
                              'right_leg':
                              {

                                  'thigh': 
                                    {
                                      'RGT': {'theta': 310, 'k': 1.0},
                                      'RT1': {'theta': 0, 'k': 0.3}, 
                                      'RT2': {'theta': 30, 'k': 0}, 
                                      'RT3': {'theta': 0, 'k': -0.3}, 
                                      'RT4': {'theta': -30, 'k': 0}
                                    }, 
                                  'shin': 
                                      {
                                        'RKNL': {'theta': 270, 'k': 1.0}, 
                                        'RKNM': {'theta': 90, 'k': 1.0},
                                        'RS1': {'theta': 0, 'k': 0.3}, 
                                        'RS2': {'theta': 30, 'k': 0}, 
                                        'RS3': {'theta': 0, 'k': -0.3}, 
                                        'RS4': {'theta': -30, 'k': 0},
                                        'RANL': {'theta': 90, 'k': -1.0}, 
                                        'RANM': {'theta': 270, 'k': -1.0},

                                      }, 

                              }
                            }, 
                              
               ):

    self.mjcf_model = mjcf.RootElement(model=name)

    # <body name="floor" pos="0 0 0" childclass="body">
    #   <geom name="floor" type="plane" conaffinity="1" size="100 100 .2" material="grid"/>
    # </body>
    
    # self.floor= self.mjcf_model.worldbody.add('body', name='floor',pos=[0,0,0])
    # <texture builtin='checker' height='512' name='texplane' rgb1='0.2 0.2 0.2' rgb2='0.3 0.3 0.3' type='2d' width='512'/>

    self.mjcf_model.compiler.settotalmass=total_mass
    self.mjcf_model.asset.add('texture', builtin='checker', height=512, name='texplane', rgb1=[0.2, 0.2, 0.2], rgb2=[0.3, 0.3, 0.3], type='2d', width=512)
    # <material name='grid' reflectance='0.000000' texrepeat='1 1' texture='texplane' texuniform='true'/>
    self.mjcf_model.asset.add('material', name='grid', reflectance=0.0, texrepeat=[1, 1], texture='texplane', texuniform=True)


    
    self.floor = self.mjcf_model.worldbody.add('geom',name='floor',type='plane',pos=[0,0,0],conaffinity=1, size=[100, 100, .2],material='grid')
    

    #<body name="torso" pos="0 0 1.5" childclass="body">

    # <body mocap='true' name="m12" pos="1 0 0" >
    #   <geom type='sphere' size="0.01" material="mocap_material"/>
    # </body>
    # nominal_torso_h ~ 0.585
    # nominal_leg_length ~ 0.45
    thigh_length =  leg_scales['left_leg']['thigh_h_scale']*0.34 

    shin_length =  leg_scales['right_leg']['shin_h_scale']*0.3
                       
    foot_radius =  leg_scales['right_leg']['foot_r_scale']*.027 
    

    initial_torso_height =  ground_clearence + \
                            thigh_length + \
                            shin_length + \
                            foot_radius
                            # com_radius # cox it is in the same vertical height of the com
    
    self.torso = self.mjcf_model.worldbody.add('body', name='torso',pos=[0,0,initial_torso_height])#pos= base_pos)#[0, -.1 ,-.04])
    #   <light name="top" pos="0 0 2" mode="trackcom"/>
    self.torso.add('light',name='top',pos=[0, 0, 2],mode='trackcom')
    #   <camera name="back" pos="-3 0 1" xyaxes="0 -1 0 1 0 2" mode="trackcom"/>
    self.torso.add('camera',name='back',pos=[-3, 0, 1],xyaxes=[0, -1, 0, 1, 0, 2] ,mode='trackcom')
    #   <camera name="side" pos="0 -3 1" xyaxes="1 0 0 0 1 2" mode="trackcom"/>
    self.torso.add('camera',name='side',pos=[0, -3, 1],xyaxes=[1, 0, 0, 0, 1, 2] ,mode='trackcom')
    #   <freejoint name="root"/>
    self.torso.add('freejoint',name='root')
    
    self.torso.add('geom', name='head',pos=[0, 0, 0], type='sphere', size=[com_radius])  
    self.torso.add(
        'site', name='com',type="sphere", rgba="1. 0. 0. 1.",size=[0.01], 
        pos=[0.,0.,0.]
            ) 
    self.torso.add('camera',name='egocentric',pos=[.09, 0, 0],xyaxes=[0, -1, 0, 0.1, 1, 2] ,fovy="80")


    link_name = 'torso'
    link_radius = torso_h_scale*.09
    y1 = -.07
    y2 =  .07
    alpha = (torso_b_scale - 1) * (y2 - y1) * 0.5 - link_radius if torso_b_scale!= 1 else 0

    link_length = y2+alpha - (y1-alpha) 

    # self.torso.add('geom', name='butt', type='capsule',
    #                 fromto=[-.02, y1-alpha, 0, -.02, y2+alpha, 0], size=[link_radius])
    
    for t_m in marker_pos_params[link_name]:
      
      r = torso_h_scale*marker_pos_params[link_name][t_m]['r_nominal'] if 'r_nominal' in marker_pos_params[link_name][t_m].keys() else link_radius
      theta = np.radians(marker_pos_params[link_name][t_m]['theta'])
      k = marker_pos_params[link_name][t_m]['k']
      
      self.torso.add(
          'site', name=t_m,type="sphere", rgba="1. 0. 0. 1.",size=[0.01], 
          pos=[
              r*np.cos(theta) - 0.02, # pelvis is asymetrical
              k*0.5*link_length ,
              r*np.sin(theta), 
              ]
              )


    left_leg_site = self.torso.add(
        'site', name='left_leg_site',size=[1e-6]*3, 
        # pos=[0, 
        # torso_b_scale*.1 ,
        # torso_h_scale*-.04]

        pos=[0, 
        0.1+ (alpha if torso_b_scale!= 1 else 0),
        0]

        )
    
    right_leg_site = self.torso.add(
        'site', name='right_leg_site',size=[1e-6]*3, 
        # pos=[0, 
        # torso_b_scale*-.1 ,
        # torso_h_scale*-.04]

        pos=[0, 
        -0.1 - (alpha if torso_b_scale!= 1 else 0) ,
        0]


        )
    
    self.left_leg = Leg(name='left_leg',
                        knee_actuation = knee_actuation,
                        marker_pos_params = marker_pos_params['left_leg'],

                        **leg_scales['left_leg']
                        )
    left_leg_site.attach(self.left_leg.mjcf_model)

    self.right_leg = Leg(name='right_leg',
                        knee_actuation = knee_actuation,
                        marker_pos_params = marker_pos_params['right_leg'],
                        **leg_scales['right_leg']

                        )
    right_leg_site.attach(self.right_leg.mjcf_model)

    for i in range(40):
      self.marker = self.mjcf_model.worldbody.add('body', name='m'+str(i),mocap=True,pos=[0,0,0])
      self.marker.add('geom',name='m'+str(i),type='sphere', size=[0.01],rgba=[0.,1.,0.,1.0],mass=0.)



    self.mjcf_model.equality.add("weld",name='world_root',active='False',body1='torso',relpose=[0., 0., -2., 1., 0, 0, 0,])

if __name__ == '__main__':

  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument('--conf_xml_filename',help='common filename of xml and conf',default='default_pm_mll',type=str)

  args = parser.parse_args()  

  assets_path = './gym_hmm_ec/envs/assets/'
  conf_file_name = args.conf_xml_filename+'.yaml'

  # load the nominal default conf pre-saved
  config_file = open(assets_path+"models/model_confs/"+ conf_file_name,'r+')
  model_conf = yaml.load(config_file, Loader=yaml.FullLoader)
  print(conf_file_name)
  body = Pm_mll(
                  **model_conf
                  )
  physics = mjcf.Physics.from_mjcf_model(body.mjcf_model)

  mjcf.export_with_assets(body.mjcf_model,"./gym_hmm_ec/envs/assets/models",args.conf_xml_filename+'.xml')