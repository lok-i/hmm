from lxml.etree import XMLParser, parse, ElementTree, Element, SubElement
from OpenGL.GL import *
from geometry import Capsule, Sphere, Box, renderer
from utils import *
import numpy as np
import yaml

class Body:

    def __init__(self, node, parent_bone):
        self.geoms = []
        self.child = []
        self.node = node
        self.symm_bone = None
        self.is_picked = False
        self.picked_geom = None
        self.parent = parent_bone
        if parent_bone is not None:
            parent_bone.child.append(self)
        self.name = node.attrib['name']
        # self.ep = np.array([0,0,0])#np.fromstring(node.attrib['user'], sep=' ')

        self.body_w_pos = np.zeros(3)
        self.body_l_pos = np.zeros(3)
        if parent_bone is not None:
            self.body_w_pos += parent_bone.body_w_pos 
        if 'pos' in node.attrib.keys():
            self.body_w_pos += np.fromstring(node.attrib['pos'], sep=' ')
            self.body_l_pos = np.fromstring(node.attrib['pos'], sep=' ')
        if self.name == 'torso':
            ground_clearence = 0.25
            self.body_w_pos[2] -= ground_clearence
        
        # dummy, to be removed
        self.ep = np.array([0,0,0])
        self.sp = np.array([0,0,0])
        self.mp = 0.5 * (self.sp + self.ep)

        # dummy, to be removed


        if 'ffp' not in self.name:
            print('\nbody:',self.name)
            print('\tsites:')
            # add sites aswell as geoms
            for site_node in node.findall('site'):
                if 'type' in site_node.attrib.keys():
                    print('\t\t',site_node.attrib['name'])
                    site_type = site_node.attrib['type']
                    if site_type == 'sphere':
                        body_marker = Sphere.from_node(site_node)
                        body_marker.is_site = True
                        body_marker.body_w_pos = self.body_w_pos
                        body_marker.name = site_node.attrib['name']
                        self.geoms.append(body_marker)                        
                        # self.geoms[-1].is_site = True
            
            print('\tgeoms:')
            # add geometries
            for geom_node in node.findall('geom'):
                print('\t\t',geom_node.attrib['name'])
                geom_type = geom_node.attrib['type']
                if geom_type == 'capsule':
                    self.geoms.append(Capsule.from_node(geom_node))
                elif geom_type == 'sphere':
                    self.geoms.append(Sphere.from_node(geom_node))
                elif geom_type == 'box':
                    self.geoms.append(Box.from_node(geom_node))
                # print(self.geoms)
                if len(self.geoms)!=0:
                    self.geoms[-1].bone = self
        
    def __str__(self):
        return self.name

    def render(self, render_options):
        if render_options['render_bone']:
            color = [1.0, 0.0, 0.0] if self.is_picked else [0.8, 0.8, 0.8]
            glColor3d(*color)
            renderer.render_point(self.ep, 0.022)
            renderer.render_capsule(self.sp, self.ep, 0.02)
            # renderer.render_capsule(np.array([0,0,0]), np.array([0,0,0.5]), 0.02)

        if render_options['render_geom']:
            for geom in self.geoms:
                if geom.type != 'sphere':
                    color = [0.0, 1.0, 0.0,0.5] if geom == self.picked_geom else [1.0, 0.65, 0.0,0.5]
                    # glColor3d(*color)
                    glColor4d(*color)

                else:
                    color = [0.0, 1.0, 0.0] if geom == self.picked_geom else [0., 0., 1.0]
                    glColor3d(*color)

                # print(self.name,geom.type)
                geom.render(local_origin=self.body_w_pos)

    def pick(self, ray):
        for geom in self.geoms:
            # print(self.name,geom.type)
            res = geom.pick(ray,self.body_w_pos)
            if res:
                self.picked_geom = geom
                self.is_picked = True
                return geom
        if ray.dist2seg(self.sp, self.ep) < 0.02:
            self.is_picked = True
        return None

    def sync_node(self, local_coord):
        for geom in self.geoms:
            geom.sync_node(local_coord)

        if self.name == 'root':
            return

        if local_coord:
            self.node.attrib['pos'] = '{:.4f} {:.4f} {:.4f}'.format(*(self.sp - self.parent.sp))
            self.node.attrib['user'] = '{:.4f} {:.4f} {:.4f}'.format(0, 0, 0)
            for j_node in self.node.findall('joint'):
                j_node.attrib['pos'] = '{:.4f} {:.4f} {:.4f}'.format(0, 0, 0)
        else:
            self.node.attrib['pos'] = '{:.4f} {:.4f} {:.4f}'.format(*self.body_l_pos)

        # self.node.attrib['user'] = '{:.4f} {:.4f} {:.4f}'.format(*self.ep)
        # self.node.attrib['pos'] = '{:.4f} {:.4f} {:.4f}'.format(*self.mp)
        # for j_node in self.node.findall('joint'):
        #     j_node.attrib['pos'] = '{:.4f} {:.4f} {:.4f}'.format(*self.sp)
        #     if self.name != 'root':
        #         j_node.attrib['armature'] = '0.01'
        #         j_node.attrib['stiffness'] = '1.0'
        #         j_node.attrib['damping'] = '5.0'

    def delete_geom(self):
        symm_geom = self.picked_geom.symm_geom
        self.node.remove(self.picked_geom.node)
        self.geoms.remove(self.picked_geom)
        if self.symm_bone is not None:
            self.symm_bone.node.remove(symm_geom.node)
            self.symm_bone.geoms.remove(symm_geom)
        self.picked_geom = None

    def add_geom(self, geom_type='capsule', bone_capsule=False, clone_picked=False):
        geom = symm_geom = None
        if clone_picked:
            geom = self.picked_geom.clone()
            symm_geom = self.picked_geom.clone()
        elif geom_type == 'capsule':
            if bone_capsule:
                p1 = self.sp.copy()
                p2 = self.ep.copy()
            else:
                p1 = self.mp.copy()
                p2 = self.mp.copy()
                p1[0] -= 0.03
                p2[0] += 0.03
            geom = Capsule(p1, p2, 0.025)
            if self.symm_bone is not None:
                symm_geom = Capsule(p1, p2, 0.025)
        elif geom_type == 'sphere':
            geom = Sphere(self.sp, 0.04)
            if self.symm_bone is not None:
                symm_geom = Sphere(self.sp, 0.04)
        elif geom_type == 'box':
            geom = Box(self.mp, np.ones(3, ) * 0.04)
            if self.symm_bone is not None:
                symm_geom = Box(self.mp, np.ones(3, ) * 0.04)

        if geom is None:
            return
        self.node.insert(0, geom.node)
        self.geoms.append(geom)
        if self.symm_bone is not None:
            self.symm_bone.node.insert(0, symm_geom.node)
            self.symm_bone.geoms.append(symm_geom)
            symm_geom.symm_geom = geom
            geom.symm_geom = symm_geom
            geom.sync_symm()
        self.picked_geom = geom
        return geom

    def sync_symm(self):
        self.symm_bone.sp = self.sp.copy()
        self.symm_bone.mp = self.mp.copy()
        self.symm_bone.ep = self.ep.copy()
        self.symm_bone.sp[0] *= -1
        self.symm_bone.mp[0] *= -1
        self.symm_bone.ep[0] *= -1
        for geom in self.geoms:
            geom.sync_symm()


class Skeleton:

    def __init__(self, xml_file,static_marker_file):
        self.bones = []
        self.static_markers = []
        self.tree = None
        self.picked_geom = None
        self.picked_bone = None
        self.picked_static_marker = None
        self.load_from_xml(xml_file)

        marker_confpath = static_marker_file.split('processed_data/')[0]+'confs/' \
                            + static_marker_file.split('processed_data/')[-1].split('_from')[0]+'.yaml' 

        marker_config_file = open(marker_confpath,'r+')
        marker_conf = yaml.load(marker_config_file, Loader=yaml.FullLoader)
        self.marker_name2id = marker_conf['marker_name2id']

        self.load_static_markers(static_marker_file,static_confpath=marker_confpath)
        self.align_model_markers()

    def load_static_markers(self,static_marker_file,static_confpath):


        
        static_marker_pos = np.load(static_marker_file)['marker_positions']
        static_marker_pos = np.mean(static_marker_pos,axis=0)
        
        y_axis = static_marker_pos[:,1].copy()
        static_marker_pos[:,1] = static_marker_pos[:,0].copy()
        static_marker_pos[:,0] = y_axis
        
        x_mean = np.mean(static_marker_pos[:,0])
        static_marker_pos[:,0] = static_marker_pos[:,0] - x_mean 

        y_mean = np.mean(static_marker_pos[:,1])
        static_marker_pos[:,1] = static_marker_pos[:,1] - y_mean 

        for i,marker_pos in enumerate(static_marker_pos):
            
            experimental_marker = Sphere( pos=marker_pos, size=0.01)
            for name in self.marker_name2id.keys():
                if  self.marker_name2id[name] == i:
                    experimental_marker.name = name
            self.static_markers.append(experimental_marker)
        
    def load_from_xml(self, xml_file):
        parser = XMLParser(remove_blank_text=True)
        self.tree = parse(xml_file, parser=parser)
        
        root = self.tree.getroot().find('worldbody').find('body')
        
        self.add_bones(root, None)
        self.build_symm_by_name()
        self.make_symm()


    def align_model_markers(self):
        
        # extract model markers
        model_markers = []
        for bone in self.bones:
            for geom in bone.geoms:
                if geom.type == 'sphere':
                    if geom.is_site:
                        # print(geom.name)
                        model_markers.append(geom)
        
        max_steps = 1000
        error_norm = np.inf
        tolerance = 0.001
        prev_error_norm = 2*tolerance 
        dz = 0.002 
        
        # update static marker's z pos to align with the model
        for i in range(max_steps):
            if abs(error_norm) < tolerance or ( error_norm - prev_error_norm > 0 and i != 0) :
                # print(i)
                break
            else:
                prev_error_norm = error_norm
                error_norm = 0.
                for model_marker in model_markers:
                    name = model_marker.name.split('/')[-1] 
                    
                    if name in self.marker_name2id.keys():
                        error_norm += np.linalg.norm( (model_marker.body_w_pos + model_marker.pos ) - self.static_markers[self.marker_name2id[name]].pos)
                        self.static_markers[self.marker_name2id[name]].pos[2] += dz
            
            
                #print(name, error_norm,(error_norm - prev_error_norm))
            


    def save_to_xml(self, xml_file, local_coord=False):
        for bone in self.bones:
            bone.sync_node(local_coord)
        
        self.tree.write(xml_file, pretty_print=True)

    def add_bones(self, bone_node, parent_bone):
        bone = Body(bone_node, parent_bone)
        self.bones.append(bone)

        for bone_node_c in bone_node.findall('body'):
            self.add_bones(bone_node_c, bone)

    def build_symm_by_name(self):
        for bone in self.bones:
            if 'Left' in bone.name:
                symm_bone_name = bone.name.replace('Left', 'Right')
                for symm_bone in self.bones:
                    if symm_bone.name == symm_bone_name:
                        break
                bone.symm_bone = symm_bone
                symm_bone.symm_bone = bone
                for i, geom in enumerate(bone.geoms):
                    geom.symm_geom = symm_bone.geoms[i]
                    symm_bone.geoms[i].symm_geom = geom

    def build_symm(self):
        for bone_a in self.bones:
            for bone_b in self.bones:
                if bone_a.name == 'root' or bone_b.name == 'root' \
                        or bone_a.name[1:] == 'hipjoint' or bone_b.name[1:] == 'hipjoint':
                    continue
                bone_b_negep = bone_b.ep.copy()
                bone_b_negep[0] *= -1
                if bone_a != bone_b and np.linalg.norm(bone_a.ep - bone_b_negep) < 1e-4:
                    bone_a.symm_bone = bone_b
                    for i, geom in enumerate(bone_a.geoms):
                        geom.symm_geom = bone_b.geoms[i]

    def make_symm(self):
        for i, bone in enumerate(self.bones):
            if bone.symm_bone is not None and self.bones.index(bone.symm_bone) > i:
                bone.sync_symm()

    def render(self, render_options):
        for static_marker in self.static_markers:

            color = [0.0, 1.0, 0.0,1.0] if static_marker == self.picked_static_marker else [1.0, 0.0, 0.0,1.0]
            # glColor3d(*color)
            glColor4d(*color)

            static_marker.render()
        for bone in self.bones:
            bone.render(render_options)

    def pick_geom(self, ray):
        self.picked_geom = None
        self.picked_bone = None
        for bone in self.bones:
            bone.picked_geom = None
            bone.is_picked = False

        for bone in self.bones:
            res = bone.pick(ray)
            if bone.is_picked:
                self.picked_bone = bone
                if res is not None:
                    self.picked_geom = res
                return True
        return False

    def pick_target_marker(self, ray):
        self.picked_static_marker = None
        # for s_m in self.static_markers:
        #     s_m.is_picked = False

        for s_m in self.static_markers:
            res = s_m.pick(ray)

            if res:
                self.picked_static_marker = s_m
                return True
        return False


