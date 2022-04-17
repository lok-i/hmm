import os.path as osp
import argparse
import pybullet_utils.bullet_client as bulllet_client
import pybullet_utils.urdfEditor as urdfEditor



def convert_mjcf_to_urdf(input_mjcf, output_path):
    """Convert MuJoCo mjcf to URDF format and save.
    Parameters
    ----------
    input_mjcf : str
        input path of mjcf file.
    output_path : str
        output directory path of urdf.
    """
    client = bulllet_client.BulletClient()
    objs = client.loadMJCF(
        input_mjcf, flags=client.URDF_USE_IMPLICIT_CYLINDER)

    for obj in objs:
        humanoid = objs[obj]
        ue = urdfEditor.UrdfEditor()
        ue.initializeFromBulletBody(humanoid, client._client)
        robot_name = str(client.getBodyInfo(obj)[1], 'utf-8')
        part_name = str(client.getBodyInfo(obj)[0], 'utf-8')
        
        save_visuals = False
        if 'torso' in part_name:
            outpath = osp.join(
                output_path, "{}_{}.urdf".format(robot_name, part_name))
            ue.saveUrdf(outpath, save_visuals)
            # print(outpath)
            
            file_read = open(outpath,'r')

            file2lines = file_read.readlines()
            # links = file2txt.split('</link>')
            for i in range(len(file2lines)):
                file2lines[i] = file2lines[i].replace('capsule','cylinder') 
            file_read.close()

            file_write = open(outpath,'w')

            for i in range(len(file2lines)):
                file_write.write(file2lines[i])
            
            file_write.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--input_mjcf',help='name of the preprocessed mocap file',default='AB1_Session1_Right6_Left6',type=str)

    # parser.add_argument('--plot_solns', help='whether to plot the ik solns',default=False, action='store_true')
    args = parser.parse_args()

    convert_mjcf_to_urdf(input_mjcf=args.input_mjcf,output_path='./urdfs')
