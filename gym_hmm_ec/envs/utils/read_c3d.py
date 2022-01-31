import c3d


file_path = './gym_hmm_ec/envs/assets/our_data/Trial_1.c3d'
with open(file_path, 'rb') as handle:
    reader = c3d.Reader(handle)
    data = reader.read_frames()
    print( type(data) )
    for  data in reader.read_frames():

        print('Frame {}:{}\n{}'.format(data[0],data[1].shape,data[2][0]))
        
        
        if data[0] > 10:
            exit()