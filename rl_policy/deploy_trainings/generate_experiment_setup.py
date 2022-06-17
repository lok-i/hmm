import os
import re
import argparse
from itertools import product
from shutil import copyfile


def extract(chunk):
    chunk = chunk[len('<START>'):-len('<STOP>')]
    return chunk.split('<NEXT>')

def next_tag(string):
    tags = ['<start>', '<next>', '<stop>']
    locs = [(string.find(tag), tag) for tag in tags if string.find(tag) != -1]
    if len(locs) == 0:
        return None
    return min(locs)

def can_split(string):
    tags = ['<start>', '<next>', '<stop>']
    locs = [(string.find(tag), tag) for tag in tags if string.find(tag) != -1]
    return len(locs) > 0

def do_split(string):
    tags = ['<start>', '<next>', '<stop>']
    locs = [(string.find(tag), tag) for tag in tags if string.find(tag) != -1]
    tag = min(locs)
    return string[:tag[0]], tag[1], string[tag[0] + len(tag[1]):]

def cross(xs, ys):
    cross = []
    for x in xs:
        for y in ys:
            cross.append(x + y)
    return cross

def process(string):
    after = string
    processed = ['']
    while can_split(after):
        before, tag, after = do_split(after)
        processed = cross(processed, [before])
        assert tag == '<start>'
        depth = 0
        done = False
        strings = ['']
        while not done:
            before, tag, after = do_split(after)
            if tag == '<start>':
                strings[-1] += before + tag
                depth += 1
            elif tag == '<next>' and depth > 0:
                strings[-1] += before + tag
            elif tag == '<next>' and depth == 0:
                strings[-1] += before
                strings.append('')
            elif tag == '<stop>' and depth > 0:
                strings[-1] += before + tag
                depth -= 1
            elif tag == '<stop>' and depth == 0:
                strings[-1] += before
                done = True
            else:
                assert False
        additions = []
        for new_string in strings:
            additions += process(new_string)
        processed = cross(processed, additions)
    processed = cross(processed, [after])
    return processed

def generate_experimets(folder):
    with open(folder + 'set_conf.yaml', 'r') as f:
        conf = f.read()

    # print(conf)
    starts = [m.start() for m in re.finditer('<start>', conf)]
    print(starts)
    
    
    stops = [m.end() for m in re.finditer('<stop>', conf)]
    print(stops)

    potential_values = []
    for (start, stop) in zip(starts, stops):
        potential_values.append(extract(conf[start:stop]))
    
    print(potential_values)
    exp_num = 0
    for values in product(*potential_values):
        version = conf[0:starts[0]]
        for i in range(len(values)):
            if i > 0:
                version += conf[stops[i - 1]:starts[i]]
            version += values[i]
        version += conf[stops[-1]:]
        exp_folder = folder + str(exp_num).zfill(3)
        os.makedirs(exp_folder)
        with open(exp_folder + '/conf.yaml', 'w') as f:
            f.write(version)
        exp_num += 1

    print('Generated \x1b[0;34;40m ' + str(exp_num) + ' \x1b[0m experiments.')

def new_generate_experimets(folder,expName,job_path):
    
    copyfile(job_path+expName+'.yaml', folder + 'set_conf.yaml')
    
    with open(folder + 'set_conf.yaml', 'r') as f:
        conf = f.read()
    
    folder += '/'+expName+'/'
    processed = process(conf)
    exp_num = 0
    for p in processed:
        # print(p)
        exp_folder = folder + str(exp_num).zfill(3)
        os.makedirs(exp_folder)
        with open(exp_folder + '/conf.yaml', 'w') as f:
            f.write(p)
        exp_num += 1
    print('Generated \033[32m' + str(exp_num) + '\033[0m experiments.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--job_folderpath',help='path to the folder with the job confs', default='')
    parser.add_argument('--exp_path',help='path to the experment folder')
    parser.add_argument('--exp_name',help='name of the experment folder')

    # parser.add_argument('--start', type=int)
    # parser.add_argument('--stop', type=int)
    # parser.add_argument('--new_id', type=int)
    # parser.add_argument('--folder')
    args = parser.parse_args()
    # args.exp_path = './experiments/'+args.f+'/'
    if os.path.isdir(args.exp_path + args.exp_name ) == False:
        os.mkdir(args.exp_path + args.exp_name)
    new_generate_experimets( 
                            folder=args.exp_path,
                            expName=args.exp_name,
                            job_path = args.job_folderpath
                            )


