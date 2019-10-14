import os
import os.path as path
import shutil
from send2trash import send2trash

dry = False

if __name__ == "__main__":

    modeldir = os.path.abspath("./modeldir_video")
    logdir = os.path.abspath("./logdir_video")

    # list subdirs
    modelruns = [d for d in os.listdir(modeldir) if d.startswith('run')]
    logruns = [d for d in os.listdir(logdir) if d.startswith('run')]
    bothruns = [d for d in modelruns if d in logruns] #intersection
    onlymodelruns = [d for d in modelruns if d not in bothruns]
    onlylogruns = [d for d in logruns if d not in bothruns]
    print('bothruns:', bothruns)
    print('onlymodelruns:', onlymodelruns)
    print('onlylogruns', onlylogruns)

    # delete folders w
    for d in onlymodelruns:
        print('Delete', path.join(modeldir, d))
        if not dry:
            send2trash(path.join(modeldir, d))
            #shutil.rmtree(path.join(modeldir, d))
    for d in onlylogruns:
        print('Delete', path.join(logdir, d))
        if not dry:
            send2trash(path.join(logdir, d))
            #shutil.rmtree(path.join(logdir, d))

    # delete runs without results
    minEpoch = 2
    for d in bothruns:
        numEpoch = len(os.listdir(path.join(modeldir, d))) - 1
        print('Run',d,'contains',numEpoch,'epochs')
        if numEpoch < minEpoch:
            print('Delete', path.join(modeldir, d))
            if not dry:
                send2trash(path.join(modeldir, d))
                #shutil.rmtree(path.join(modeldir, d))

            print('Delete', path.join(logdir, d))
            if not dry:
                send2trash(path.join(logdir, d))
                #shutil.rmtree(path.join(logdir, d))