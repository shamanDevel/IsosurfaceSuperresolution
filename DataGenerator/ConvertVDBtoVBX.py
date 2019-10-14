import os
import os.path
import subprocess

renderer = '../bin/GPURenderer.exe'
inPath = '../../data/clouds/input/'
outPath = '../../data/clouds/inputVBX/'

def main():
    if not os.path.exists(outPath):
        os.mkdir(outPath)

    datasets = [file for file in os.listdir(inPath) if file.endswith('vdb')]
    print('Datasets found:', datasets)

    for file in datasets:
        inputFile = os.path.join(inPath, file)
        ouputFile = os.path.join(outPath, os.path.splitext(file)[0] + ".vbx")
        args = [
            renderer,
            '-m','convert',
            inputFile,
            ouputFile
            ]
        print(' '.join(args))
        subprocess.run(args, stdout=None, stderr=None, check=True)

if __name__ == "__main__":
    main()