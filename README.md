# Volumetric Isosurface Rendering with Deep Learning-Based Super-Resolution

This repository contains the code accompaning the TVCG submission.

## Requirements
The code was written under Windows 10 with Visual Studio and CUDA 10.0 and Python 3.6.
For the python requirements, see `SuperresolutioNetwork\Requirements.txt`.
The network will run (probably) platform-independently but the isosurface renderer is most likely fixed to Windows. Use it on other platforms on your own risk.

Note: for the sake of a smaller repository, we only added the third-party libraries for a release build and excluded the debug build.

## Project structure
The project contains the following sub-project
 - CopyLibraries: utility that copies dlls from the third-party folders to the binary folder
 - CPURenderer: cpu isosurface renderer, callable from command line or interactive over streans
 - GPURenderer: cuda isosurface renderer, callable from command line or interactive over streans
 - GPURendererDirect: cuda isosurface renderer, but build as a shared library to be included directly in Python
 - DataGenerator: python projects with scripts to generate the training data
 - SuperresolutionNetwork: the python project with the network specification, dataset generation and loading, training and interactive inference.
   All files prefixed with `main` are executable python scripts.
    - mainVideo.py: training code for the Shaded network
    - mainVideoUnshaded.py: training code for the unshaded networks (networks on the geometry/normals with shading in post-process)
    - mainDatasetViewer.py: simple GUI to view the dataset
    - mainGUI.py: interactive GUI to explore the datasets and different networks. Depends on GPURendererDirect
    - mainComparisonImages.py / mainComparisonVideo1,2,3.py: scripted benchmarks and video creation
    - mainPSNR1,2,3,4.py: scripts used to create the statistics reported in the paper

## How to use it
The `mainGUI.py` can be directly launched without command line arguments. It opens a window and allows you to select the volume to render and the networks to use.

`mainVideoUnshaded.py` is the main training code. All network and training parameters are specified via command line.
For example, these are the arguments used to train our best performing network:
```
python3 mainVideoUnshaded.py \
	--dataset cloud-video \
	--inputPathUnshaded "...../all_ejecta.txt" \
	--numberOfImages -1 \
	--model EnhanceNet \
	--losses l1:mask:1,l1:ao:1,l1:normal:10,l1:depth:10,temp-l2:color:0.1 \
	--lossAO 0.0 \
	--lossAmbient 0.1 \
	--lossDiffuse 0.9 \
	--initialImage zero \
	--samples 5000 \
	--batchSize 16 \
	--testBatchSize 16 \
	--nEpochs 1000 \
	--lr 0.0001 \
	--lrStep 100 \
	--logdir "./logdir_video_2/" \
	--modeldir "./modeldir_video_2/" \
	--pretrained "./pretrained/gen_l1normalDepth_2.pth" \
	--cuda
```

## Datasets and binary releases
Volume datasets, prerendered data for the training and pretrained networks can be found under `Releases`

## License
This software, excluding third party libraries, is distributed under the MIT open source license. See `LICENSE` for details.
