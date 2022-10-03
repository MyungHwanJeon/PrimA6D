<h2 align="center">
  ** Source code will be available soon after review process **
</h2>

# PrimA6D

<div align="left">  
  <a href="https://myunghwanjeon.github.io">Myung-Hwan Jeon</a>,  
  <a href="https://scholar.google.co.kr/citations?user=vW2JtFAAAAAJ">Jeongyun Kim</a> and
  <a href="https://ayoungk.github.io/">Ayoung Kim</a> at <a href="https://rpm.snu.ac.kr">RPM Robotics Lab</a>
</div>

## Note
- Our study, ***PrimA6D***, is accepted for RA-L.
  - [Paper](https://arxiv.org/abs/2006.07789), [Video](https://youtu.be/HbNmsmTLRmk)
- The extended version, ***PrimA6D++***, is under review.
  - [Paper](), [Video](https://youtu.be/akbI61jUJgY)

<hr/>

## What Is PrimA6D?
 - ***PrimA6D (RA-L 2020)***
    - PrimA6D reconstructs the rotation primitive and its associated keypoints corresponding to the target object for enhancing the orientation inference.
    <div align="center">
      <a href="https://www.youtube.com/watch?v=HbNmsmTLRmk"><img src="assets/prima6d.png" width="75%" alt="IMAGE ALT TEXT"></a>
    </div>
    
    - More details in [PrimA6D: Rotational Primitive Reconstruction for Enhanced and Robust 6D Pose Estimation](https://arxiv.org/abs/2006.07789)

 - ***PrimA6D++ (Under Review)***
   - PrimA6D++ estimates three rotation axis primitive images and their associated uncertainties.   
   
    <div align="center">
      <a href="https://www.youtube.com/watch?v=akbI61jUJgY"><img src="assets/prima6d++_1.gif" width="49%" alt="IMAGE ALT TEXT"></a>
      <a href="https://www.youtube.com/watch?v=akbI61jUJgY"><img src="assets/prima6d++_3.gif" width="49%" alt="IMAGE ALT TEXT"></a>
    </div>
    
   - With estimated uncertainties, PrimA6D++ handles object ambiguity without prior information on object shape.
   
    <div align="center">
      <a href="https://www.youtube.com/watch?v=akbI61jUJgY"><img src="assets/prima6d++_2.gif" width="49%" alt="IMAGE ALT TEXT"></a>
      <a href="https://www.youtube.com/watch?v=akbI61jUJgY"><img src="assets/prima6d++_4.gif" width="49%" alt="IMAGE ALT TEXT"></a>
    </div>
    
   - More details in [Ambiguity-Aware Multi-Object Pose Optimization for Visually-Assisted Robot Manipulation]()

 - ***Object-SLAM for Multi-Object Pose Optimization (Under Review)***
   - Leveraging the uncertainty, we formulate the problem as an object-SLAM to optimize multi-object poses.
   
    <div align="center">
      <a href="https://www.youtube.com/watch?v=akbI61jUJgY"><img src="assets/slam.gif" width="75%" alt="IMAGE ALT TEXT"></a>      
    </div>   
   
   - More details in [Ambiguity-Aware Multi-Object Pose Optimization for Visually-Assisted Robot Manipulation]()

<hr/>

# How To Use: 6D Object Pose Estimation

## Environment Setup

 - ***Download Repo***   
   ````shell
   $ git clone git@github.com:rpmsnu/PrimA6D.git
   ````

 - ***Docker Image Download & Run***
   ````shell
   $ docker pull jmong1994/jeon:prima6d_new

   $ xhost +local:docker
   $ docker run --gpus all -it --env="DISPLAY" --net=host --ipc=host --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" -v /:/mydata  

   $ export PrimA6D_path=/path/to/PrimA6D
   ````
   We provide a docker image with an environment setup. You can download this docker image on the docker hub.
 
 - ***3D Model Download***
   ````shell
   $ ~~~~~~          
   ```` 
 
 - ***Dataset Download (only for training)***
   - Download ***Sun2012Pascal*** and ***BOP*** dataset
   ````shell
   $ cd $pose_estimation_path/dataset/raw_dataset
   $ bash get_sun2012pascalformat.sh
   $ cd bop          
   $ bash get_bop_ycbv.sh          
   ````

   - Prepare data
   ````shell
   $ cd $pose_estimation_path/dataset/YCB
   $ python3 YCB_train_synthetic.py -o=[obj_id]
   $ python3 YCB_train_pbr.py -o=[obj_id]    
   $ python3 YCB_test.py -o=[obj_id]         
   ````     
   For example, to prepare the No.1 of YCB object, `python3 YCB_train_synthetic.py -o=1`      

## How To Use: PrimA6D
  
  All commands are executed in the docker environment.
   
 - ***Inference***   
   ````shell
   $ cd $PrimA6D_path/Pose-Estimation/PrimA6D
   $ python3 4_test_all.py -o=[obj_id]         
   ````            
   For example, to infer the No.1 of YCB object, `python3 4_test_all.py -o=1`    
   
   For the corresponding object,   
   Download [PrimA6D](https://drive.google.com/drive/folders/1SGk-zpoSRXB2-bMpKy4YBKnvaL-NB2Wr?usp=sharing) weights and extract this weights to `$PrimA6D_path/Pose-Estimation/PrimA6D/trained_weight`.   
   Download [Segmentation](https://drive.google.com/drive/folders/1urfMR9EffdMRK-H-xjmvGQ3o7B9qmPDj?usp=sharing) weights and extract this weights to `$PrimA6D_path/Pose-Estimation/Segmentation/trained_weight`.         
   
 - ***Train*** 
   ***TBA***
   

## How To Use: PrimA6D++

  All commands are executed in the docker environment.
   
 - ***Inference***   
   ````shell
   $ cd $PrimA6D_path/Pose-Estimation/PrimA6D++
   $ python3 test_prima6d.py -o=[obj_id]         
   ````            
   For example, to infer the No.1 of YCB object, `python3 test_prima6d.py -o=1`    
   
   For the corresponding object,   
   Download [PrimA6D++](https://drive.google.com/drive/folders/1ZU790_b8nKZ8UttLl-iHuXC3jcgQ3wS8?usp=sharing) weights and extract this weights to `$PrimA6D_path/Pose-Estimation/PrimA6D/trained_weight`.   
   Download [Segmentation](https://drive.google.com/drive/folders/1urfMR9EffdMRK-H-xjmvGQ3o7B9qmPDj?usp=sharing) weights and extract this weights to `$PrimA6D_path/Pose-Estimation/Segmentation/trained_weight`.      

 - ***Train*** 
   ***TBA***
   
<hr/>

# How To Use: Object-SLAM for Multi-Object Pose Optimization  

## Citation

Please consider citing the paper as:
```
@ARTICLE{jeon-2020-prima6d,
author={Jeon, Myung-Hwan and Kim, Ayoung},
journal={IEEE Robotics and Automation Letters}, 
title={PrimA6D: Rotational Primitive Reconstruction for Enhanced and Robust 6D Pose Estimation}, 
year={2020},
volume={5},
number={3},
pages={4955-4962},
doi={10.1109/LRA.2020.3004322}}

@ARTICLE{jeon-2022-prima6d,
title={Ambiguity-Aware Multi-Object Pose Optimization for Visually-Assisted Robot Manipulation},
author={Myung-Hwan Jeon and Jeongyun Kim and Ayoung Kim},
journal={Under Review}}
```

## Contact
If you have any questions, contact here please
```
myunghwan.jeon@kaist.ac.kr
```
