# FRCT: Leveraging Foveated Vision and  Relative Position Invariance For Bimanual Manipulation

[![Code style](https://img.shields.io/badge/code%20style-black-black)](https://black.readthedocs.io/en/stable/)

FRCT(Foveated-Relative Coordination Transformer) is a unified architecture enabling dynamic bimanual coordination. Inspired by human visuomotor behavior, we categorize robotic arms as Foveated arm and Relative arm(Non-Foveated arm). Foveated arm occupies the visual focus region of model, requiring precise object detection and localization. Relative arm Operates in peripheral vision, with actions predicted in the coordinate system represented by the pose of Foveated arm. This framework leverages the invariant relative positioning between arms to substantially reduce coordination complexity. Context-dependent focus switching is achieved through a classification predictor that dynamically assigns Foveated/Relative roles to left/right arms. Across RLBench2's benchmark encompassing synchronous, asynchronous, and long-horizon tasks, FRCT achieves an average success rate exceeding state-of-the-art methods by at least $40\%$. These results demonstrate FRCT's efficacy in complex bimanual manipulation.

The repository and documentation are still work in progress.

For the latest updates, see: 


## Installation

We provide a [Dockerfile](Dockerfile) for Docker environments.
Please see [Installation](INSTALLATION.md) for further details.

### Prerequisites

The code FRCT is built-off the [PerAct2](https://bimanual.github.io/) and [RVT](https://bimanual.github.io/). The prerequisites are the same as PerAct2 and RVT.


#### 1. Environment


Install miniconda if not already present on the current system.
You can use `scripts/install_conda.sh` for this step:

```bash

sudo apt install curl 

curl -L -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh 
./Miniconda3-latest-Linux-x86_64.sh

SHELL_NAME=`basename $SHELL`
eval "$($HOME/miniconda3/bin/conda shell.${SHELL_NAME} hook)"
conda init ${SHELL_NAME}
conda install mamba -c conda-forge
conda config --set auto_activate_base false
```

Next, create the rlbench environment and install the dependencies

```bash
conda create -n FRCT python=3.8
conda activate FRCT
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```


#### 2. Dependencies

You need to setup  [RLBench](https://github.com/markusgrotz/rlbench/), [Pyrep](https://github.com/markusgrotz/Pyrep/), and [YARR](https://github.com/markusgrotz/YARR/).
Please note that due to the bimanual functionallity the main repository does not work.
You can use `scripts/install_dependencies.sh` to do so.
See [Installation](INSTALLATION.md) for details.

```bash
./scripts/install_dependencies.sh
```



### Pre-Generated Datasets


Please checkout the website for [pre-generated RLBench
demonstrations](https://bimanual.github.io). If you directly use these
datasets, you don't need to run `tools/bimanual_data_generator.py` from
RLBench. Using these datasets will also help reproducibility since each scene
is randomly sampled in `data_generator_bimanual.py`.

### Training


#### Single-GPU Training

To configure and train the model, follow these guidelines:

- **General Parameters**: You can find and modify general parameters in the `conf/config_frct.yaml` file. This file contains overall settings for the training environment, such as the number of cameras or the the tasks to use.

- **Method-Specific Parameters**: For parameters specific to each method, refer to the corresponding files located in the `conf/method` directory. These files define configurations tailored to each method's requirements.



When training adjust the `replay.batch_size` parameter to maximize the utilization of your GPU resources. Increasing this value can improve training efficiency based on the capacity of your available hardware.
You can either modify the config files directly or you can pass parameters directly through the command line when running the training script. This allows for quick adjustments without editing configuration files:

```bash
python train_frct.py replay.batch_size=64
```

In this example, the command sets replay.batch_size to 64 and specifies the use of the FRCT method for training.
Another important parameter to specify the tasks is `rlbench.task_name`, which sets the overall task, and `rlbench.tasks`, which is a list of tasks used for training. Note that these can be different for evaluation.
A complete set of tasks is shown below:

```yaml

rlbench:
  task_name: multi
  tasks:
  - bimanual_push_box
  - bimanual_lift_ball
  - bimanual_pick_plate
  - bimanual_put_item_in_drawer
  - bimanual_put_bottle_in_fridge
  - bimanual_pick_laptop
  - bimanual_sweep_to_dustpan
  - bimanual_lift_tray
  - bimanual_handover_item_easy
  - bimanual_take_tray_out_of_oven
```


#### Multi-GPU and Multi-Node Training

This repository supports multi-GPU training and distributed training across multiple nodes using [PyTorch Distributed Data Parallel (DDP)](https://pytorch.org/docs/stable/notes/ddp.html). 
Follow the instructions below to configure and run training across multiple GPUs and nodes.

#### Multi-GPU Training on a Single Node

To train using multiple GPUs on a single node, set the parameter `ddp.num_devices` to the number of GPUs available. For example, if you have 4 GPUs, you can start the training process as follows:

```bash
python train_frct.py replay.batch_size=64 ddp.num_devices=4
```

This command will utilize 4 GPUs on the current node for training. Remember to set the `replay.batch_size`, which is per GPU.

#### Multi-Node Training Across Different Nodes

If you want to perform distributed training across multiple nodes, you need to set additional parameters: ddp.master_addr and ddp.master_port. These parameters should be configured as follows:

`ddp.master_addr`: The IP address of the master node (usually the node where the training is initiated).
`ddp.master_port`: A port number to be used for communication across nodes.

Example Command:

```bash
python train_frct.py replay.batch_size=64 ddp.num_devices=4 ddp.master_addr=192.168.1.1 ddp.master_port=29500
```

Note: Ensure that all nodes can communicate with each other through the specified IP and port, and that they have the same codebase, data access, and configurations for a successful distributed training run.



### Evaluation


Similar to training you can find general parameters in  `conf/eval_frct.yaml` and method specific parameters in the `conf/method` directory.
For each method, you have to set the execution mode in RLBench. For bimanual agents such as `FRCT` or `BIMANUAL_PERACT` this is:

```yaml
rlbench:
  gripper_mode: 'BimanualDiscrete'
  arm_action_mode: 'BimanualEndEffectorPoseViaPlanning'
  action_mode: 'BimanualMoveArmThenGripper'
```


To generate videos of the current evaluation you can set `cinematic_recorder.enabled` to `True`.
It is recommended during evalution to disable the recorder, i.e. `cinematic_recorder.enabled=False`, as rendering the video increases the total evaluation time.

Example Command:
```bash
python eval_frct.py 
```
#### Running on a Headless Computer
```bash
Xvfb :611 -screen 0 1024x768x16 &
python DISPLAY=:611 eval_frct.py 
```
## Checkpoints
Release the [checkpoints](https://huggingface.co/navelorange/FRCT/tree/main).
## Acknowledgements

We sincerely thank the authors of the following repositories for sharing their code.

[PerAct2](https://github.com/markusgrotz/peract_bimanual)   
[RVT](https://github.com/nvlabs/rvt)     
[Pytorch-lamb](https://github.com/cybertronai/pytorch-lamb)   
[CLIP](https://github.com/openai/CLIP)  

## Licenses
This repository is released under the Apache 2.0 license.



## Citations 
If you find this repository helpful, please consider citing:
