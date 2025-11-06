
# **TERL: Large-Scale Multi-Target Encirclement Using Transformer-Enhanced Reinforcement Learning**  

## **Overview**  
This repository contains the code implementation for our paper:  [**TERL: Large-Scale Multi-Target Encirclement Using Transformer-Enhanced Reinforcement Learning**](https://arxiv.org/abs/2503.12395).
TERL (**Transformer-Enhanced Reinforcement Learning**) is a deep reinforcement learning framework designed for **large-scale multi-target encirclement** in multi-robot pursuit-evasion (PE) tasks. Unlike traditional RL-based approaches that primarily focus on **single-target pursuit**, TERL integrates a **transformer-based policy network** with an **adaptive target selection mechanism**, enabling efficient and coordinated multi-target encirclement, even in large-scale settings.  

A demonstration video is available [here](https://youtu.be/niYCRtdcDs0?si=G6p_a9j2fI6LzS4r).

![TERL Network Architecture.png](TERL%20Network%20Architecture.png)

If you find this repository useful in your research, please cite the following paper:
```bibtex
@article{zhang2025terl,
  title={TERL: Large-Scale Multi-Target Encirclement Using Transformer-Enhanced Reinforcement Learning},
  author={Zhang, Heng and Zhao, Guoxiang and Ren, Xiaoqiang},
  journal={arXiv preprint arXiv:2503.12395},
  year={2025}
}
```
## **Key Features**
- 🚀 **Transformer-Based Policy**: Utilizes **self-attention mechanisms** to enhance feature extraction and decision-making.  
- 🎯 **Adaptive Target Selection**: Dynamically prioritizes encirclement targets to improve coordination.  
- 🏆 **Superior Performance**: Achieves higher **encirclement success rates** and **faster task completion** compared to existing RL-based methods.  
- 🔄 **Scalability & Generalization**: Trained in small-scale scenarios **(15 pursuers, 4 targets)** and successfully generalized to large-scale environments **(80 pursuers, 20 targets)** with a **100% success rate**, without requiring retraining.  

---

## **Installation**  
### **Using Docker (Recommended)**  
A **Dockerfile** is provided to simplify environment setup. You can build and run the environment using the following commands:

#### **Build the Docker Image**  
```shell
docker build -t [image_name] .
```
You can specify additional arguments in the **Dockerfile** based on your hardware and preferences.

#### **Run the Docker Container**  
```shell
docker run -it --rm --gpus all --name [container_name] [image_name]
```

### **Manual Installation (Alternative)**  
If you prefer to set up the environment manually, install the required dependencies using:
```shell
pip install -r requirements.txt
```

---

## **Training Models**  
You can train the **TERL** model using the following command:

```shell
python train_rl_with_configs.py -D [device] -R [config_file_index] -I [model_index]
```
**Arguments**:  
- `-D [device]`: Specifies the device to run the training (e.g., `cuda:0`).  
- `-R [config_file_index]`: Chooses the configuration file index from the `configs` directory (e.g., `0`). `config_dqn.yaml`(1) is used for `DQN` training, and `config.yaml`(0) is used for other RL algorithms.
- `-I [model_index]`: Specifies the model index to be trained (e.g., `0`).  

Example:
```shell
python train_rl_with_configs.py -D cuda:0 -R 0 -I 0
```

---

## **Experimentation**  
We provide different experiment scripts for **ablation studies** and **baseline comparisons**:

```shell
python run_experiments_ablation.py / run_experiments_baseline.py -D [device] -C [experiment_config_file_index] [-S]
```
**Arguments**:  
- `-D [device]`: Specifies the device to run the experiments on (e.g., `cuda:0`).  
- `-C [experiment_config_file_index]`: Chooses the experiment configuration file index from the `configs` directory (e.g., `0`).  
- `-S`: (Optional) If specified, trajectory data will be saved.  

Example:
```shell
python run_experiments_ablation.py -D cuda:0 -C 0 -S
```

### **Experiment Configuration**
Experiment configurations are located in the `configs/` directory. Modify these files to customize experiments.  
Parameter `perception: max_evader_num` can be adjusted according to the number of evaders in the environment.

---

## **Project Structure**  
```
TERL/
│── TrainedModels/         # Pre-trained models and saved checkpoints
│── config/                # Configuration files for training & experiments
│── environment/           # Environment implementation for pursuit-evasion tasks
│── policy/                # RL policy network implementation
│── robots/                # Robot control and movement logic
│── thirdparty/            # Third-party dependencies (e.g., evader strategy)
│── utils/                 # Utility functions and helper scripts
│── visualization/         # Visualization scripts for experiments and results
│
│── .dockerignore          # Files to be ignored by Docker
│── .gitignore             # Files to be ignored by Git
│── Dockerfile             # Docker environment setup
│── README.md              # Project documentation
│── config_manager.py      # Manages configuration loading and saving
│── requirements.txt       # Required dependencies
│── run_experiments_ablation.py   # Script for running ablation experiments
│── run_experiments_baseline.py   # Script for running baseline experiments
│── train_rl_with_configs.py      # Script for training RL models with configurations

```

---


## **Contributing**  
We welcome contributions! Please follow these steps:  
1. **Fork** the repository 🍴  
2. Create a **new branch**: `git checkout -b feature-name`  
3. **Commit changes**: `git commit -m "Add new feature"`  
4. **Push** to your branch: `git push origin feature-name`  
5. Create a **pull request (PR)** 📩  

---

## **License**  
This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

---

## **Contact & Support**  
For questions, feedback, or collaboration opportunities, feel free to:  
📌 Open an **issue** in this repository  


## Updates
- Fixed a minor bug in `global_reward` (boundary penalty sign).  
  This had negligible effect on results, since boundary violations were rarely triggered.  

