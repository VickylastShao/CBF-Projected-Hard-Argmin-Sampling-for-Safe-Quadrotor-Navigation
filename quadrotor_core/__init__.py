# -*- coding: utf-8 -*-
"""
PTRM-NMPC 核心模块包

包含四轴飞行器动力学仿真、NMPC 专家求解器、TRM 神经网络、
在线概率推理预测器、数据集生成及联合训练的全部核心组件。
"""

from .dynamics import QuadrotorDynamics
from .nmpc_solver import GoldenNMPCSolver
from .trm_network import TRMNMPC, SimpleEncoderQHead
from .ptrm_predictor import PTRMNMPCPredictor
from .dataset import generate_quadrotor_dataset, generate_cl_trm_dataset, evaluate_batch_decoded_trajectory_cost
from .training import train_trm_jointly, train_trm_candidate_conditioned, train_simple_encoder_qhead
