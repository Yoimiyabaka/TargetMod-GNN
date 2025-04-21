import torch
import os
print("Current working directory:", os.getcwd())
import sys
sys.path.append(os.getcwd())

from gnn_models_sag import ppi_model  

ckpt_path = "result_save\gnn_2024-12-21-16-56-10-最好的\gnn_model_valid_best.ckpt"  
checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))


state_dict = checkpoint['state_dict']

print("GIN Model Layers:")
for key in state_dict.keys():
    if key.startswith("TGNN"):  
        print(f"{key}: {state_dict[key].shape}")
        

print("\nGCN Model Layers:")
for key in state_dict.keys():
    if key.startswith("BGNN"):  
        print(f"{key}: {state_dict[key].shape}")


print("TGNN (Top Graph Neural Network) Parameters:")
for key, value in state_dict.items():
    if key.startswith("TGNN"):
        print(f"{key}: First 5 values = {value.flatten()[:5].tolist()}")

print("\nBGNN (Bottom Graph Neural Network) Parameters:")
for key, value in state_dict.items():
    if key.startswith("BGNN"):
        print(f"{key}: First 5 values = {value.flatten()[:5].tolist()}")