from math import sqrt
from typing import Optional, Tuple, Union

import torch
from torch import Tensor
from torch.nn.parameter import Parameter

from torch_geometric.explain import ExplainerConfig, Explanation, ModelConfig
from torch_geometric.explain.algorithm import ExplainerAlgorithm
from torch_geometric.explain.algorithm.utils import clear_masks, set_masks
from torch_geometric.explain.config import MaskType, ModelMode, ModelTaskLevel

class GNNExplainer(ExplainerAlgorithm):
    coeffs = {
        'edge_size': 0.005,
        'edge_reduction': 'sum',
        'node_feat_size': 1.0,
        'node_feat_reduction': 'mean',
        'edge_ent': 1.0,
        'node_feat_ent': 0.1,
        'EPS': 1e-15,
    }

    def __init__(self, epochs=100, lr=0.01, **kwargs):
        super().__init__()
        self.epochs = epochs
        self.lr = lr
        self.coeffs.update(kwargs)
        self.node_mask = self.hard_node_mask = None
        self.edge_mask_bgnn = self.hard_edge_mask_bgnn = None
        self.edge_mask_tgnn = self.hard_edge_mask_tgnn = None

    def forward(self, model, x, edge_index_dict, target, index=None, **kwargs):
        self._train(model, x, edge_index_dict, target=target, index=index, **kwargs)
        node_mask = self._post_process_mask(self.node_mask, self.hard_node_mask, True)
        edge_mask_bgnn = self._post_process_mask(self.edge_mask_bgnn, self.hard_edge_mask_bgnn, True)
        edge_mask_tgnn = self._post_process_mask(self.edge_mask_tgnn, self.hard_edge_mask_tgnn, True)
        self._clean_model(model)
        return Explanation(node_mask=node_mask, edge_mask_bgnn=edge_mask_bgnn, edge_mask_tgnn=edge_mask_tgnn)

    def _train(self, model, x, edge_index_dict, target, index, **kwargs):
        self._initialize_masks(x, edge_index_dict)
        params = []
        if self.node_mask: params.append(self.node_mask)
        if self.edge_mask_bgnn: params.append(self.edge_mask_bgnn)
        if self.edge_mask_tgnn: params.append(self.edge_mask_tgnn)
        optimizer = torch.optim.Adam(params, lr=self.lr)
        
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            h = x * self.node_mask.sigmoid() if self.node_mask else x
            y_hat = model(h, edge_index_dict['tgnn'], p_edge_all=edge_index_dict['bgnn'],
                          edge_mask_bgnn=self.edge_mask_bgnn, edge_mask_tgnn=self.edge_mask_tgnn, **kwargs)
            loss = self._loss(y_hat, target if index is None else y_hat[index], target[index])
            loss.backward()
            optimizer.step()
            # 收集梯度信息
            if epoch == 0:
                self.hard_node_mask = self.node_mask.grad != 0 if self.node_mask else None
                self.hard_edge_mask_bgnn = self.edge_mask_bgnn.grad != 0 if self.edge_mask_bgnn else None
                self.hard_edge_mask_tgnn = self.edge_mask_tgnn.grad != 0 if self.edge_mask_tgnn else None

    def _initialize_masks(self, x, edge_index_dict):
        N, F = x.size()
        E_bgnn = edge_index_dict['bgnn'].size(1)
        E_tgnn = edge_index_dict['tgnn'].size(1)
        device = x.device
        
        # 初始化node mask
        if self.explainer_config.node_mask_type == 'object':
            self.node_mask = Parameter(torch.randn(N, 1, device=device) * 0.1)
        
        # 初始化两个edge mask
        if self.explainer_config.edge_mask_type == 'object':
            std = torch.nn.init.calculate_gain('relu') * sqrt(2.0 / (2 * N))
            self.edge_mask_bgnn = Parameter(torch.randn(E_bgnn, device=device) * std)
            self.edge_mask_tgnn = Parameter(torch.randn(E_tgnn, device=device) * std)

    def _loss(self, y_hat, y):
        # 基础损失计算
        loss = ...  # 根据任务类型计算
        
        # Edge masks正则化
        if self.edge_mask_bgnn is not None:
            m = self.edge_mask_bgnn.sigmoid()
            loss += self.coeffs['edge_size'] * m.sum()
            loss += self.coeffs['edge_ent'] * (-m * torch.log(m + 1e-15)).mean()
        
        if self.edge_mask_tgnn is not None:
            m = self.edge_mask_tgnn.sigmoid()
            loss += self.coeffs['edge_size'] * m.sum()
            loss += self.coeffs['edge_ent'] * (-m * torch.log(m + 1e-15)).mean()
        
        # Node mask正则化
        if self.node_mask is not None:
            m = self.node_mask.sigmoid()
            loss += self.coeffs['node_feat_size'] * m.mean()
            loss += self.coeffs['node_feat_ent'] * (-m * torch.log(m + 1e-15)).mean()
        
        return loss
    

class GNNExplainer_old():
    r"""Deprecated version for :class:`GNNExplainer`."""

    coeffs = GNNExplainer.coeffs

    conversion_node_mask_type = {
        'feature': 'common_attributes',
        'individual_feature': 'attributes',
        'scalar': 'object',
    }

    conversion_return_type = {
        'log_prob': 'log_probs',
        'prob': 'probs',
        'raw': 'raw',
        'regression': 'raw',
    }

    def __init__(
        self,
        model: torch.nn.Module,
        epochs: int = 100,
        lr: float = 0.01,
        return_type: str = 'log_prob',
        feat_mask_type: str = 'feature',
        allow_edge_mask: bool = True,
        **kwargs,
    ):
        assert feat_mask_type in ['feature', 'individual_feature', 'scalar']

        explainer_config = ExplainerConfig(
            explanation_type='model',
            node_mask_type=self.conversion_node_mask_type[feat_mask_type],
            edge_mask_type=MaskType.object if allow_edge_mask else None,
        )
        model_config = ModelConfig(
            mode='regression'
            if return_type == 'regression' else 'multiclass_classification',
            task_level=ModelTaskLevel.node,
            return_type=self.conversion_return_type[return_type],
        )

        self.model = model
        self._explainer = GNNExplainer(epochs=epochs, lr=lr, **kwargs)
        self._explainer.connect(explainer_config, model_config)

    @torch.no_grad()
    def get_initial_prediction(self, *args, **kwargs) -> Tensor:

        training = self.model.training
        self.model.eval()

        out = self.model(*args, **kwargs)
        if (self._explainer.model_config.mode ==
                ModelMode.multiclass_classification):
            out = out.argmax(dim=-1)

        self.model.train(training)

        return out

    def explain_graph(
        self,
        x: Tensor,
        edge_index: Tensor,
        **kwargs,
    ) -> Tuple[Tensor, Tensor]:
        self._explainer.model_config.task_level = ModelTaskLevel.graph

        explanation = self._explainer(
            self.model,
            x,
            edge_index,
            target=self.get_initial_prediction(x, edge_index, **kwargs),
            **kwargs,
        )
        return self._convert_output(explanation, edge_index)

    def explain_node(
        self,
        node_idx: int,
        x: Tensor,
        edge_index: Tensor,
        **kwargs,
    ) -> Tuple[Tensor, Tensor]:
        self._explainer.model_config.task_level = ModelTaskLevel.node
        explanation = self._explainer(
            self.model,
            x,
            edge_index,
            target=self.get_initial_prediction(x, edge_index, **kwargs),
            index=node_idx,
            **kwargs,
        )
        return self._convert_output(explanation, edge_index, index=node_idx,
                                    x=x)

    def _convert_output(self, explanation, edge_index, index=None, x=None):
        node_mask = explanation.get('node_mask')
        edge_mask = explanation.get('edge_mask')

        if node_mask is not None:
            node_mask_type = self._explainer.explainer_config.node_mask_type
            if node_mask_type in {MaskType.object, MaskType.common_attributes}:
                node_mask = node_mask.view(-1)

        if edge_mask is None:
            if index is not None:
                _, edge_mask = self._explainer._get_hard_masks(
                    self.model, index, edge_index, num_nodes=x.size(0))
                edge_mask = edge_mask.to(x.dtype)
            else:
                edge_mask = torch.ones(edge_index.size(1),
                                       device=edge_index.device)

        return node_mask, edge_mask