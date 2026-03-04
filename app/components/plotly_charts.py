"""
Plotly 交互式图表组件
=====================
可复用的图表绘制函数
"""

import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from typing import Dict, List, Optional


def plot_training_curve(history: Dict, metric: str = 'total', 
                        title: str = '训练曲线') -> go.Figure:
    """
    绘制训练损失曲线
    
    Args:
        history: 训练历史字典 {'step': [...], 'total': [...], ...}
        metric: 要绘制的指标名
        title: 图表标题
    """
    fig = go.Figure()
    
    if 'step' in history and metric in history:
        steps = history['step']
        values = history[metric]
        
        fig.add_trace(go.Scatter(
            x=steps, y=values,
            mode='lines',
            name=metric.upper(),
            line=dict(width=2, color='#3498DB')
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title='训练步数',
        yaxis_title=f'{metric.upper()} Loss',
        yaxis_type='log',
        template='plotly_white',
        hovermode='x unified',
        height=500,
    )
    
    return fig


def plot_multi_loss_curves(history: Dict, 
                           metrics: List[str] = None) -> go.Figure:
    """
    绘制多条损失曲线对比
    
    Args:
        history: 训练历史
        metrics: 指标列表，默认 ['total', 'pde', 'qg']
    """
    if metrics is None:
        metrics = ['total', 'pde', 'qg', 'ic', 'bc']
    
    fig = go.Figure()
    
    colors = {
        'total': '#E74C3C',
        'pde': '#3498DB',
        'qg': '#2ECC71',
        'ic': '#F39C12',
        'bc': '#9B59B6',
    }
    
    if 'step' in history:
        steps = history['step']
        for metric in metrics:
            if metric in history and len(history[metric]) > 0:
                fig.add_trace(go.Scatter(
                    x=steps,
                    y=history[metric],
                    mode='lines',
                    name=metric.upper(),
                    line=dict(width=2, color=colors.get(metric, '#95A5A6'))
                ))
    
    fig.update_layout(
        title='训练损失分解',
        xaxis_title='训练步数',
        yaxis_title='Loss',
        yaxis_type='log',
        template='plotly_white',
        hovermode='x unified',
        height=600,
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.8)')
    )
    
    return fig


def plot_connectivity_heatmap(matrix: pd.DataFrame) -> go.Figure:
    """
    绘制连通性矩阵热力图
    
    Args:
        matrix: 连通性矩阵 DataFrame
    """
    fig = go.Figure(data=go.Heatmap(
        z=matrix.values,
        x=matrix.columns.tolist(),
        y=matrix.index.tolist(),
        colorscale='YlOrRd',
        text=np.round(matrix.values, 3),
        texttemplate='%{text}',
        textfont={"size": 10},
        colorbar=dict(title="连通性 C_ij"),
    ))
    
    fig.update_layout(
        title='井间连通性矩阵',
        xaxis_title='井号',
        yaxis_title='井号',
        template='plotly_white',
        height=600,
        width=700,
    )
    
    return fig


def plot_production_comparison(t_days: np.ndarray, 
                               qg_pred: np.ndarray,
                               qg_obs: np.ndarray,
                               well_id: str = 'SY9') -> go.Figure:
    """
    绘制产量预测对比图
    
    Args:
        t_days: 时间（天）
        qg_pred: 预测产气量
        qg_obs: 观测产气量
        well_id: 井号
    """
    fig = go.Figure()
    
    # 观测值
    fig.add_trace(go.Scatter(
        x=t_days, y=qg_obs,
        mode='markers',
        name='观测值',
        marker=dict(size=4, color='black', opacity=0.5)
    ))
    
    # 预测值
    fig.add_trace(go.Scatter(
        x=t_days, y=qg_pred,
        mode='lines',
        name='PINN 预测',
        line=dict(width=2, color='#E74C3C')
    ))
    
    fig.update_layout(
        title=f'井 {well_id} 产气量拟合',
        xaxis_title='时间 (天)',
        yaxis_title='产气量 (m³/d)',
        template='plotly_white',
        hovermode='x unified',
        height=500,
    )
    
    return fig


def plot_strategy_comparison(strategies: Dict) -> go.Figure:
    """
    绘制制度优化策略对比
    
    Args:
        strategies: {策略名: {'t_days': [...], 'qg': [...], 'Gp': [...], 'sw': [...]}}
    """
    fig = go.Figure()
    
    colors = {
        '稳产方案': '#E74C3C',
        '阶梯降产': '#F39C12',
        '控压方案': '#27AE60',
    }
    
    for name, data in strategies.items():
        if 't_days' in data and 'Gp' in data:
            fig.add_trace(go.Scatter(
                x=data['t_days'],
                y=data['Gp'] / 1e6,  # 百万 m³
                mode='lines',
                name=name,
                line=dict(width=3, color=colors.get(name, 'blue'))
            ))
    
    fig.update_layout(
        title='累计产气对比',
        xaxis_title='时间 (天)',
        yaxis_title='累计产气 (百万 m³)',
        template='plotly_white',
        hovermode='x unified',
        height=500,
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.9)')
    )
    
    return fig


def plot_risk_index(risk_data: Dict) -> go.Figure:
    """
    绘制水侵风险指数
    
    Args:
        risk_data: {well_id: {'t_days': [...], 'R_w': [...]}}
    """
    fig = go.Figure()
    
    for wid, data in risk_data.items():
        if 't_days' in data and 'R_w' in data:
            fig.add_trace(go.Scatter(
                x=data['t_days'],
                y=data['R_w'],
                mode='lines',
                name=wid,
                line=dict(width=2)
            ))
    
    # 风险区域着色
    fig.add_hrect(y0=0, y1=0.15, fillcolor="green", opacity=0.1, 
                  annotation_text="安全", annotation_position="top left")
    fig.add_hrect(y0=0.15, y1=0.35, fillcolor="yellow", opacity=0.1,
                  annotation_text="预警", annotation_position="top left")
    fig.add_hrect(y0=0.35, y1=0.60, fillcolor="orange", opacity=0.1,
                  annotation_text="危险", annotation_position="top left")
    fig.add_hrect(y0=0.60, y1=1.0, fillcolor="red", opacity=0.1,
                  annotation_text="水淹", annotation_position="top left")
    
    fig.update_layout(
        title='水侵风险指数 R_w(t)',
        xaxis_title='时间 (天)',
        yaxis_title='风险指数 R_w',
        template='plotly_white',
        hovermode='x unified',
        height=500,
        yaxis=dict(range=[-0.05, 1.05])
    )
    
    return fig
