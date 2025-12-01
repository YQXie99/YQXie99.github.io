---
layout: blog
title: 从0开始看懂PPO
date: 2022-06-10
permalink: /blogs/2022-06/ppo-intro/
tags:
  - RL
  - Chinese
  - Notes
intro: 简明介绍强化学习基础与PPO核心思想。
---


## 强化学习简介

强化学习（RL）通常将问题建模为两个部分，即智能体（agent）与环境（environment），智能体通过与环境交互，学习如何进行决策。在每一时刻，智能体可以进行环境观测（Observation），采取行为（Action），并从环境中获取奖励（Reward）；环境则基于智能体采取的行为进行更新，准备好下一时刻反馈给智能体的观测以及奖励。


这一过程通常被建模为马尔可夫决策过程（Markov Decision Process，简称MDP），记为$M(S,A,P,R,\gamma)$。其中$S$代表状态（State）空间，$A$代表行为（Action）空间，$P$代表状态转移函数（Probability），表示选择某一动作的可能性，$R$代表在特定状态下选取某动作的奖励（Reward），$\gamma\in(0,1)$是奖励的折扣因子(discount factor)。将奖励按照一定衰减比例计算叠加起来，即可得到收益（Return）：$G_t=\sum_{i=0}^{\infty} \gamma ^{i} R_{t+i+1}$。RL的目的是选择一个最优的策略$\pi$，而对策略好坏的评判标准则是收益的大小。智能体需要根据自己的观测调整行为，从而最大化收益。

在RL中，智能体通常由策略（Policy），价值函数（Value Function）与模型（Model）中的一种或多种组成。其中，策略是状态（Status，即智能体观测到的全部信息）与行为的映射，描述某一状态下执行不同行为的概率；价值函数用于预测奖励，判断当前所处状态的好坏，好的状态或行为对应着更高的收益；模型则是智能体对环境的建模，包括预测下一个状态的概率，以及预测环境可能给出的奖励。

为了更好地描述奖励与收益，我们定义了两个价值函数，即行为价值函数与行为-状态价值函数。

- 行为价值函数：从状态s开始，遵循当前策略$\pi$可获得的收益的期望。 \\( v(s)= \\mathbb{E} [G_t\\mid S_t=s] \\)。

- 行为-状态价值函数：给定状态与策略 ，执行具体行为时可以得到的收益的期望，一般使用状态行为对进行描述。 \\( q_\\pi(s,a)=\\mathbb{E}_\\pi[G_t\\mid S_t=s,A_t=a] \\)。

根据定义，我们可以推出行为价值函数与行为-状态价值函数的关系。式2.3表明，给定状态$s$与策略$\pi$，则$s$的价值为$\pi$采取的全部可能行为的期望；式2.4表明，给定状态$s$与策略$\pi$，行为$a$的价值可以分为两部分，其一是通过这个状态获得的价值，其二是下一步状态价值的期望。
$$
   v_\pi(s)=\sum_{a\in A}{\pi(a|s)q_\pi(s,a)} \\
   q_\pi(s,a)=R^a_s+\gamma \sum_{s'\in S}{P^a_{ss'}v_\pi(s')}
$$

我们想要求解的是最优策略，即在所有可能的策略中，选择使价值函数最大策略。这也就是说，我们求解的最优策略对应的最优价值函数需要满足下式：
$$
   v_*(s)=max_\pi v_\pi(s) \\
   q_*(s,a)=max_\pi q_\pi(s,a)
$$

简单代换，即可得到贝尔曼最优方程（Bellman Optimality Equation），也被称为Q函数。对其求解，即可得到最优策略。
$$
q_*(s,a)=R^a_s+\gamma \sum_{s'\in S}{P^a_{ss'} max_{a'} q_*(s',a')}
$$

## 策略梯度
然而，由于算力及内存的限制，我们很难针对最优方程直接求解从而得到最优的策略。尤其对于较为复杂的任务，他们涉及到庞大或连续的状态空间与动作空间，连状态-行为的表征都很难实现。因此，我们需要对价值函数或策略函数进行估算，从而在有限的资源限制下逐步逼近最优解。策略梯度（Policy Gradient）方法便是一种常用的估算方法。这个算法将策略参数化，以$\theta$表示，计算过程中需要根据收益的梯度确定策略参数的更新幅度。

通过定义，我们需要求取最大值的目标函数是
$$
L(\theta)=\mathbb{E}[\sum_{i=0} \gamma ^{i}r_{i+1} \mid \pi_{\theta}]
$$，
即$L(\theta)=v_{\pi_\theta}(s)$。一般而言，我们可以直接采用$\theta_{t+1} \leftarrow \theta_t+\alpha \nabla L(\theta)$的算法对参数$\theta$进行更新，但是对于上式而言，直接求解梯度是很难实现的，所以我们通过蒙特卡洛策略梯度（Monte Carlo Policy Gradient）对其进行无偏估计，之后根据估计值更新参数。因此，重点在于如何求取梯度。这里的梯度可以表示为：
$$
\nabla L(\theta) = \mathbb{E}_\pi\left[G_t \, \frac{\nabla \pi(a_t\mid s_t,\theta_t)}{\pi(a_t\mid s_t,\theta_t)}\right]
$$

但是，这样的方式仍然存在着一定的问题。对策略梯度优化会导致收益波动大，即样本方差较大，因此收敛速度慢。另外，在更新策略时，策略梯度倾向于增加收益大的动作出现的概率。考虑如下情况：不论行为的好坏，其带来的收益始终为正，而没有采样到的行为的收益始终是0，那么，好的行为如果没有被采样到，它出现的可能性同样会越来越低。因此，这里引入一个简单的值函数$b(S)$作为基线，在计算收益时，减去均值（或加权后的均值），从而使得收益有正有负，减少样本方差，并且保证策略更新的公平性和合理性：
$$
\nabla L(\theta) = \mathbb{E}_\pi\left[ (G_t-b(S_t))  \, \frac{\nabla \pi(a_t\mid s_t,\theta_t)}{\pi(a_t\mid s_t,\theta_t)}\right]
$$

进一步的，我们可以更换基线，使用状态价值函数作为基线函数。在近似求解过程中，价值函数需要自身迭代（bootstrap），并且对策略函数给出的结果进行评估。这便是演员-评判家（Actor-Critic）算法，评判家指值函数，演员指策略函数。为了表达方便起见，我们定义优势函数（advantage function），$A(s,a)=G-b(S)$，Actor-Critic算法中，优势函数即为$A(s,a)=G-V(s)$。

## PPO算法

PPO算法对于策略梯度算法进行了一定的修改与提升。上述算法的问题在于难以确定合适的步长，以及更新策略的样本利用率太低。为了利用旧策略的数据，我们使用重要性采样（Importance Sampling）的思想，将需要求取梯度的目标函数改写为：
$$
L_{\theta_{\text{old}}}(\theta)=\mathbb{E}\left[\frac{ \pi_{\theta}(a_t\mid s_t,\theta_t)}{\pi_{\theta_{\text{old}}}(a_t\mid s_t,\theta_t)}A_t\right]
$$

当新旧策略相差太大，这样的更新方式将使参数剧烈波动。因此，我们引入KL散度衡量两个策略的差异，当差异过大时相应地减小更新幅度，从而避免策略变化过大带来的波动，并且保证策略在迭代中逐渐趋近最优解。将KL散度的限制引入式中作为惩罚，即可得到新的待优化的目标函数： 
$$
L(\theta)=\mathbb{E}\left[\frac{ \pi_{\theta}(a_t\mid s_t,\theta_t)}{\pi_{\theta_{\text{old}}}(a_t\mid s_t,\theta_t)}A_t-\beta \, \text{KL}[\pi_{\theta_{\text{old}}}(\cdot\mid s_t),\pi_\theta (\cdot\mid s_t)]\right]
$$

其中，KL散度函数为：
$$
\text{KL}(P\parallel Q)=\int P(x)\log \frac{P(x)}{Q(x)}\,dx
$$

然而，KL散度惩罚的更新梯度不是很好求解。为此，PPO算法提出了一种实现起来更为简便、快速的算法，直接通过计算两个策略的比例衡量策略的不同，即计算
$$
r_t(\theta)=\frac{ \pi_{\theta}(a_t\mid s_t)}{\pi_{\theta_k}(a_t\mid s_t)}
$$


如果这个比值超过预先设置的范围，那么价值函数将被“修剪”（clipped），从而避免出现过大的波动。Clip函数定义为：
$$
\text{clip}(x;\,1-\epsilon,1+\epsilon)=
\begin{cases}
1-\epsilon, & x < 1-\epsilon \\
x, & 1-\epsilon \le x \le 1+\epsilon \\
1+\epsilon, & x > 1+\epsilon
\end{cases}
$$

此时待更新的函数为：
$$
L_{\text{clip}}(\theta)=\mathbb{E}_t\left[\min\left(r_t(\theta)A_t, 
\text{clip}(r_t(\theta);1-\epsilon,1+\epsilon)A_t\right)\right]
$$

PPO完整算法流程如下所示。PPO易于实现与调参，结果胜过当前最优算法，目前是强化学习领域里非常主流的算法。

PPO算法（简化伪代码）：

1. 初始化 Actor 参数 \\(\\theta\\) 与 Critic 参数 \\(\\phi\\)
2. 对于 \\(k=0,1,2,\\dots\\)：
   - 使用旧策略 \\(\\pi_{\\theta_{\\text{old}}}\\) 交互收集轨迹 \\(D_k\\)
   - 估计优势函数 \\(A_t\\)
   - 使用 Adam 优化器更新策略：\\(\\theta_{k+1}=\\arg\\max_{\\theta} L_{\\text{clip}}(\\theta)\\)
   - 使用 Adam 优化器更新价值函数：\\(\\phi_{k+1}=\\arg\\min_{\\phi} \, \\mathbb{E}[(V_{\\phi_k}(s_t)-R_t)^2]\\)