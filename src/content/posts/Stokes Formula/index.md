---
title: Stokes Formula
published: 2025-09-19
description: 这是一篇从外微分视角统一数学分析遇到的Green、Gauss、Stokes等公式的笔记。
tags: [Math, Analysis]
image: ./cover-image.jpg
category: Front-end
draft: false
---

$${\int}_{\partial D} \omega = {\int}_D d\omega$$

## Exterior Differential

在三维空间上，给出所有的外微分形式：

$$\omega = f$$
$$\omega_1 = Pdx + Qdy + Rdz$$
$$\omega_2 = A \, dy \wedge dz + B \, dz \wedge dx + C \, dx \wedge dy$$
$$\omega_3 = H \, dx \wedge dy \wedge dz$$

对外微分求微分：

$$d\omega = \frac{\partial f}{\partial x}dx + \frac{\partial f}{\partial y}dy + \frac{\partial f}{\partial z}dz$$

$$
d\omega_1 = \left| \begin{array}{ccc}
dy \wedge dz & dz \wedge dx & dx \wedge dy \\
\frac{\partial}{\partial x} & \frac{\partial}{\partial y} & \frac{\partial}{\partial z}  \\
P & Q & R \\
\end{array}\right| = \left(\frac{\partial R}{\partial y}-\frac{\partial Q}{\partial z}\right) dy \wedge dz+\left(\frac{\partial P}{\partial z}-\frac{\partial R}{\partial x}\right) dz \wedge dx+\left(\frac{\partial Q}{\partial x}-\frac{\partial P}{\partial y}\right) dx \wedge dy
$$

$$d\omega_2= \left(\frac{\partial A}{\partial x}+\frac{\partial B}{\partial y}+\frac{\partial C}{\partial z}\right) dx\wedge dy \wedge dz$$

$$d\omega_3=0$$

我们会发现，对于n-形式求微分得到n+1-形式

| Differential Forms (Exterior Calculus)                       | Vector Calculus                                              |
| :----------------------------------------------------------- | :----------------------------------------------------------- |
| $ω = f$                                                      | $\oint_L \mathbf{f} \cdot d\mathbf{s} = \oint_L \mathbf{f} \cdot \mathbf{e}_{\tau} ds$ |
| $ω_1 = Pdx + Qdy + Rdz$                                      | $\iint_S \mathbf{f} \, d\mathbf{S} = \iint_S \mathbf{f} \cdot \mathbf{e}_{n} dS$ |
| $ω_2 = A \, dy \wedge dz + B \, dz \wedge dx + C \, dx \wedge dy$ |                                                              |
| $ω_3 = H \, dx \wedge dy \wedge dz$                          |                                                              |
|                                                              |                                                              |
| $dω = \frac{\partial f}{\partial x} dx + \frac{\partial f}{\partial y} dy + \frac{\partial f}{\partial z} dz$ | $\mathbf{grad} \, f = \nabla f = \frac{\partial f}{\partial x} \mathbf{i} + \frac{\partial f}{\partial y} \mathbf{j} + \frac{\partial f}{\partial z} \mathbf{k}$ |
| $dω_1 = (\frac{\partial R}{\partial y} - \frac{\partial Q}{\partial z}) dy \wedge dz + (\frac{\partial P}{\partial z} - \frac{\partial R}{\partial x}) dz \wedge dx + (\frac{\partial Q}{\partial x} - \frac{\partial P}{\partial y}) dx \wedge dy$ | $\mathbf{rot} \, \mathbf{u} = \nabla \times \mathbf{u} = (\frac{\partial R}{\partial y} - \frac{\partial Q}{\partial z}) \mathbf{i} + (\frac{\partial P}{\partial z} - \frac{\partial R}{\partial x}) \mathbf{j} + (\frac{\partial Q}{\partial x} - \frac{\partial P}{\partial y}) \mathbf{k}$ |
| $dω_2 = (\frac{\partial A}{\partial x} + \frac{\partial B}{\partial y} + \frac{\partial C}{\partial z}) dx \wedge dy \wedge dz$ | $\mathbf{div} \, \mathbf{v} = \nabla \cdot \mathbf{v} = \frac{\partial A}{\partial x} + \frac{\partial B}{\partial y} + \frac{\partial C}{\partial z}$ |
| $dω_3 = 0$                                                   |                                                              |


## grad、div、rot

对于 $\boldsymbol{f} = f(x,y,z)$

$$\mathbf{grad} \, \boldsymbol{f} = \nabla \boldsymbol{f} = \frac{\partial f}{\partial x}\mathbf{i} + \frac{\partial f}{\partial y}\mathbf{j} + \frac{\partial f}{\partial z}\mathbf{k}$$

对于 $\boldsymbol{u} = (P,Q,R)$

$$\mathbf{rot} \, \boldsymbol{u} = \nabla \times \mathbf{u} = \left(\frac{\partial R}{\partial y}-\frac{\partial Q}{\partial z}\right)\mathbf{i}+\left(\frac{\partial P}{\partial z}-\frac{\partial R}{\partial x}\right)\mathbf{j}+\left(\frac{\partial Q}{\partial x}-\frac{\partial P}{\partial y}\right)\mathbf{k}$$

对于 $\boldsymbol{v} = (A,B,C)$

$$\mathbf{div} \, \boldsymbol{v} = \nabla \cdot \mathbf{v} = \frac{\partial A}{\partial x}+\frac{\partial B}{\partial y}+\frac{\partial C}{\partial z}$$

请注意梯度和散度用 $\nabla$ 算子时一个是直接运算，另一个是点乘，内积符号不可忽视！这导致了散度是唯一标量！

三维空间中只能有三种度，分别是梯度、旋度、散度

## 有向线元、有向面元

知道
$$d\mathbf{s} = (dx,dy,dz) = \left(\frac{dx}{ds},\frac{dy}{ds},\frac{dz}{ds}\right)ds = \mathbf{e}_{\tau} \, ds$$
表示有向线元

同时
$$d\mathbf{S} = (dy \wedge dz,dz \wedge dx,dx \wedge dy) = (dS\cos\alpha, dS\cos\beta, dS\cos\gamma) = \mathbf{e}_n dS$$
表示有向面元

为什么有向线元是线元与切向量之积，而有向面元是面元与法向量之积？

这种形式完全是由三维空间中的外微分形式决定，表现在物理中就是环量与通量，在自由度理解就是将向量投影到唯一确定线元和面元的东西上，这个只能是单自由度的向量（投影到平面的方法与结果不唯一），切向量确定线元，法向量确定面元。

## Green Formula

$${\oint}_L Pdx + Qdy = {\iint}_D \left(\frac{\partial Q}{\partial x}-\frac{\partial P}{\partial y}\right) dx dy$$

根据第二型线、面积分定义强调一下计算：

$${\oint}_L\mathbf{f} \cdot d\mathbf{s} = {\oint}_L \mathbf{f} \cdot \mathbf{e}_{\tau} \, ds$$

$${\iint}_D \nabla \times \mathbf{f} \cdot d\mathbf{S} = {\iint}_D (\nabla \times \mathbf{f}) \cdot \mathbf{e}_n \, dS$$

得到常见写法：

$${\oint}_L\mathbf{f} \cdot d\mathbf{s} = {\iint}_D (\nabla \times \mathbf{f}) \cdot d\mathbf{S}$$

实际上可以理解为Stokes公式在xy的投影，dz=0

## Gauss Formula and Stokes Formula

Gauss Formula：
$$
\begin{aligned}
{\oiint}_{\Sigma_{外}} A \, dy \wedge dz + B \, dz \wedge dx + C \, dx \wedge dy &= {\iiint}_V \left(\frac{\partial A}{\partial x}+\frac{\partial B}{\partial y}+\frac{\partial C}{\partial z}\right) dxdydz
\end{aligned}
$$

$$
{\oiint}_{\Sigma_{外}} \mathbf{v} \cdot d\mathbf{S} = {\iiint}_V \nabla \cdot \mathbf{v} \, dV = {\iiint}_V \textbf{div} \, \mathbf{v} \, dV
$$

Stokes Formula：
$$
\begin{aligned}
{\oint}_L Pdx + Qdy+Rdz = &{\iint}_{\Sigma} \left(\frac{\partial R}{\partial y}-\frac{\partial Q}{\partial z}\right) dy \wedge dz \\
&+\left(\frac{\partial P}{\partial z}-\frac{\partial R}{\partial x}\right) dz \wedge dx \\
&+\left(\frac{\partial Q}{\partial x}-\frac{\partial P}{\partial y}\right) dx \wedge dy
\end{aligned}
$$

$${\oint}_L \mathbf{u} \cdot d\mathbf{s} = {\iint}_{\Sigma} (\nabla \times \mathbf{u}) \cdot d\mathbf{S} = {\iint}_{\Sigma} \textbf{rot} \, \mathbf{u} \cdot d\mathbf{S}$$

现在我们回顾一下外微分形式就会发现：

| Differential Forms (Exterior Calculus)                       | Vector Calculus                                              |
| :----------------------------------------------------------- | :----------------------------------------------------------- |
| $ω = f$                                                      | $\oint_L \mathbf{f} \cdot d\mathbf{s} = \oint_L \mathbf{f} \cdot \mathbf{e}_{\tau} ds$ |
| $ω_1 = Pdx + Qdy + Rdz$                                      | $\iint_S \mathbf{f} \, d\mathbf{S} = \iint_S \mathbf{f} \cdot \mathbf{e}_{n} dS$ |
| $ω_2 = A \, dy \wedge dz + B \, dz \wedge dx + C \, dx \wedge dy$ |                                                              |
| $ω_3 = H \, dx \wedge dy \wedge dz$                          |                                                              |
|                                                              |                                                              |
| $dω = \frac{\partial f}{\partial x} dx + \frac{\partial f}{\partial y} dy + \frac{\partial f}{\partial z} dz$ | $\mathbf{grad} \, f = \nabla f = \frac{\partial f}{\partial x} \mathbf{i} + \frac{\partial f}{\partial y} \mathbf{j} + \frac{\partial f}{\partial z} \mathbf{k}$ |
| $dω_1 = (\frac{\partial R}{\partial y} - \frac{\partial Q}{\partial z}) dy \wedge dz + (\frac{\partial P}{\partial z} - \frac{\partial R}{\partial x}) dz \wedge dx + (\frac{\partial Q}{\partial x} - \frac{\partial P}{\partial y}) dx \wedge dy$ | $\mathbf{rot} \, \mathbf{u} = \nabla \times \mathbf{u} = (\frac{\partial R}{\partial y} - \frac{\partial Q}{\partial z}) \mathbf{i} + (\frac{\partial P}{\partial z} - \frac{\partial R}{\partial x}) \mathbf{j} + (\frac{\partial Q}{\partial x} - \frac{\partial P}{\partial y}) \mathbf{k}$ |
| $dω_2 = (\frac{\partial A}{\partial x} + \frac{\partial B}{\partial y} + \frac{\partial C}{\partial z}) dx \wedge dy \wedge dz$ | $\mathbf{div} \, \mathbf{v} = \nabla \cdot \mathbf{v} = \frac{\partial A}{\partial x} + \frac{\partial B}{\partial y} + \frac{\partial C}{\partial z}$ |
| $dω_3 = 0$                                                   |                                                              |

最左列的外微分形式和等式左边相似，外微分的微分和等式右边相似
我们来看看Green Formula：

记 $\omega_1 = Pdx+Qdy$，则 $d\omega_1 = \left(\frac{\partial Q}{\partial x}-\frac{\partial P}{\partial y}\right) dx \wedge dy$
写作：

$${\oint}_L \omega_1 =  {\iint}_D d\omega_1$$

同理Gauss Formula：

$${\oiint}_{\Sigma_{外}} \omega_2 = {\iiint}_V d\omega_2$$

最后的Stokes Formula：

$${\oint}_L \omega_1 = {\iint}_{\Sigma} d\omega_1$$

而回顾Newton-Leibniz Formula：

$${\int}_a^b \frac{d}{dx}f(x)dx = f(x) \bigg|_a^b = f(b)-f(a)$$

以上公式都是在说：区域上的信息可以通过边界上高一阶的信息表出

就像Gauss Formula，空间区域各点的散度之和可以通过区域边界的通量之和得到；对于牛莱公式，平面曲线无数个点的函数值之和可以通过曲线边界仅仅两个点的函数值之差获得，这样我们避开了复杂的内部形状，直接通过最易刻画的边界获取内部信息；同时我们也可以不考虑边界的形状，通过电荷散度获得边界的电通量

从外微分的视角来看，区域 D 上某形式的外导数的积分，等于该形式在区域 D 边界上的积分，也就是广义Stokes Formula，也是高维空间的微积分基本定理：
$${\int}_{\partial D} \omega = {\int}_D d\omega$$

| 外微分形式的次数 | 空间         | 公式                   |
| ---------------- | ------------ | ---------------------- |
| $0$              | 直线段       | Newton-Leibniz Formula |
| $1$              | 平面区域     | Green Formula          |
| $1$              | 空间曲面     | Stokes Formula         |
| $2$              | 空间中的区域 | Gauss Formula          |

下面我们站在外微分的高度回过头看看课本上的一些奇形怪状的定理，在这之前先给出一个重要引理：

## Wedge Product

在这里顺便提一下楔积的两条性质：
> **Note**
>
> $dx \wedge dx = 0$
>
> $dx \wedge dy = -dy \wedge dx$

似乎很像外积

|              |                                                              |                                        |                                                 |
| ------------ | ------------------------------------------------------------ | -------------------------------------- | ----------------------------------------------- |
| **特性**     | **楔积 (Wedge Product, ∧)**                                  | **外积 (Outer Product) - 张量积形式**  | **外积 (Outer Product) - 三维叉积形式 (×)**     |
| **通用性**   | 适用于任意维度的向量空间，可作用于 k-向量和微分形式。        | 适用于任意维度的向量，结果是矩阵。     | 主要定义在三维欧氏空间中 (也有七维的特殊情况)。 |
| **结果类型** | 两个向量的楔积是2-向量 (bivector)。k-向量和 l-向量的楔积是 (k+l)-向量。 | 两个向量的外积是一个矩阵。             | 两个向量的叉积是一个向量 (伪向量)。             |
| **代数结构** | 属于外代数。满足结合律和反对称性 (u∧v=−v∧u)。                | 属于张量代数。一般不满足反对称性。     | 不满足结合律，但满足反对称性 (u×v=−v×u)。       |
| **几何意义** | 表示有向的“超体积”元素 (例如，长度、面积、体积等)。          | 将两个向量的元素相乘构成矩阵。         | 结果向量垂直于原向量，长度为平行四边形面积。    |
| **符号**     | ∧                                                            | 通常用并列或 ⊗ (更一般的张量积) 表示。 | ×                                               |

这在引理证明中发挥重要作用，很多项可以直接变成0和相消

## Poincaré Lemma

> **Danger: Poincaré Lemma**
>
> 若 $\omega$ 为一外微分形式，其微分形式的系数具有二阶连续偏微商，则 $dd\omega = 0$
>
> 逆定理：若 $\omega$ 是一个 $p$ 次外微分式且 $d\omega = 0$，则存在一个 $p-1$ 次外微分形式 $a$，使 $\omega = da$

在三维空间中的证明就是直接微分计算，这里不具体展开
说人话就是存在一个$p-1$次的势函数

下面展开一些个人理解：总的来说，Poincaré Lemma 是连接“场”、“势”和外微分形式的关键

在满足Poincaré Lemma的情况下，外微分式存在与之对应的场，**外微分式**进行微分运算得到新的外微分式，这一旧一新外微分式对应的两个场之间也存在运算关系：
零次外微分式求微分得到一次外微分式，对应于梯度算子运算旧场得到新场，这就是梯度场
一次外微分式求微分得到二次外微分式（环量），对应于梯度算子叉乘旧场得到新场，这就是旋度场
二次外微分式求微分得到三次外微分式（通量），对应于梯度算子点乘旧场得到新场，这就是散度场

| Differential Forms (Exterior Calculus)                       | Vector Calculus                                              |
| :----------------------------------------------------------- | :----------------------------------------------------------- |
| $ω = f$                                                      | $\oint_L \mathbf{f} \cdot d\mathbf{s} = \oint_L \mathbf{f} \cdot \mathbf{e}_{\tau} ds$ |
| $ω_1 = Pdx + Qdy + Rdz$                                      | $\iint_S \mathbf{f} \, d\mathbf{S} = \iint_S \mathbf{f} \cdot \mathbf{e}_{n} dS$ |
| $ω_2 = A \, dy \wedge dz + B \, dz \wedge dx + C \, dx \wedge dy$ |                                                              |
| $ω_3 = H \, dx \wedge dy \wedge dz$                          |                                                              |
|                                                              |                                                              |
| $dω = \frac{\partial f}{\partial x} dx + \frac{\partial f}{\partial y} dy + \frac{\partial f}{\partial z} dz$ | $\mathbf{grad} \, f = \nabla f = \frac{\partial f}{\partial x} \mathbf{i} + \frac{\partial f}{\partial y} \mathbf{j} + \frac{\partial f}{\partial z} \mathbf{k}$ |
| $dω_1 = (\frac{\partial R}{\partial y} - \frac{\partial Q}{\partial z}) dy \wedge dz + (\frac{\partial P}{\partial z} - \frac{\partial R}{\partial x}) dz \wedge dx + (\frac{\partial Q}{\partial x} - \frac{\partial P}{\partial y}) dx \wedge dy$ | $\mathbf{rot} \, \mathbf{u} = \nabla \times \mathbf{u} = (\frac{\partial R}{\partial y} - \frac{\partial Q}{\partial z}) \mathbf{i} + (\frac{\partial P}{\partial z} - \frac{\partial R}{\partial x}) \mathbf{j} + (\frac{\partial Q}{\partial x} - \frac{\partial P}{\partial y}) \mathbf{k}$ |
| $dω_2 = (\frac{\partial A}{\partial x} + \frac{\partial B}{\partial y} + \frac{\partial C}{\partial z}) dx \wedge dy \wedge dz$ | $\mathbf{div} \, \mathbf{v} = \nabla \cdot \mathbf{v} = \frac{\partial A}{\partial x} + \frac{\partial B}{\partial y} + \frac{\partial C}{\partial z}$ |
| $dω_3 = 0$                                                   |                                                              |

现在我们运用一下

## 无旋场、无源场、调和场

> **Note: 无旋场**
>
> 设 $(G)$ 是一维单连通域，$\mathbf{F} = (P,Q,R) \in C^{(1)}((G))$，则以下命题等价：
>
> （1）$\mathbf{F}$ 是一个无旋场，即 $\textbf{rot} \, \mathbf{F} = \nabla \times \mathbf{F} = 0$
>
> （2）单连域内任一简单闭曲线 $C$ 的环量为0，即 ${\oint}_C \mathbf{F} \cdot d\mathbf{s} = {\oint}_C Pdx + Qdy+Rdz =0$
>
> （3）$\mathbf{F}$ 是保守场，即 ${\oint}_C \mathbf{F} \cdot d\mathbf{s}$ 与路径无关
>
> （4）$\mathbf{F}$ 是有势场，即 $Pdx + Qdy+Rdz$ 是某一函数的全微分，或者说 $\mathbf{F} = (P,Q,R) = \left(\frac{\partial f}{\partial x},\frac{\partial f}{\partial y},\frac{\partial f}{\partial z}\right) = \nabla f$

Proof:
	（1）-> （2）就是Stokes Formula
	（2）->（3）曲线加减再运用定理即可
	（3）->（1）路径为闭合曲线即可
	现在来看（1）->（4）和（4）->（1）：梯度场等价于无旋场
	

​	由Poincaré引理逆定理知，$Pdx + Qdy+Rdz$ 是一个一次外微分式且 $d(Pdx + Qdy+Rdz) = \textbf{rot} \, \mathbf{F}=0$，则存在一个零次外微分式 $a$，使 $Pdx + Qdy+Rdz = da$，$a$ 就是势函数，$\mathbf{F}$ 因此是有势场

​	由Poincaré引理知，势函数 $a$ 作为零次外微分式，其微分系数具有二阶连续偏导，则 $\textbf{rot} \, \mathbf{F} =dda =  0$

因此我们看到，势函数其实是更高一阶的信息，我们往往可以从势函数这个更高的角度获取信息和不变量，比如我们摆脱对力的形式和大小的具体关注，将会发现一系列守恒定律；我们在使用动能定理时只关注首末状态而不关心中间具体变化，这就是从边界上获取内部信息


> **Note: 无源场**
>
> 设 $(G)$ 是二维单连通域，$\mathbf{F} \in C^{(1)}((G))$，则以下命题等价：
>
> （1）$\mathbf{F}$ 是一个无源场，即 $\textbf{div} \, \mathbf{F} = \nabla \cdot \mathbf{F} = 0$
>
> （2）$\mathbf{F}$ 沿 $(G)$ 内任一不自交闭曲面 $(S)$ 的通量为0，即 ${\oiint}_{S} \mathbf{F} \cdot d\mathbf{S} = 0$
>
> （3）在 $(G)$ 内存在一向量函数 $\mathbf{B}(M)$，使 $\mathbf{F} = \nabla \times \mathbf{B}$，即 $\mathbf{F}$ 是某一向量场 $\mathbf{B}$ 的旋度场，$\mathbf{B}$ 称为 $\mathbf{F}$ 的一个向量势

Proof:
	在无旋场证明里已经展示了引理使用过程，现在直观感受一下（3）在说什么：旋度场等价于无源场
	$\mathbf{F} \cdot d\mathbf{S}$ 是二次外微分式，==同时满足了引理==，那么会有一个一次外微分式的势函数，这个势函数存在于一个场 $\mathbf{B}$
	势函数作为一次外微分式求微分得到 $\mathbf{F} \cdot d\mathbf{S}$，同时 $\mathbf{B}$ 叉乘梯度算子得到 $\mathbf{F}$


对比一下：
> **Note**
>
> $F$ 是一个无旋场，即 $\textbf{rot} \, \mathbf{F} = \nabla \times \mathbf{F} = 0$
>
> $\mathbf{F}$ 是有势场，即 $Pdx + Qdy+Rdz$ 是某一函数的全微分，或者说 $\mathbf{F} = (P,Q,R) = (\frac{\partial f}{\partial x},\frac{\partial f}{\partial y},\frac{\partial f}{\partial z}) = \nabla f$
>
> $\mathbf{F}$ 是一个无源场，即 $\textbf{div} \, \mathbf{F} = \nabla \cdot \mathbf{F} = 0$
>
> 在 $(G)$ 内存在一向量函数 $\mathbf{B}(M)$，使 $\mathbf{F} = \nabla \times \mathbf{B}$，即 $\mathbf{F}$ 是某一向量场 $\mathbf{B}$ 的旋度场，$\mathbf{B}$ 称为 $\mathbf{F}$ 的一个向量势

（次数是指外微分的阶数，不要和多项式的次数搞混掉了，搞混就想想高阶导数）
什么感觉？判定无旋还是无源都是要从低次走向高次，因此对应的场运算次序是叉乘、点乘
由引理寻找势函数都是从高次(p次)走向低次(p-1次)，因此会发现场运算次序是直接运算、叉乘

似乎我们可以隐约感受到广义Stokes Formula表现出的某种对偶吧！

| Differential Forms (Exterior Calculus)                       | Vector Calculus                                              |
| :----------------------------------------------------------- | :----------------------------------------------------------- |
| $ω = f$                                                      | $\oint_L \mathbf{f} \cdot d\mathbf{s} = \oint_L \mathbf{f} \cdot \mathbf{e}_{\tau} ds$ |
| $ω_1 = Pdx + Qdy + Rdz$                                      | $\iint_S \mathbf{f} \, d\mathbf{S} = \iint_S \mathbf{f} \cdot \mathbf{e}_{n} dS$ |
| $ω_2 = A \, dy \wedge dz + B \, dz \wedge dx + C \, dx \wedge dy$ |                                                              |
| $ω_3 = H \, dx \wedge dy \wedge dz$                          |                                                              |
|                                                              |                                                              |
| $dω = \frac{\partial f}{\partial x} dx + \frac{\partial f}{\partial y} dy + \frac{\partial f}{\partial z} dz$ | $\mathbf{grad} \, f = \nabla f = \frac{\partial f}{\partial x} \mathbf{i} + \frac{\partial f}{\partial y} \mathbf{j} + \frac{\partial f}{\partial z} \mathbf{k}$ |
| $dω_1 = (\frac{\partial R}{\partial y} - \frac{\partial Q}{\partial z}) dy \wedge dz + (\frac{\partial P}{\partial z} - \frac{\partial R}{\partial x}) dz \wedge dx + (\frac{\partial Q}{\partial x} - \frac{\partial P}{\partial y}) dx \wedge dy$ | $\mathbf{rot} \, \mathbf{u} = \nabla \times \mathbf{u} = (\frac{\partial R}{\partial y} - \frac{\partial Q}{\partial z}) \mathbf{i} + (\frac{\partial P}{\partial z} - \frac{\partial R}{\partial x}) \mathbf{j} + (\frac{\partial Q}{\partial x} - \frac{\partial P}{\partial y}) \mathbf{k}$ |
| $dω_2 = (\frac{\partial A}{\partial x} + \frac{\partial B}{\partial y} + \frac{\partial C}{\partial z}) dx \wedge dy \wedge dz$ | $\mathbf{div} \, \mathbf{v} = \nabla \cdot \mathbf{v} = \frac{\partial A}{\partial x} + \frac{\partial B}{\partial y} + \frac{\partial C}{\partial z}$ |
| $dω_3 = 0$                                                   |                                                              |

>[!note] 调和场
>无源且无旋的向量场 $\mathbf{A}$ 称为调和场，即 $$\nabla \cdot \mathbf{A} = \mathbf{0} ，\nabla \times \mathbf{A} = \mathbf{0}$$
>因为无旋 $\mathbf{A}$ 存在势函数 $u$，使$$ \mathbf{A} = \nabla u$$
>因为无源，所以$$\nabla \cdot \mathbf{A} = \nabla^2 u = \Delta u = 0$$
>即满足 $\textbf{Laplace}$ 方程$$ \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} + \frac{\partial^2 u}{\partial z^2} = 0$$
>
>若 $\mathbf{A}$ 无旋而有连续分布的源，源强度为 $\rho(M)$，且存在势函数 $u$
>$$\rho = \nabla \cdot \mathbf{A} = \Delta u$$
>$$\Delta u = \rho$$
>满足 $\textbf{Poisson}$ 方程


## 运算法则

> **Note: 梯度运算法则**
>
> $$\nabla (C_1u+C_2v) = C_1\nabla u + C_2\nabla v$$
>
> $$\nabla(uv) = u\nabla v+v\nabla u$$
>
> $$\nabla\left(\frac{u}{v}\right) = \frac{v\nabla u - u\nabla v}{v^2}$$
>
> $$\nabla(f(u)) = f'(u)\nabla u$$

> **Important: 散度运算法则**
>
> $$\nabla \cdot (u\mathbf{A}) = u (\nabla \cdot \mathbf{A}) + (\nabla u) \cdot \mathbf{A}$$
>
> $u\mathbf{A} = (uP,uQ,uR)$ 由散度定义易证
>
> $$\textbf{div} \, \mathbf{A} = \nabla \cdot \mathbf{A} = \frac{\partial P}{\partial x} + \frac{\partial Q}{\partial y} + \frac{\partial R}{\partial z}$$

> **Example: 旋度运算法则**
>
> $u$为数量值函数，否则无意义
>
> $$ \nabla \times (u\mathbf{A}) = u(\nabla \times \mathbf{A}) + (\nabla u)\times \mathbf{A}$$
>
> 由旋度定义易证
>
> $$
> \textbf{rot} \, \mathbf{A} = \nabla \times \mathbf{A} = 
> \left| \begin{array}{ccc}
> \mathbf{i} & \mathbf{j} & \mathbf{k} \\
> \frac{\partial}{\partial x} & \frac{\partial}{\partial y} & \frac{\partial}{\partial z}  \\
> P & Q & R \\
> \end{array}\right|
> $$

> **Danger: 其他运算法则**
>
> $$\nabla \cdot(\nabla \times \mathbf{A}) = 0$$
>
> $$\nabla \times(\nabla u) = 0$$
>
> $$\nabla \times(\nabla \times \mathbf{A}) = \nabla(\nabla \cdot \mathbf{A}) - \nabla^2 \mathbf{A}$$
>
> $$\nabla \cdot (\mathbf{A} \times \mathbf{B}) = \mathbf{B} \cdot (\nabla \times \mathbf{A}) - \mathbf{A} \cdot (\nabla \times \mathbf{B})$$

第一条说的是旋度场等价于无源场
第二条说的是梯度场等价于无旋场

计算上会发现原因是：若二阶混合偏导数连续，则可以交换次序
我们知道从低次升到高次是在进行直接运算、叉乘、点乘，直到散度场对应的3-形式求微分为0，对应到新场就是$\textbf 0$

更广义的就是
> **Danger: Poincaré Lemma**
>
> 若 $\omega$ 为一外微分形式，其微分形式的系数具有二阶连续偏微商，则 $dd\omega = 0$

至于第三第四条与霍奇对偶相关