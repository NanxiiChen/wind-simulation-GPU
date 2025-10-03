# Shinozuka's harmonic synthesis method

为了便于分析脉动风的影响，工程上常将脉动风假定为零均值的正态随机平稳过程。其频率特性可由功率谱密度展现。规范中使用式（2.2）与式（2.3）来计算顺风向脉动风功率谱密度$S_u(n)$与竖向脉动风功率谱密度$S_w(n)$：

$$
\begin{gather}
\frac{nS_u(n)}{u^2_*} = \frac{200f}{(1+50f)^{5/3}} \tag{2.2}\\
\frac{nS_w(n)}{u^2_*} = \frac{6f}{(1+4f)^2} \tag{2.3}\\
f = \frac{nZ}{U_d} \tag{2.4}\\
u_* = \frac{KU_d}{\ln\dfrac{Z-z_d}{z_0}} \tag{2.5}\\
z_d = \bar{H} - \frac{z_0}{K} \tag{2.6}
\end{gather}
$$
式中：
- $n$ 为脉动风频率（Hz）；
- $u_*$ 为风的摩阻速度，亦称剪切速度（m/s）；
- $K$ 为无量纲常数，可取为0.4；
- $\bar{H}$ 为周围建筑物平均高度（m）；
- $z_0$ 为地表粗糙高度。

在模拟脉动风样本时，应考虑空间各点的相关性。设桥轴向为x方向，横向为y方向，竖向为z方向，则空间相关函数可用Davenport经验公式进行计算：
$$
\text{Coh}(x_i, x_j, y_i, y_j, z_i, z_j, w) = \exp\left\{-\frac{
    2w\left[ C_x^2(x_i - x_j)^2 + C_y^2(y_i - y_j)^2 + C_z^2(z_i - z_j)^2 \right]^{1/2}
}{
    2\pi(U_{zi}+U_{zj})
}\right\}
$$
其中：
- $C_x, C_y, C_z$ 为衰减系数，通常可取经验值$C_x = 16, C_y = 6, C_z = 10$；
- $(x_i, y_i, z_i)$ 与 $(x_j, y_j, z_j)$ 分别为节点 $i$ 与节点 $j$ 的空间坐标;
- $U_{zi}, U_{zj}$ 为节点 $i$ 与节点 $j$ 的平均风速；

从而，节点 $I$ 与节点 $j$ 的脉动风互谱密度函数 $S_{ij}$ 可表示为：
$$
S_{ij} = \sqrt{S_{ii}S_{jj}}\text{Coh}(x_i, x_j, y_i, y_j, z_i, z_j, w)
$$

由 $S_{ij}$ 可计算出节点 $i$ 与节点 $j$ 的互谱密度矩阵 $S(w)$。在获得空间各点的两两之间的脉动风互谱密度矩阵之后，便可进行谐波合成法的相关工作，该方法的主要步骤如下。

1. 将得到的互谱密度矩阵 $S(w)$ 进行Cholesky分解得到 $H(w)$：
$$
S(w) = H(w)H^T(w)
$$
2. 第 $j$ 点的脉动风时程样本可由Shinozuka's谐波合成法结合FFT技术进行模拟：
$$
f_j(pt) = 2\sqrt{\frac{\Delta w}{2\pi}} \text{Re}\left\{
    G_j(p\Delta t) \exp\left[
        i\left(
            \frac{p\pi}{M}
        \right)
    \right]
\right\}, \quad (p = 0, 1, \ldots, M-1; j=0, 1, \ldots, n)
$$

$G_j(p\Delta t)$ 可以通过 FFT 进行计算：
$$
G_j(p\Delta t) = \sum_{l=0}^{M-1} B_j(w_l) \exp\left(ilp\frac{2\pi}{M}\right)
$$
其中
$$
B_j(w_l) = \begin{cases}
    \sum_{m=1}^{j} H_{jm}(w_l)\exp(i\phi_{ml})  & 0\leq l < N \\
    0 & N \leq l < M
\end{cases}\\
w_l = (l-0.5)\Delta w, (l=1, 2, \ldots, N)\\
\Delta t \leq \frac{\pi}{w_{up}}
$$
其中
- $n$ 为待模拟的点的个数；
- $N$ 为充分大的正整数，代表频率的分段数目，取值越大，模拟精度越高；
- $M$ 为时间点的数目，$M = 2N$；
- $\Delta w=w_{up}/N$ 为频率增量；
- $w_{up}$ 为截止频率，即模拟的最高频率；
- $\phi_{ml}$ 为随机相位，通常取为 $0$ 到 $2\pi$ 的均匀分布；
- $H_{jm}(w_l)$ 为 $H(w)$ 中的元素。

本研究中，风的频率区段为 $(0,5]$。生成脉动风场样本周期等于频率增量的周期，即：
$$
T_0 = \frac{2\pi}{\Delta w} = \frac{2\pi N}{w_{u}} \tag{2.22}
$$
通常$T_0$取值应大于600s，即10min，本文取10min作为模拟的时长，由式（2.22）计算得N取为3000，因此需模拟6000个时间点，时间步长为0.1s。