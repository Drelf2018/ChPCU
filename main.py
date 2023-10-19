import random
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rcParams
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats

rcParams['font.family'] = 'SimHei'

class Result(Enum):
    """
    结果枚举
    """
    Unknown = 0
    Succeed = 1
    Failed = 2


def check10(A: float, S: float, L: float = 15) -> Result:
    """
    判断 10 份数据是否合格
    """
    if A + 2.2 * S <= L:
        return Result.Succeed
    if A + S > L:    
        return Result.Failed
    return Result.Unknown


def check30(A: float, S: float, L: float = 15) -> Result:
    """
    判断 30 份数据是否合格
    """
    if A <= 0.25 * L:
        if A**2 + S**2 <= 0.25 * L:
            return Result.Succeed
        return Result.Failed
    else:
        if A + 3**0.5 * S <= L:
            return Result.Succeed
        return Result.Failed


def checkN(A: float, S: float, L: float = 15):
    """
    判断正态分布是否合格
    """
    r = check10(A, S, L)
    if r != Result.Unknown:
        return r
    return check30(A, S, L)


def get_A_and_S(x: List[float], stop: Optional[int] = None) -> Tuple[float, float]:
    """
    计算所需参数 A 和样本标准差 S

    Args:
        x: 数据列表

        stop: 所需数据个数 使用前 stop 个数据
    """
    if stop is None:
        stop = len(x)
    arr = np.array(x[:stop])
    return abs(arr.mean() - 100), arr.std(ddof=1)


def generate(loc: float, scale: float, L: float = 15, total: int = 100) -> float:
    """
    生成正态分布
    """
    normal = stats.norm.rvs(loc=loc, scale=scale, size=1000)  # type: np.ndarray
    data = normal.tolist()
    succeed = 0  # 有效结果数
    for _ in range(total):  # 重复实验 100 次
        arr = random.sample(data, 30)  # 随机抽取 30 个不同数据
        r = check10(*get_A_and_S(arr, 10), L)  # 用前 10 个数据简单判断
        if r == Result.Unknown:  # 不能直接判断则用所有 30 个数据再次判断
            r = check30(*get_A_and_S(arr), L)
        if r == Result.Succeed:  # 记录结果
            succeed += 1
    if checkN(abs(loc-100), scale, L) == Result.Succeed:
        return succeed / total
    return (total - succeed) / total

if __name__ == "__main__":
    L = 15
    succeed = half = total = 0
    xx = np.linspace(100-L-5, 100+L+5, num=40)
    yy = np.linspace(0, 3*L/4, num=20)
    Z = np.zeros((len(yy), len(xx)))
    accept = 0.95

    for i, loc in enumerate(xx):
        for j, scale in enumerate(yy):
            r = generate(loc, scale, L=L)
            Z[j, i] = 1 - r
            if r > accept:
                succeed += 1
            if r >= 0.5:
                half += 1
            total += 1

    fig = plt.figure()
    ax3: Axes3D = plt.axes(projection='3d')
    ax3.set_xlabel("样本均值")
    ax3.set_ylabel("样本标准差")
    ax3.set_zlabel("误判率")
    ax3.set_title("检查法接受概率：%.2f 正确率：%d/%d(%.2f%%)\n检查法接受概率：%.2f 正确率：%d/%d(%.2f%%)" % (accept, succeed, total, succeed * 100 / total, 0.5, half, total, half * 100 / total))
    X, Y = np.meshgrid(xx, yy)
    ax3.plot_surface(X, Y, Z, cmap="rainbow")
    plt.show()
