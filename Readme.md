# RM-Vision03 弹丸轨迹拟合

## 最终结果

| 参数 | 数值   |
| ---- | ------ |
| Vx₀ | 252.36 |
| Vy₀ | 346.91 |
| g    | 497.88 |
| k    | 0.06   |

## 项目结构

```
Task3/
├── main.cpp                    # 主程序：弹丸检测与轨迹拟合
├── CMakeLists.txt             # CMake配置文件
├── video.mp4                  # 弹丸飞行视频
├── 任务要求.png               # 任务说明文档
├── trajectory_result.png      # 轨迹拟合结果图
├── build/                     # CMake构建目录
│   ├── compile_commands.json  # 编译数据库
│   └── Task3                  # 可执行文件
└── Readme.md                  # 项目说明文档
```

> 生成的轨迹图保存在项目根目录中

---

# 任务简介：基于Ceres库的弹丸轨迹拟合

## 思路

看了题目之后大致思路如下:

1. 定义残差
2. 构建优化问题（把所给的函数式表达）
3. 读取视频
   - 框定有效时长数据
   - 按照60帧的给定条件读取视频
   - 建立坐标系，定义x，y（目前还不清楚单位怎么处理）
4. 参数约束（$g \in (100,1000)$且$k \in (0.01,1)$）
5. 选择求解器进行优化问题
6. （可选）用鲁棒核函数处理异常值

### 弹道模型

根据任务要求，弹丸轨迹满足以下物理模型：

- $\Delta t = t - t_0$ (其中 $t_0 = 0$)
- $x(t) = x_0 + \frac{v_{x0}}{k}(1 - e^{-k\Delta t})$
- $y(t) = y_0 + \frac{v_{y0} + g/k}{k}(1 - e^{-k\Delta t}) - \frac{g}{k}\Delta t$

### 参数拟合

使用Ceres优化库拟合以下参数：

- $v_{x0}, v_{y0}$：初始速度分量 (px/s)
- $g$：重力加速度 (px/s²)，范围 [100, 1000]
- $k$：阻力系数 (1/s)，范围 [0.01, 1]

---

# 日志♿：

> 这次任务主要涉及计算机视觉、数值优化和物理建模，从零开始实现了完整的弹丸轨迹拟合系统 💦

---

## 项目初期

### 弹道模型理解

刚开始看到这个物理模型时有点懵，特别是那个指数项 $e^{-k\Delta t}$ 和复杂的Y方向公式。后来查了资料才知道这是考虑空气阻力的弹道方程，其中k是阻力系数。

> **物理意义**：
>
> - 当k=0时，退化为经典抛物线运动
> - 当k>0时，阻力使弹丸逐渐减速
> - 指数项 $e^{-k\Delta t}$ 表示速度的指数衰减

---

### 轮廓分析

找到最大轮廓作为弹丸，计算质心：

```cpp
cv::Moments moments = cv::moments(contours[max_idx]);
double cx = moments.m10 / moments.m00;
double cy = moments.m01 / moments.m00;
```

---

## 坐标系转换

### 问题发现

刚开始直接用OpenCV的像素坐标进行拟合，结果轨迹完全不对。后来意识到需要坐标转换：

- **OpenCV坐标系**：原点在左上角，Y轴向下
- **物理坐标系**：原点在左下角，Y轴向上

### 转换实现

```cpp
// 检测时：OpenCV → 物理坐标系
double x_phys = detected.x;
double y_phys = frameHeight - detected.y; // Y轴翻转

// 绘制时：物理坐标系 → OpenCV
int imgY = frameHeight - static_cast<int>(y); // Y轴翻转
```

> **关键点**：Y轴翻转是核心，其他坐标保持不变

---

## Ceres优化

### 残差函数定义

这是最核心的部分，需要将物理模型转换为Ceres的残差函数：

```cpp
struct TrajModelCost {
    template <typename T>
    bool operator()(const T* const params, T* residuals) const {
        const T vx0 = params[0];
        const T vy0 = params[1];
        const T g   = params[2];
        const T k   = params[3];

        const T expkt = ceres::exp(-k * T(t));
        const T x = T(x0) + (vx0 / k) * (T(1.0) - expkt);
        const T y = T(y0) + ((vy0 + g / k) / k) * (T(1.0) - expkt) - (g / k) * T(t);

        residuals[0] = x - T(xi);
        residuals[1] = y - T(yi);
        return true;
    }
};
```

### 参数边界约束

根据任务要求设置参数边界：

```cpp
problem.SetParameterLowerBound(params, 2, 100.0);   // g ≥ 100
problem.SetParameterUpperBound(params, 2, 1000.0);  // g ≤ 1000
problem.SetParameterLowerBound(params, 3, 0.01);    // k ≥ 0.01
problem.SetParameterUpperBound(params, 3, 1.0);     // k ≤ 1.0
```

### 鲁棒核函数

使用HuberLoss处理异常值：

```cpp
problem.AddResidualBlock(cost, new ceres::HuberLoss(3.0), params);
```

---

## 可视化优化

### 轨迹图绘制

刚开始创建的轨迹图尺寸固定，后来改为使用原视频尺寸：

```cpp
cv::Mat trajectoryImg(frameHeight, frameWidth, CV_8UC3, cv::Scalar(0, 0, 0));
```

### 坐标转换修正

发现绘制时的坐标转换有问题，Y轴方向反了：

```cpp
// 修正前（错误）
int imgY = static_cast<int>((y - minY) / (maxY - minY) * height);

// 修正后（正确）
int imgY = static_cast<int>((maxY - y) / (maxY - minY) * height);
```

---

## 最终成果

### 输出格式

程序输出一行结果：

`vx0: 252.358 vy0: 346.914 g: 497.884 k: 0.0645409`

---

至此全部任务圆满完成✅！
