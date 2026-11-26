# ros2_rl_nav — 用强化学习训练 ROS2 导航策略

用 PPO 训练一台差速小车（TurtleBot3 Burger 尺度）在随机障碍地图中避障并到达目标，
全部在本机 Windows + GPU 上完成：训练、评估、可视化。

## 与 ROS2 的对应关系（学会迁移到真机/Gazebo）

| 本项目 | ROS2 中对应 | 说明 |
|---|---|---|
| `scan` 36×[0,1] | `/scan` (sensor_msgs/LaserScan) | 360° 激光按 10° 降采样，距离归一化（÷3.5m） |
| 目标相对位姿 sin/cos/d | `/goal_pose` (PoseStamped) | 目标转到车体坐标系 |
| 上一步动作 | — | 让控制更平滑 |
| 动作 (v̂, ŵ) | `/cmd_vel` (geometry_msgs/Twist) | v∈[0, 0.22] m/s，ω∈[-2.84, 2.84] rad/s |

## 项目结构

```
rl_nav/world.py       2D 世界：围墙 + 圆形/矩形障碍，向量化激光射线投射、碰撞检测
rl_nav/scenarios.py   场景生成器（随机地图+起终点，BFS 可达性校验）+ 固定评估集
rl_nav/nav_env.py     Gymnasium 环境（步长 0.25s，奖励=靠近目标-碰撞/超时罚+到达奖）
rl_nav/render.py      matplotlib 渲染与 GIF 保存
rl_nav/policy.py      加载模型 + 归一化统计
train.py              PPO 训练（8 并行环境，VecNormalize，TensorBoard，自动评估存最优）
evaluate.py           50 个未见场景批量评估：成功率/碰撞率/步数/最小间距 + 轨迹图 + GIF
play.py               实时窗口观看小车导航（或无头模式存 GIF）
data/                 eval_scenarios.json（固定评估集）、评估结果、GIF
runs/                 模型 checkpoint、最优模型、vecnormalize.pkl
```

## 快速开始

```bash
cd ros2_rl_nav
# 1) 训练（约 5-15 分钟，500k 步）
.venv/Scripts/python.exe train.py
# 2) 仿真测试：50 个未见场景评估
.venv/Scripts/python.exe evaluate.py
# 3) 实时观看
.venv/Scripts/python.exe play.py            # 弹出窗口实时看
.venv/Scripts/python.exe play.py --no-show  # 无窗口，存 GIF 到 data/
# 4) 训练曲线
.venv/Scripts/python.exe -m tensorboard.main --logdir runs/tb
```

## 训练数据说明

RL 的"数据"是策略与环境交互产生的轨迹（ experience ），由训练过程自动收集。
本项目提前准备的"数据"有两部分：

1. **场景生成器**：随机圆形+矩形障碍、起点终点间距 ≥4m，并用 BFS 在膨胀栅格上
   校验存在无碰撞路径（保证没有不可达的死题）。
2. **固定评估集** `data/eval_scenarios.json`：种子 2024 生成 50 个场景，训练中
   从未见过，用于客观评估泛化能力。

## 奖励设计（可自己改着玩）

- 每步 `5 × (上步距离 - 当前距离)`：鼓励靠近目标
- 到达目标 `+10`；碰撞 `-10`（回合结束）
- 每步 `-0.01` 时间罚：鼓励走短路径

## 下一步：接入真实 ROS2 仿真（Gazebo）

写一个 ROS2 节点桥接即可复用本策略：
订阅 `/scan` → 填充 36 束归一化观测；订阅目标 → 算相对位姿；
`model.predict(obs)` → 发布 `Twist(v, w)` 到 `/cmd_vel`。
linorobot2 的 Docker + Gazebo 栈可以直接挂这个节点做闭环测试。
