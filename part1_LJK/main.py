
from part1.env import ArmEnv
from part1.rl import DDPG

MAX_EPISODES = 500
MAX_EP_STEPS = 200

# set env 生成环境
env = ArmEnv()
s_dim = env.state_dim # 设置参数
a_dim = env.action_dim
a_bound = env.action_bound # 输出范围

# set RL method 引用 RL 方法
# DDPG 建立神经网络
rl = DDPG(a_dim, s_dim, a_bound)

# start training 开始循环
for i in range(MAX_EPISODES): # 循环次数
    s = env.reset() # 提取功能，初始化回合设置
    for j in range(MAX_EPISODES): # 每回合步数
        env.render() # 展示
        a = rl.choose_action(s) # s输入 a输出
        s_, r, done = env.step(a) # 环境反馈

        rl.store_transition(s, a, r, s_) # 放入记忆库当中，离线学习

        if rl.memory_full: # 记忆存满
            rl.learn() # 开始学习

        s = s_ # 修改s 开始下一步循环

# summary:

"""
env should have at least:
env.reset()
env.render()
env.step()

while RL should have at least:
rl.choose_action()
rl.store_transition()
rl.learn()
rl.memory_full
"""




