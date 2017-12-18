import pyglet
import numpy as np
# pyglet 可视化模块

class ArmEnv(object):
    viewer = None
    dt = 0.1 # refresh rate 转动角
    action_bound = [-1, 1]
    goal = {'x':100., 'y':100., 'l':40} # 蓝色球的位置
    state_dim = 2
    action_dim = 2 # 两个action操作

    def __init__(self):
        # (2,2)
        # 手臂的基本信息
        # 手臂长度和手臂的转动角
        self.arm_info = np.zeros(
            2, dtype=[('l', np.float32), ('r', np.float32)])
        self.arm_info['l'] = 100
        self.arm_info['r'] = np.pi/6

    def step(self, action):
        done = False
        r = 0.
        action = np.clip(action, *self.action_bound)
        self.arm_info['r'] += action * self.dt
        self.arm_info['r'] %= np.pi * 2 #normalize

    def reset(self):
        self.arm_info['r'] = 2 * np.pi * np.random.rand(2)
        return self.arm_info['r']


    def render(self):
        if self.viewer is None:
            self.viewer = Viewer(self.arm_info, self.goal)
        self.viewer.render()

    def sample_aciton(self):
        return np.random.rand(2) - 0.5


class Viewer(pyglet.window.Window):
    # 画出手臂
    bar_thc = 5

    def __init__(self, arm_info, goal):
        # vsync 刷新频率是否要和显示器一样。 False程序多快就刷新多快
        super(Viewer, self).__init__(width=400, height=400, resizable=False, caption='Arm', vsync=False)
        pyglet.gl.glClearColor(1, 1, 1, 1)
        self.arm_info = arm_info
        self.center_coord = np.array([200, 200])

        # 将构件放入一个batch中去
        self.batch = pyglet.graphics.Batch()    # display whole batch at once
        self.point = self.batch.add(
            4, pyglet.gl.GL_QUADS, None,    # 4 corners
            ('v2f', [goal['x'] - goal['l'] / 2, goal['y'] - goal['l'] / 2,                # location
                     goal['x'] - goal['l'] / 2, goal['y'] + goal['l'] / 2,
                     goal['x'] + goal['l'] / 2, goal['y'] + goal['l'] / 2,
                     goal['x'] + goal['l'] / 2, goal['y'] - goal['l'] / 2]),
            ('c3B', (86, 109, 249) * 4))    # color
        self.arm1 = self.batch.add(
            4, pyglet.gl.GL_QUADS, None,
            ('v2f', [250, 250,                # location
                     250, 300,
                     260, 300,
                     260, 250]),
            ('c3B', (249, 86, 86) * 4,))    # color
        self.arm2 = self.batch.add(
            4, pyglet.gl.GL_QUADS, None,
            ('v2f', [100, 150,              # location
                     100, 160,
                     200, 160,
                     200, 150]), ('c3B', (249, 86, 86) * 4,))

    # 刷新并且展现在屏幕上
    def render(self):
        self._update_arm()
        self.switch_to()
        self.dispatch_events()
        self.dispatch_event('on_draw')
        self.batch.draw()

    # 刷新手臂等位置
    def on_draw(self):
        self.clear()
        self.batch.draw()

    # 更新手臂的位置信息
    def _update_arm(self):
        # 三角函数的运算
        (a1l, a2l) = self.arm_info['l']     # radius, arm length
        (a1r, a2r) = self.arm_info['r']     # radian, angle
        a1xy = self.center_coord            # a1 start (x0, y0)
        a1xy_ = np.array([np.cos(a1r), np.sin(a1r)]) * a1l + a1xy   # a1 end and a2 start (x1, y1)
        a2xy_ = np.array([np.cos(a1r+a2r), np.sin(a1r+a2r)]) * a2l + a1xy_  # a2 end (x2, y2)

        a1tr, a2tr = np.pi / 2 - self.arm_info['r'][0], np.pi / 2 - self.arm_info['r'].sum()
        xy01 = a1xy + np.array([-np.cos(a1tr), np.sin(a1tr)]) * self.bar_thc
        xy02 = a1xy + np.array([np.cos(a1tr), -np.sin(a1tr)]) * self.bar_thc
        xy11 = a1xy_ + np.array([np.cos(a1tr), -np.sin(a1tr)]) * self.bar_thc
        xy12 = a1xy_ + np.array([-np.cos(a1tr), np.sin(a1tr)]) * self.bar_thc

        xy11_ = a1xy_ + np.array([np.cos(a2tr), -np.sin(a2tr)]) * self.bar_thc
        xy12_ = a1xy_ + np.array([-np.cos(a2tr), np.sin(a2tr)]) * self.bar_thc
        xy21 = a2xy_ + np.array([-np.cos(a2tr), np.sin(a2tr)]) * self.bar_thc
        xy22 = a2xy_ + np.array([np.cos(a2tr), -np.sin(a2tr)]) * self.bar_thc

        self.arm1.vertices = np.concatenate((xy01, xy02, xy11, xy12))
        self.arm2.vertices = np.concatenate((xy11_, xy12_, xy21, xy22))

if __name__ == "__main__":
    env = ArmEnv()
    while True:
        s = env.reset()
        for i in range(40):
            env.render()
            env.step(env.sample_aciton())
