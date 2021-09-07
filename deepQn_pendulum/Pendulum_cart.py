import numpy as np
import matplotlib.pyplot as plt
from math import sin
from math import cos
from math import pi
class CartPendulum:

    def __init__(self, m2, m1, l, kt):
        # parameters
        self.m2 = m2
        self.m1 = m1
        self.l = l
        self.kt = kt
        self.g = 9.8


    def d_w(self, w, alfa, f):
        d_w = ((self.m2 + self.m1) / (self.m2 ** (2) * self.l ** (2) * (cos(alfa)) ** (2) - (self.m2 ** (2) * self.l ** (2)) - (self.m2 * self.m1 * self.l ** (2)))) \
              * (self.kt * w - self.m2 * self.g * self.l * sin(alfa)) \
              + (cos(alfa) / (self.m2 * self.l * (cos(alfa)) ** (2) - self.m2 * self.l - self.m1 * self.l)) * (f + w ** (2) * self.m2 * self.l * sin(alfa))
        return d_w

    def d_v2(self, w, alfa, f):
        d_v2 = (1 / (self.m2 * self.l * cos(alfa))) * (self.m2 * self.g * self.l * sin(alfa) - self.kt * w \
                                                       - self.m2 * self.l ** (2) * (((self.m2 + self.m1) / (self.m2 ** (2) * self.l ** (2) * (cos(alfa)) ** (2) - (self.m2 ** (2) * self.l ** (2)) - (self.m2 * self.m1 * self.l ** (2)))) \
                                                                                    * (self.kt * w - self.m2 * self.g * self.l * sin(alfa)) \
                                                                                    + (cos(alfa) / (self.m2 * self.l * (cos(alfa)) ** (2) - self.m2 * self.l - self.m1 * self.l)) * (f + w ** (2) * self.m2 * self.l * sin(alfa))))
        return d_v2


    def sim_one_step_RK4(self,alfa,x2, w, v2,f_i,dt):

        # first terms/start point
        d_alfa_start = w
        d_x2_start = v2
        d_w_start = self.d_w(w, alfa, f_i)
        d_v2_start = self.d_v2(w, alfa, f_i)

        # forward euler on midpoint
        d_alfa_forward_mid = w + (1 / 2) * d_w_start * dt
        d_x2_forward_mid = v2 + (1 / 2) * d_v2_start * dt
        d_w_forward_mid = self.d_w(w + (1 / 2) * d_w_start * dt, alfa + (1 / 2) * d_alfa_start * dt, f_i)
        d_v2_forward_mid = self.d_v2(w + (1 / 2) * d_w_start * dt, alfa + (1 / 2) * d_alfa_start * dt, f_i)

        # backward euler on midpoint
        d_alfa_backward_mid = w + (1 / 2) * d_w_forward_mid * dt
        d_x2_backward_mid = v2 + (1 / 2) * d_v2_forward_mid * dt
        d_w_backward_mid = self.d_w(w + (1 / 2) * d_w_forward_mid * dt, alfa + (1 / 2) * d_alfa_forward_mid * dt, f_i)
        d_v2_backward_mid = self.d_v2((w + (1 / 2) * d_w_forward_mid * dt), alfa + 1 / 2 * d_alfa_forward_mid * dt, f_i)

        # CN on endpoint
        d_alfa_CN_end = w + d_w_backward_mid * dt
        d_x2_CN_end = v2 + d_v2_backward_mid * dt
        d_w_CN_end = self.d_w(w + d_w_backward_mid * dt, alfa + d_alfa_backward_mid * dt, f_i)
        d_v2_CN_end = self.d_v2(w + d_w_backward_mid * dt, alfa + d_alfa_backward_mid * dt, f_i)

        alfa_next = alfa + (dt / 6) * (d_alfa_start + 2 * d_alfa_forward_mid + 2 * d_alfa_backward_mid + d_alfa_CN_end)
        x2_next = x2 + (dt / 6) * (d_x2_start + 2 * d_x2_forward_mid + 2 * d_x2_backward_mid + d_x2_CN_end)
        w_next = w + (dt / 6) * (d_w_start + 2 * d_w_forward_mid + 2 * d_w_backward_mid + d_w_CN_end)
        v2_next = v2 + (dt / 6) * (d_v2_start + 2 * d_v2_forward_mid + 2 * d_v2_backward_mid + d_v2_CN_end)


        return np.array([alfa_next, x2_next, w_next, v2_next])

    """
    def getReward(self, theta, theta_lim, x, x_lim, move_reward, border_reward):
        if theta >= theta_lim or theta <= -theta_lim:
            reward = border_reward
        elif x >= x_lim or x <= -x_lim:
            reward = border_reward
        else:
            reward = move_reward
        return reward
    """

    def getReward(self, theta, theta_lim, x, x_lim, move_reward, border_reward):
        if x >= x_lim or x <= -x_lim:
            return border_reward
        #print(abs(theta % (pi)))
        if abs(theta % (2*pi))>= pi:
            return 0.1*(pi-((2*pi)-abs(theta % (2*pi))))
        return 0.1*(pi -abs(theta % (pi)))
        #if abs(theta % (2*pi))< 100*(pi/180) or abs(theta % (2*pi)) > 260*(pi/180):

            #return move_reward
        #return border_reward
        #return abs(0.1/)#pi*0.1 - (abs(theta) % (2*pi))*0.1
        #if theta >= theta_lim or theta <= -theta_lim:
        #    return border_reward
        #return move_reward
        #else:
         #   move_reward = move_reward - (abs(theta) % (2*pi))
          #  return move_reward*0.1



    def getTwoStateReward(self, alfa, alfa_lim, move_reward, border_reward):
        if alfa >= alfa_lim or alfa <= -alfa_lim:
            reward = border_reward
        else:
            reward = move_reward
        return reward

    def getXposReward(self, x_pos, x_lim, border_reward, move_reward):
        if x_pos > x_lim or x_pos< -x_lim:
            reward = border_reward
        else:
            reward = move_reward
        return reward


    def getForce(self, action, applied_force):

        if action.item() == 0:
            return -applied_force
        return applied_force

if __name__ == "__main__":
    p_obj = CartPendulum(0.1,1,1,0.03)
    theta = 0
    x = 0
    w = 0
    v = 0
    #print((abs(-(720)*(pi/180)) % (2*pi))*180/pi)
    #theta,x,w,v = p_obj.sim_one_step_RK4(360*(pi/180),0,0.5,0,10,0.02)
    #print(theta*(180/pi))
    test = np.arange(-720,360)* (pi/180)
    for i in range(len(test)):
        r =  p_obj.getReward(test[i], 90, 1, 1.5, 0.1, -0.1)
        print(r)

