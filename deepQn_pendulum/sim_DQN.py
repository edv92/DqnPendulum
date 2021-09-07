from Pendulum_cart import CartPendulum
from pendulum_graphics import Wagon_graphics, Pendulum_graphics
from dqn_agent import Q_Agent
from dqn_agent import NeuralNetworkz
import pygame
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from math import sin
from math import cos
from math import pi
if __name__ == "__main__":





    # Environment parameters
    m2 = 0.1  # [kg]
    m1 = 1.0  # [kg]
    l = 1.0  # [m]
    kt = 0.03  # [kg/rad^2]

    applied_f = 10
    states_tensor = None
    states_k = np.zeros(4)
    states_k1 = np.zeros(4)
    dt = 0.02



    # initialize pygame display
    window_width = 800
    window_height = 600

    pygame.init()
    running = True
    window = pygame.display.set_mode((window_width, window_height))

    pygame.display.set_caption("q_table_328.npy")

    timer = pygame.time.Clock()

    wagon = Wagon_graphics(window_width * 0.5, window_height * 0.5)
    pendulum = Pendulum_graphics()  # window,window_width * 0.5,window_height * 0.5,alfa,l)
    q_agent_obj = Q_Agent()


    theta_lim_deg = 15
    theta_max = theta_lim_deg * (pi / 180)


    x_lim = 3


    pendulum_obj = CartPendulum(m2, m1, l, kt)
    q_agent_obj = Q_Agent()

    num_obs = 4
    num_actions = 2
    num_neurons = 30

    deepQModel = NeuralNetworkz(num_obs, num_actions, num_neurons)
    deepQModel.load_state_dict(torch.load("models/404716.pth"))
    deepQModel.eval()

    #"models/500451.pth"

    step_counter = 0
    states_k= np.random.uniform(-0.1 * (pi / 180), 0.1 * (pi / 180), 4)
    #state_tensor = torch.tensor(states_k)
    #print(state_tensor)
    while running:
        pygame.event.get()
        step_counter +=1
        state_tensor = torch.from_numpy(states_k).float()
        action = torch.tensor([[torch.argmax(deepQModel(state_tensor))]])
        #print(action)
        f = pendulum_obj.getForce(action, applied_f)

        if states_k[2] >= 0:
            states_k[2] += np.random.uniform(0,0.1)
        else:
            states_k[2] += np.random.uniform(-0.1, 0)

        states_k1[0], states_k1[1], states_k1[2], states_k1[3] = pendulum_obj.sim_one_step_RK4(states_k[0], states_k[1], states_k[2], states_k[3], f, dt)
        x2_next = states_k1[1]+3.5
        theta_next = states_k1[0]
        window.fill((255, 255, 255))
        wagon.move_wagon(x2_next, window_height * 0.5)
        window.blit(wagon.surf, wagon.rect)

        pygame.draw.line(window, (0, 0, 0), (window_width * 0.05, window_height * 0.5 + 31),
                         (window_width * 0.95, window_height * 0.5 + 31), 5)
        pygame.draw.line(window, (205, 0, 0), (205, window_height * 0.5 + 31),
                         (195, window_height * 0.5 + 31), 5)
        pygame.draw.line(window, (255, 0, 0), (505, window_height * 0.5 + 31),
                         (495, window_height * 0.5 + 31), 5)
        pendulum.move_pendulum(window, x2_next, window_height * 0.5, theta_next, l)
        pygame.display.flip()

        states_k[0], states_k[1], states_k[2], states_k[3] = states_k1[0], states_k1[1], states_k1[2], states_k1[3]
        #print(f"Angle is now: {states_k[0] * (180 / pi)}")
        timer.tick(50)



        if step_counter > 1000:
            step_counter = 0
            states_k[0], states_k[1], states_k[2], states_k[3] = np.random.uniform(-0.1 * (pi / 180), 0.1 * (pi / 180), 4)
            #states_k[0] = pi#np.random.uniform(-20 * (pi / 180), 20 * (pi / 180))
            #states_k[1] = 0#np.random.uniform(-1, 1)
            #states_k[2] = 0
            #states_k[3] = 0
            print(f"starting angle is: {states_k[0]*(180/pi)}")
            #if theta_next*(180 / pi)<theta_lim_deg and theta_next*(180 / pi)>-theta_lim_deg:
                #print("current episode was a success")


    pygame.quit()
    sys.exit()