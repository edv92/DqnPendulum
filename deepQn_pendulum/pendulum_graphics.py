from Pendulum_cart import CartPendulum
import numpy as np
import pygame
import sys
from math import sin, cos, pi
from pygame.locals import(
K_LEFT,
K_RIGHT,
K_a,
K_d,
K_ESCAPE,
KEYDOWN,
KEYUP,
K_SPACE,
QUIT
)

class Wagon_graphics():
    def __init__(self, x_wagon, y_wagon):
        self.surf = pygame.image.load("cart_img.png").convert()
        print(self.surf.get_rect().size)
        self.surf.set_colorkey((255,255,255))
        self.rect = self.surf.get_rect(center = (x_wagon, y_wagon))
    def move_wagon(self, x_wagon, y_wagon):
        self.rect = self.surf.get_rect(center = (int(x_wagon*100),y_wagon))


class Pendulum_graphics():
    def __init__(self):
        None

    def getBallPosition(self, x_wagon, y_wagon, alfa, l):
        x_ball_pos = int((x_wagon+l*sin(alfa))*100)
        #print(y_wagon)
        y_ball_pos = int((y_wagon- (l*cos(alfa)*100)))
        return x_ball_pos, y_ball_pos

    def move_pendulum(self, surf,x_wagon, y_wagon, alfa,l):
        x_ball, y_ball = self.getBallPosition(x_wagon, y_wagon, alfa,l)
        pygame.draw.line(surf, (0, 0, 0), (int(x_wagon * 100), y_wagon), (x_ball, y_ball), 1)
        pygame.draw.circle(surf,(0,0,0),(x_ball,y_ball),10)


if __name__ =="__main__":
    # Environment parameters
    m2 = 0.1 # [kg]
    m1 = 1 # [kg]
    l = 1  # [m]
    kt = 0.03  # [kg/rad^2]

    # initialize environment
    alfa_0 = (0.1 * pi) / 180
    w_0 = 0  # [rad/s]
    x2_0 = 3  # [m]
    v2_0 = 0.0  # [m/s]
    theta, w, x2, v2 = alfa_0, w_0, x2_0, v2_0
    f = 0
    dt = 0.02
    x_max = 6
    x_min = 1
    pendulum_obj = CartPendulum(m2, m1, l, kt)


    window_width = 800
    window_height = 600


    pygame.init()
    running = True
    window = pygame.display.set_mode((window_width, window_height))

    pygame.display.set_caption("pendulum environment")

    timer = pygame.time.Clock()

    wagon = Wagon_graphics(window_width * 0.5, window_height * 0.5)
    pendulum = Pendulum_graphics()#window,window_width * 0.5,window_height * 0.5,alfa,l)
    while running:
        theta_next, x2_next, w_next, v2_next = pendulum_obj.sim_one_step_RK4(theta, x2, w, v2, f, dt)
        for event in pygame.event.get():
            if event.type == KEYDOWN:
                if event.key == K_a:
                    f = -10
                if event.key == K_d:
                    f = 10

                if event.key == K_SPACE:
                    theta_next, w_next, x2_next, v2_next = alfa_0, w_0, x2_0, v2_0

                if event.key == K_ESCAPE:
                    running = False

            if event.type == KEYUP:
                f = 0

            elif event.type == QUIT:
                running = False
        #print(f)
        print(f"current x2 is: {x2_next}")

        window.fill((255, 255, 255))
        wagon.move_wagon(x2_next, window_height * 0.5)
        window.blit(wagon.surf, wagon.rect)

        pygame.draw.line(window, (0, 0, 0), (window_width * 0.05, window_height * 0.5 + 31),
                        (window_width * 0.95, window_height * 0.5 + 31), 5)
        pendulum.move_pendulum(window, x2_next, window_height * 0.5, theta_next, l)
        pygame.display.flip()

        theta, w, x2, v2 = theta_next, w_next, x2_next, v2_next

        #print(x2_next)
        timer.tick(50)
    pygame.quit()
    sys.exit()