import numpy as np
import copy
import pygame

class Grid:
    def __init__(self, params=None):
        if params is None:
            params = {
                'size': (4,4),
                'terminal_state': [(0,0), (3,3)],
                'actions': [[0, 1], [1, 0], [0, -1], [-1, 0]],
                'window_size': (255, 255),
                'transition_reward': -1,
                'ground_color':[0, 0, 0]
            }

        self.params = params
        self.grid_shape = self.params['size']
        self.actions = np.array(params['actions'])

        self.window_size = params['window_size']
        self.grid = self.createEmptyGrid()
        self.terminal_states = params['terminal_state']
        self.reward = params['transition_reward']
        self.ground_color = params['ground_color']
        self.cnt = 0
        self.screen = None

    def restart(self):
        self.grid = self.createEmptyGrid()

    def inGrid(self, pos):
        if 0 <= pos[0] < self.grid_shape[0] \
                and 0 <= pos[1] < self.grid_shape[1]:  # if inside the grid
            return True
        return False

    def createEmptyGrid(self):
        grid = np.zeros(shape=self.grid_shape)
        return grid

    def isterminal(self, pos):
        for i, state in enumerate(self.terminal_states):
            if pos == state:
                return True
        return False


    def render(self, grid= None, policy=None, show_direction=True, values= None):
        if grid is None:
            grid = self.grid

        ground_color = [255, 255 , 255]

        text_color = (0,0,0)
        info_color = (0, 0, 0)
        # This sets the WIDTH and HEIGHT of each grid location
        WIDTH = int(self.window_size[0] / grid.shape[1])
        HEIGHT = int(self.window_size[1] / grid.shape[0])
        # This sets the margin between each cell
        MARGIN = 1


        # Initialize pygame
        pygame.init()

        # Set the HEIGHT and WIDTH of the screen
        WINDOW_SIZE = [self.window_size[0], self.window_size[1]]
        if self.screen == None:
            self.screen = pygame.display.set_mode(WINDOW_SIZE)

        # Set title of screen
        pygame.display.set_caption("Grid_World")

        # Used to manage how fast the screen updates
        clock = pygame.time.Clock()

        font = pygame.font.Font('freesansbold.ttf', 15)
        info_font = pygame.font.Font('freesansbold.ttf', 10)


        # -------- Main Program Loop -----------
        # while not done:
        for event in pygame.event.get():  # User did something
            if event.type == pygame.QUIT:  # If user clicked close
                exit(0)


        # Set the screen background
        self.screen.fill((0,0,0))


        # Draw the grid
        # for x in range(self._grid_shape[0]):
        #     for y in range(self._grid_shape[1]):
        for x, y in np.ndindex(self.grid_shape):
            color = ground_color


            pygame.draw.rect(self.screen,
                             color,
                             [(MARGIN + WIDTH) * y + MARGIN,
                              (MARGIN + HEIGHT) * x + MARGIN,
                              WIDTH,
                              HEIGHT])
            if self.isterminal((x, y)):
                pygame.draw.rect(self.screen,
                             (0,0,0),
                             [(MARGIN + WIDTH) * y + MARGIN,
                              (MARGIN + HEIGHT) * x + MARGIN,
                              WIDTH,
                              HEIGHT])
            if  True:
                # showing values only for 4 basic actions
                up_left_corner = [(MARGIN + WIDTH) * y + MARGIN,
                                  (MARGIN + HEIGHT) * x + MARGIN]
                up_right_corner = [(MARGIN + WIDTH) * y + MARGIN + WIDTH,
                                  (MARGIN + HEIGHT) * x + MARGIN]
                down_left_corner = [(MARGIN + WIDTH) * y + MARGIN,
                                   (MARGIN + HEIGHT) * x + MARGIN + HEIGHT]
                down_right_corner = [(MARGIN + WIDTH) * y + MARGIN + WIDTH,
                                    (MARGIN + HEIGHT) * x + MARGIN + HEIGHT]
                center = [(up_right_corner[0] + up_left_corner[0]) // 2,
                          (up_right_corner[1] + down_right_corner[1]) // 2]




                pygame.draw.rect(self.screen, (0,0,0),
                                 [up_left_corner[0], up_left_corner[1],
                                  (down_right_corner[0]-up_left_corner[0]),(down_right_corner[1]-up_left_corner[1])],
                                 1)

                state_value_text = info_font.render(str(round(grid[x,y],1)), True, info_color)
                self.screen.blit(state_value_text,
                                 (center[0]-8, center[1]-4))

                if policy is not None and show_direction is True:
                    p = policy[x,y]
                    if p[0] != 0:
                        pygame.draw.rect(self.screen,(0, 0, 0),
                                         [center[0]+15,center[1],
                                          WIDTH//2 -18 ,2*MARGIN])
                        pygame.draw.circle(self.screen,(0, 0, 0),
                                           [center[0] + WIDTH//2 -5, center[1]+MARGIN], 3, 3)
                    if p[1] != 0:
                        pygame.draw.rect(self.screen,(0, 0, 0),
                                         [center[0],center[1]+15,
                                          2*MARGIN ,HEIGHT//2 -18])
                        pygame.draw.circle(self.screen, (0, 0, 0),
                                           [center[0] + MARGIN,center[1] + HEIGHT // 2 - 5, ], 3, 3)
                    if p[2] != 0:
                        pygame.draw.rect(self.screen,(0, 0, 0),
                                         [center[0]-15 - (WIDTH//2 -18),center[1],
                                          WIDTH//2 -18 ,2*MARGIN])
                        pygame.draw.circle(self.screen, (0, 0, 0),
                                           [center[0] - (WIDTH // 2 - 5), center[1] + MARGIN], 3, 3)
                    if p[3] != 0:
                        pygame.draw.rect(self.screen, (0, 0, 0),
                                         [center[0], center[1] - 15 -(HEIGHT // 2 - 18),
                                          2 * MARGIN, HEIGHT // 2 - 18])
                        pygame.draw.circle(self.screen, (0, 0, 0),
                                           [center[0] + MARGIN,center[1] - (HEIGHT // 2 - 5), ], 3, 3)
        #




        # Limit to 60 frames per second
        clock.tick(60)


        # Go ahead and update the screen with what we've drawn.
        pygame.display.flip()
        self.cnt += 1
        # pygame.image.save(self.screen, "image/screenshot"+str(self.cnt)+".jpeg")
        # print("loop")
        while True:
            flag = False
            for e in pygame.event.get():
                if e.type == pygame.KEYDOWN:

                    flag=True
            if flag: break

class DP:
    def __init__(self, init_policy=[0.25, 0.25, 0.25, 0.25], grid_info=None, gamma=1, theta = 0.000000000000001):
        self.init_policy = init_policy
        self.grid_info = grid_info
        self.grid = self.make_grid()

        self.p = np.zeros((self.grid.grid_shape[0],self.grid.grid_shape[1],4))
        self.p[:,:] = init_policy
        self.gamma = gamma
        self.theta = theta


    def policy_eval(self, p=None, main_grid=None, theta=None, debug=None, params=None):
        if p is None: p = self.p
        if main_grid is None: main_grid=self.grid
        if theta is None: theta = self.theta

        grid_k0 = np.copy(main_grid.grid)
        grid_k1 = np.copy(main_grid.grid)


        while True:
            delta = 0
            # main_grid.render(policy=self.p, grid=grid_k1)
            for i, j in np.ndindex(main_grid.grid_shape):

                if main_grid.isterminal((i,j)):
                    continue
                value = 0

                for k, a in enumerate(main_grid.actions):
                    s_prime = (i, j) + a

                    r = -1
                    # if main_grid.isterminal((s_prime[0], s_prime[1])): r=0
                    if not main_grid.inGrid(s_prime): s_prime = (i,j)
                    value += p[i,j, k] * (r + self.gamma * grid_k0[s_prime[0], s_prime[1]])

                grid_k1[i, j] = value

                if np.abs(grid_k1[i,j]-grid_k0[i,j]) > delta:
                    delta = np.abs(grid_k1[i,j]-grid_k0[i,j])
            grid_k0 = np.copy(grid_k1)


            if delta < theta: break
        return grid_k1

    def policy_iter(self):
        while True:
            # self.grid.render(policy=self.p)
            grid = self.policy_eval()
            self.grid.grid = np.copy(grid)
            new_policy, stable_policy = self.policyUpdate(self.p, self.grid)
            self.grid.render(policy=self.p)
            if stable_policy:
                break
            self.p = np.copy(new_policy)
            self.grid.render(policy=self.p)

        return self.grid, self.p

    def policyUpdate(self, policy, grid):
        new_policy = np.zeros_like(policy).astype(float)
        stable_policy = True
        for i, j in np.ndindex(grid.grid_shape):
            value = np.zeros(4)
            for k, a in enumerate(grid.actions):
                s_prime = (i, j) + a
                r = -1
                if grid.isterminal([s_prime[0], s_prime[1]]): r = 0
                if not grid.inGrid(s_prime): s_prime = (i, j)
                value[k] = (r + self.gamma * grid.grid[s_prime[0], s_prime[1]])

            max_indices = np.where(value == np.max(value))[0]
            max_count = len(max_indices)
            new_policy[i,j,max_indices] = 1/max_count
            if stable_policy and not np.all(policy[i,j] == new_policy[i,j]):
                stable_policy = False

        return new_policy, stable_policy

    def valueiteration(self, p=None, main_grid=None, theta=None, debug=None):
        if p is None: p = self.p
        if main_grid is None: main_grid = self.grid
        if theta is None: theta = self.theta

        grid_k0 = np.copy(main_grid.grid)
        grid_k1 = np.copy(main_grid.grid)
        policy = np.copy(self.p)
        while True:
            delta = 0
            for i, j in np.ndindex(main_grid.grid_shape):

                if main_grid.isterminal((i, j)):
                    continue
                values = np.zeros(4)
                for k, a in enumerate(main_grid.actions):
                    s_prime = (i, j) + a

                    r = -1
                    # if main_grid.isterminal([s_prime[0], s_prime[1]]): r = 0
                    if not main_grid.inGrid(s_prime): s_prime = (i, j)

                    values[k] = (r + self.gamma * grid_k0[s_prime[0], s_prime[1]])
                    max_indices = np.where(values == np.max(values))[0]
                    max_count = len(max_indices)
                    policy[i, j] = np.zeros(4)
                    policy[i, j, max_indices] = 1 / max_count

                # self.grid.render(policy=policy, grid=grid_k1)

                grid_k1[i, j] = values[max_indices[0]]

                self.grid.render(policy=policy, grid=grid_k1)






                if np.abs(grid_k1[i, j] - grid_k0[i, j]) > delta:
                    delta = np.abs(grid_k1[i, j] - grid_k0[i, j])
                grid_k0 = np.copy(grid_k1)
            # self.grid.render(policy=policy, grid=grid_k1)
            if delta < theta: break

        main_grid.grid = grid_k1
        policy, _ = self.policyUpdate(self.p, main_grid)

        return main_grid, policy

    def make_grid(self):
        return Grid()





if __name__ == '__main__':
    dp = DP()
    grid1, policy1 = dp.policy_iter()
    grid2, policy2 = dp.valueiteration()


    print(np.all(policy1==policy2))