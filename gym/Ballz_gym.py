import gym
from gym import spaces

import numpy as np
from JSAnimation.IPython_display import display_animation
from matplotlib import animation
from IPython.display import display
import matplotlib.pyplot as plt

import time
# Constants
SCREEN_WIDTH = 350
SCREEN_HEIGHT = 450
SCREEN_TITLE = "Ballz v1"
BLOCK_SCALING = 0.6

# Initital 2D Array representings
blockArray = np.array([[0, 0, 0, 0, 0, 1, 0],
              [0, 0, 1, 0, 0, 0, 1],
              [0, 0, 1, 0, 0, 0, 1],
              [0, 0, 1, 0, 0, 0, 1],
              [0, 0, 0, 0, 0, 0, 0],
              [0, 0, 1, 0, 0, 0, 1],
              [0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0]])

class Ballz(gym.Env):
    
    def __init__(self):
        '''
        Initialize the environment
        '''

        self.ball_speed = 1
        self.max_x, self.min_x = 3.5, -3.5
        self.max_y, self.min_y = 4.5, -4.5
        self.delta_time = 0.1
        self.action_space = spaces.Box(low=0, high=np.pi, dtype=np.float32, shape=(1,))
        self.observation_space = spaces.Box(low=0, high=255, dtype=np.uint8, shape=(10,9))
        self.viewer = None
        self.state = None    # (blockArray, ball_pos)
        self.pos_list = []
        
        world_width = self.max_x - self.min_x
        world_height = self.max_y - self.min_y
        self.scale_x = SCREEN_WIDTH/world_width
        self.scale_y = SCREEN_HEIGHT/world_height
        
        self.reset()
        
    def reset(self):
        '''
        Reset the environment
        '''
        
        self.state = {'blocks':blockArray, 'pos':[0,self.min_y]}
        self.pos_list = [[0,-4]]
        
    def coord2Index(self, position):
            '''
            Transform coordinate into index of block array
            '''
            assert len(position) == 2
            
            index = np.zeros(2)
            index[0], index[1] = position[0], position[1]*-1   # Flip y axis
            index[0], index[1] = index[1]-self.min_y, index[0]-self.min_x # Shift all the index, e.g. (min_x,min_y)->(0,0)
        
            return index.astype('int')
    
    def index2Coord(self, index):
            '''
            Transform index of block array into coordinate
            '''
            assert len(index) == 2
        
            position = np.zeros(2)
            # (row,col) of index corresponds to (y,x) of coord
            position[0], position[1] = index[1]+self.min_x, index[0]+self.min_y   # Shift all the index, e.g. (0,0)->(min_x,min_y)
            position[0], position[1] = position[0]+0.5, position[1]+0.5 # Make it the center of block
            position[1] *= -1                                           # Flip the y axis
        
            return position
        
        
    def step(self, action):
        '''
        Action changes environment
        '''
        assert 0<=action<=np.pi
        
        print(r'Angle: %.5f pi'%(action/np.pi))
        position = self.state['pos']
        velocity = [float(np.cos(action)*self.ball_speed), float(np.sin(action)*self.ball_speed)]
        done = False
        iteration = 0
        self.pos_list = [[(self.min_x+self.max_x)/2,self.min_y]]
        self.collision_list = [[]]
        
        while(not done):
            
            position[0] = position[0] + self.delta_time*velocity[0]
            position[1] = position[1] + self.delta_time*velocity[1]
            index_ball = self.coord2Index(position)
            
            # Handle out of border situation
            if index_ball[0] >= blockArray.shape[1]:
                index_ball[0] = blockArray.shape[1]-1
            elif index_ball[0] < 0:
                index_ball[0] = 0
            if index_ball[1] >= blockArray.shape[0]:
                index_ball[1] = blockArray.shape[0]-1
            elif index_ball[1] < 0:
                index_ball[1] = 0
            
            # Hit lower border of screen
            if (position[1]<self.min_y):
                done = True
                print('Hit bottom wall')
                break
            
            # Hit other border of screen
            if (position[0]>self.max_x or position[0]<self.min_x):
                velocity[0] *= -1
                print('Hit right or left wall')
            elif (position[1]>self.max_y):
                velocity[1] *= -1
                print('Hit top wall')
            # Hit border of block
            elif (blockArray[index_ball[0], index_ball[1]]!=0):
                print(index_ball)
                coord_block = self.index2Coord(index_ball)
                diff = [abs(coord_block[i]-position[i]) for i in range(2)]
                
                if (diff[0]>diff[1]):
                    velocity[0] *= -1
                    print('Hit right or left of block')
                elif (diff[0]<diff[1]):
                    velocity[1] *= -1
                    print('Hit top or bottom of block')
                else:
                    velocity *= -1
                    print('Hit corner of block')
                
                self.collision_list.append(index_ball.copy())
                #blockArray[index_ball[0], index_ball[1]] -= 1
                print('Hit block (%d,%d) at (%.1f,%.1f)'%(index_ball[0], index_ball[1], position[0],position[1]))
            
            self.pos_list.append(position.copy())
            iteration += 1
            
        print('Iteration: ', iteration)
            
        return self.pos_list
    
    def coord_transform(self, position):
        '''
        Transform coordinate from step to render
        '''
        
        assert len(position) == 2
        
        #position[0] = (position[0]-self.min_x*0.5) * self.scale_x
        position[0] = (position[0]-self.min_x) * self.scale_x
        position[1] = (position[1]-self.min_y) * self.scale_y
        
        return tuple(position)
        
    def render(self, mode='human'):
        '''
        Plot the environment
        '''
        frames = []
        
        if self.viewer is None:
            
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(SCREEN_WIDTH, SCREEN_HEIGHT)
            
            # Add ball
            ball = rendering.make_circle(4)
            position = self.coord_transform(self.pos_list[0])
            self.ballTrans = rendering.Transform(translation=position)
            ball.add_attr(self.ballTrans)
            ball.set_color(.5, .5, .5)
            self.viewer.add_geom(ball)
            
            # Add block
            self.blockTrans = []
            square_long = 40
            #l, r, t, b = -square_long/2 + 5, square_long/2 - 5, square_long/2 - 5 , -square_long/2 + 5
            l, r, t, b = -square_long/2, square_long/2, square_long/2, -square_long/2
            
            for i_row in range(blockArray.shape[0]):
                for i_col in range(blockArray.shape[1]):
                    if blockArray[i_row][i_col] != 0:
                        render_coord = self.coord_transform( self.index2Coord([i_row,i_col]) )
                        self.blockTrans.append( rendering.Transform(translation=render_coord) )
                        
                        block = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
                        block.set_color(.8, .6, .4)
                        block.add_attr(self.blockTrans[-1])
                        self.viewer.add_geom(block)
        
        # Repeat positions
        for i in range(1,len(self.pos_list)):
            position = self.coord_transform(self.pos_list[i])
            self.ballTrans.set_translation(*position)
            #self.ballTrans.set_translation(180, 260)
            #time.sleep(1e-2)
            self.viewer.render(return_rgb_array= mode=='rgb_array')
        
        return frames
            
    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
    
if __name__ == '__main__':
    
    
    env = Ballz()
    env.reset()
    for _ in range(10):
        env.render()
        action = env.action_space.sample()
        pos_list = env.step(action) # take a random action
    env.render()
    env.close()
    
    #display_frames_as_gif(frames)