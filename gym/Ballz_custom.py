import time

import gym
from gym import spaces
from gym import utils
from gym.envs.classic_control import rendering
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image 
import PIL 

# Constants
SCREEN_WIDTH = 350
SCREEN_HEIGHT = 450

class Ballz(gym.Env, utils.EzPickle):
    
    def __init__(self, max_tile=2):
        '''
        Initialize the environment
        '''

        self.ball_speed = 5
        self.max_x, self.min_x = 3.5, -3.5
        self.max_y, self.min_y = 4.5, -4.5
        self.num_row, self.num_col = 9, 7
        self.delta_time = 0.01
        self.action_space = spaces.Box(low=0.2*np.pi, high=0.8*np.pi, dtype=np.float32, shape=(1,))
        self.observation_space = spaces.Box(low=0, high=255, dtype=np.uint8, shape=(SCREEN_HEIGHT, SCREEN_WIDTH, 3))
        self.viewer = None
        self.state = None    # (blockArray, ball_pos)
        self.render_blocks = None
        self.pos_list = []
        
        world_width = self.max_x - self.min_x
        world_height = self.max_y - self.min_y
        self.scale_x = SCREEN_WIDTH/world_width
        self.scale_y = SCREEN_HEIGHT/world_height
        self.max_tile = max_tile                    # Max block value
        
    def reset(self):
        '''
        Reset the environment
        '''
        self.state = {'blocks':np.zeros((self.num_row,self.num_col)), 'X':0}
        self.pos_list = [[0,self.min_y]]
        self.close()
        
        # Generate two blocks of row
        self.generateNewRow()
        self.generateNewRow()
        
        self.render_blocks = self.state['blocks'].copy()
        
        return self._get_obs('reset')
        
    def step(self, action):
        '''
        Action changes environment
        '''
        assert 0<=action<=np.pi
        #print(r'Action: %.2f pi'%(action/np.pi))
        
        position = np.array([self.state['X'],self.min_y])
        velocity = np.array([float(np.cos(action)*self.ball_speed), float(np.sin(action)*self.ball_speed)])
        done = False
        info = None
        iteration = 0
        self.pos_list = [position]
        self.collision_map = {}
        self.remove_map = {}    # Remove the block
        
        # Shift one row and randomly generate first row
        self.generateNewRow()
        blockArray = self.state['blocks']
        block_values = 100000 if np.all(blockArray==0) else np.sum(blockArray)    # Sum of block array
        
        # Update render blocks
        self.render_blocks = self.state['blocks'].copy()
        
        # End game if any block reaches the last row
        if (np.any(blockArray[-1]!=0)):
            reward = 0
            return self._get_obs('step'), reward, True, info
            
        while(not done):
            position[0] = position[0] + self.delta_time*velocity[0]
            position[1] = position[1] + self.delta_time*velocity[1]
            index_ball = self.coord2Index(position)
            
            # Hit lower border of screen
            if (position[1]<self.min_y):
                done = True
                #print('Hit bottom wall')
                break
            
            # Hit other border of screen
            if (position[0]>self.max_x):        # Right border
                velocity[0] = -abs(velocity[0])
            elif (position[0]<self.min_x):      # Left border
                velocity[0] = abs(velocity[0])
            elif (position[1]>self.max_y):      # Top boder
                velocity[1] = -abs(velocity[1])
                
            # Hit border of block
            elif (blockArray[index_ball[0], index_ball[1]]!=0):
                #print(index_ball)
                coord_block = self.index2Coord(index_ball)
                diff = [abs(coord_block[i]-position[i]) for i in range(2)]          # Coordinate diff between ball and block
                index_prev_ball = self.coord2Index(self.pos_list[-1])
                diff_ball = [index_ball[i]-index_prev_ball[i] for i in range(2)]    # Index diff between current and previous ball
                
                if (diff[0]>diff[1]):    # In the right or left part
                    
                    if blockArray[index_ball[0],index_prev_ball[1]]>0:
                        if blockArray[index_prev_ball[0],index_ball[1]]>0:       # Hit block is between two other blocks
                            velocity *= -1
                        else:                                                       # Hit block is next to one block
                            velocity[1] *= -1
                    else:
                        velocity[0] *= -1
                    
                elif (diff[0]<diff[1]): # In the lower or right part
                    
                    if blockArray[index_prev_ball[0],index_ball[1]]>0:
                        if blockArray[index_ball[0],index_prev_ball[1]]>0:       # Hit block is between two other blocks
                            velocity *= -1
                        else:                                                       # Hit block is next to one block
                            velocity[0] *= -1
                    else:
                        velocity[1] *= -1
                    
                else:                   # In the diangonal part
                    velocity *= -1
                
                self.collision_map[len(self.pos_list)] = index_ball
                blockArray[index_ball[0], index_ball[1]] -= 1
                if blockArray[index_ball[0], index_ball[1]] == 0:
                    self.remove_map[len(self.pos_list)] = index_ball
            
            self.pos_list.append(position.copy())
            iteration += 1
        
        # Set reward function
        reward = len(self.collision_map) / block_values if len(self.collision_map)>0 else -1
        #reward = len(self.collision_map) if len(self.collision_map)>0 else -1
        #reward = 1 if len(self.collision_map)>0 else -1
        ''''
        # Prefer center actions
        scale = np.abs(action-self.action_space.low) if action<0.5*np.pi else np.abs(action-self.action_space.high)
        scale = int(scale/np.pi*10) + 1
        reward = len(self.collision_map)*scale if len(self.collision_map)>0 else 0
        
        '''
        '''
        # Prefer lower blocks
        reward = 0
        for key in self.collision_map:
            reward += self.collision_map[key][0]+1
        reward = reward if len(self.collision_map)>0 else 0
        '''
        if np.all(blockArray==0):
            return self._get_obs('step'), reward, True, info
        
        # Update X with last position
        self.state['X'] = position[0]
        
        return self._get_obs('step'), reward, False, info
        
    def render(self, mode='human', call_loc='render'):
        '''
        Plot the environment
        '''
        frames = []
        
        if self.viewer is None:
            self.viewer = rendering.Viewer(SCREEN_WIDTH, SCREEN_HEIGHT)
        
            # Add ball
            ball = rendering.make_circle(4)
            position = self.coord_transform(self.pos_list[0])
            self.ballTrans = rendering.Transform(translation=position)
            ball.add_attr(self.ballTrans)
            ball.set_color(.5, .5, .5)
            self.viewer.add_geom(ball)
            
            # Add block
            self.blockImages = {}
            square_long = 40
            l, r, t, b = -square_long/2, square_long/2, square_long/2, -square_long/2
        
            for i_row in range(self.num_row):
                for i_col in range(self.num_col):
                    render_coord = self.coord_transform( self.index2Coord([i_row,i_col]) )
                    
                    block = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
                    if self.render_blocks[i_row][i_col] == 1:
                        block.set_color(.8, .6, .4)
                    elif self.render_blocks[i_row][i_col] > 1:
                        block.set_color(.4, .6, .8)
                    else:
                        block._color.vec4 = (1, 1, 1, 0)
                    block.add_attr(rendering.Transform(render_coord))
                    self.viewer.add_geom(block)
                    
                    self.blockImages[(i_row,i_col)] = block
                    
        # Plot the blocks before step
        for i_row in range(self.num_row):
            for i_col in range(self.num_col):
                if self.render_blocks[i_row][i_col] == 1:
                    self.blockImages[(i_row,i_col)].set_color(.8,.6,.4)
                elif self.render_blocks[i_row][i_col] > 1:
                    self.blockImages[(i_row,i_col)].set_color(.4,.6,.8)
                else:
                    self.blockImages[(i_row,i_col)]._color.vec4 = (1,1,1,0)
                    
        img = self.viewer.render(return_rgb_array= mode=='rgb_array')
        if call_loc == 'reset':
            return img
            
        # Repeat positions
        for i in range(1,len(self.pos_list)):
            position = self.coord_transform(self.pos_list[i])
            self.ballTrans.set_translation(*position)
            
            # Remove
            if (i in self.remove_map):
                self.blockImages[self.remove_map[i]]._color.vec4 = (1,1,1,0)
            
            if call_loc == 'render':
                img = self.viewer.render(return_rgb_array= mode=='rgb_array')
            
        img = self.viewer.render(return_rgb_array = mode=='rgb_array')
        if call_loc == 'step':
            return img
            
        # Update the render blocks
        self.render_blocks = self.state['blocks'].copy()
        
        return img
            
    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
            
    def coord_transform(self, position):
        '''
        Transform coordinate from step to render
        '''
        
        assert len(position) == 2
        
        #position[0] = (position[0]-self.min_x*0.5) * self.scale_x
        position[0] = (position[0]-self.min_x) * self.scale_x
        position[1] = (position[1]-self.min_y) * self.scale_y
        
        return tuple(position)
        
    def coord2Index(self, position):
        '''
        Transform coordinate into index of block array
        position (x,y) <-> index (row, col)
        row corresponds to y, col corresponds to x
        '''
        assert len(position) == 2
        
        index = np.zeros(2)
        index[0], index[1] = position[0], position[1]*-1   # Flip y axis
        index[0], index[1] = index[1]-self.min_y, index[0]-self.min_x # Shift all the index, e.g. (min_x,min_y)->(0,0)
        
        # Handle out of border situation
        index[0] = 0 if index[0]<0 else index[0]
        index[0] = self.num_row-1 if index[0]>=self.num_row else index[0]
        index[1] = 0 if index[1]<0 else index[1]
        index[1] = self.num_col-1 if index[1]>=self.num_col else index[1]
            
        return tuple(index.astype('int'))
    
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
        
    def generateNewRow(self):
        
        blockArray = self.state['blocks']
        blockArray = np.roll(blockArray, 1, axis=0)
        if self.max_tile == 1:
            blockArray[0,:] = np.random.choice([0,1], p=[0.6,0.4], size=self.num_col)
        elif self.max_tile == 2:
            blockArray[0,:] = np.random.choice([0,1,2], p=[0.2,0.6,0.2], size=self.num_col)
        else:
            blockArray[0,:] = np.random.randint(self.max_tile+1, size=self.num_col)
        self.state['blocks'] = blockArray
        
    def _get_obs(self, call_loc):
        
        img = self.render(mode='rgb_array', call_loc=call_loc)
            
        return img
    
if __name__ == '__main__':
    
    start_time = time.time()
    env = Ballz(2)
    
    for i_episode in range(3):
        print('Episode %d: '%(i_episode))
        img = env.reset()
        image = PIL.Image.fromarray(img, "RGB")
        image.save('reset.jpg')
        for i in range(100):
            action = env.action_space.sample()
            state, reward, done, info = env.step(action) # take a random action
            print('Action: %.1f pi, Reward: %f'%(action/np.pi, reward))
            image = PIL.Image.fromarray(state, "RGB")
            image.save('step%d.jpg'%(i))
            #env.render()
            if done:
                break
    env.close()
    
    print('Elapsed %.1f s'%(time.time()-start_time))
