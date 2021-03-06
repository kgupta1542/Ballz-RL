import gym
from gym import spaces

import numpy as np
import matplotlib.pyplot as plt
from gym.envs.classic_control import rendering

import time
# Constants
SCREEN_WIDTH = 350
SCREEN_HEIGHT = 450
SCREEN_TITLE = "Ballz v1"

class Ballz(gym.Env):
    
    def __init__(self, max_tile=2, normalized_state=False, mode='train', pattern=1):
        '''
        Initialize the environment
        '''

        self.ball_speed = 5
        self.max_x, self.min_x = 3.5, -3.5
        self.max_y, self.min_y = 4.5, -4.5
        self.num_row, self.num_col = 9, 7
        self.delta_time = 0.01
        self.action_space = spaces.Box(low=0.2*np.pi, high=0.8*np.pi, dtype=np.float32, shape=(1,))
        self.observation_space = spaces.Dict({
            'blocks' : spaces.Box(low=0, high=255, dtype=np.uint8, shape=(self.num_row,self.num_col)),
            'X' : spaces.Box(low=self.min_x, high=self.max_x, dtype=np.float32, shape=(1,))
            })
        self.viewer = None
        self.state = None    # (blockArray, ball_pos)
        self.render_blocks = None
        self.pos_list = []
        
        world_width = self.max_x - self.min_x
        world_height = self.max_y - self.min_y
        self.scale_x = SCREEN_WIDTH/world_width
        self.scale_y = SCREEN_HEIGHT/world_height
        self.max_tile = max_tile                    # Max block value
        self.normalized_state = normalized_state    # If it's true, return normalized observation
        self.mode = mode                            # train: randomly generate blocks, test: generate fixed blocks
        self.pattern = pattern                      # testing pattern
        self.i_step = 0                             # index of step in one episode
        
    def reset(self):
        '''
        Reset the environment
        '''
        self.state = {'blocks':np.zeros((self.num_row,self.num_col)), 'X':0}
        self.pos_list = [[0,self.min_y]]
        self.i_step = 0
        
        if self.mode.find('test') != -1:
            if self.pattern == 1:
                self.state['blocks'] = np.array([[0,0,0,0,0,0,1],
                                                 [0,0,0,0,0,0,0],
                                                 [0,0,0,0,0,0,0],
                                                 [0,0,0,0,0,0,0],
                                                 [0,0,0,0,0,0,1],
                                                 [0,0,0,0,0,0,0],
                                                 [0,0,0,0,0,0,0],
                                                 [0,0,0,0,0,0,0],
                                                 [0,0,0,0,0,0,0]])
            elif self.pattern == 2:
                self.state['blocks'] = np.array([[0,0,0,0,0,0,1],
                                                 [0,1,0,0,0,0,0],
                                                 [0,0,0,0,0,0,0],
                                                 [0,0,0,0,0,0,0],
                                                 [0,0,0,0,0,0,1],
                                                 [0,1,0,0,0,0,0],
                                                 [0,0,0,0,0,0,0],
                                                 [0,0,0,0,0,0,0],
                                                 [0,0,0,0,0,0,0]])
            elif self.pattern == 3:
                self.state['blocks'] = np.array([[0,0,0,1,0,0,0],
                                                 [0,0,0,0,0,0,0],
                                                 [0,0,0,0,0,0,1],
                                                 [0,0,0,0,0,0,0],
                                                 [0,0,0,1,0,0,0],
                                                 [0,0,0,0,0,0,0],
                                                 [0,0,0,0,0,0,1],
                                                 [0,0,0,0,0,0,0],
                                                 [0,0,0,0,0,0,0]])
            else:
                np.random.seed(self.pattern)
                self.generateNewRow()
                self.generateNewRow()
                
            return self._get_obs()
            '''
            A pattern that can be learned to one shot two blocks without using 0.2 pi or 0.8 pi
            np.array([[0,0,0,0,0,0,1],
                     [0,0,0,0,0,0,0],
                     [0,0,0,0,0,0,0],
                     [0,0,0,0,0,0,0],
                     [0,0,0,0,0,0,1],
                     [0,0,0,0,0,0,0],
                     [0,0,0,0,0,0,0],
                     [0,0,0,0,0,0,0],
                     [0,0,0,0,0,0,0]])
            More complex pattern
            np.array([[0,0,0,0,0,0,1],
                     [0,1,0,0,0,0,0],
                     [0,0,0,0,0,0,0],
                     [0,0,0,0,0,0,0],
                     [0,0,0,0,0,0,1],
                     [0,1,0,0,0,0,0],
                     [0,0,0,0,0,0,0],
                     [0,0,0,0,0,0,0],
                     [0,0,0,0,0,0,0]])
            One shot pattern
            np.array([[0,0,0,1,0,0,0],
                     [0,0,0,0,0,0,0],
                     [0,0,0,0,0,0,1],
                     [0,0,0,0,0,0,0],
                     [0,0,0,1,0,0,0],
                     [0,0,0,0,0,0,0],
                     [0,0,0,0,0,0,1],
                     [0,0,0,0,0,0,0],
                     [0,0,0,0,0,0,0]])
            
            '''
        
        # Generate two blocks of row
        self.generateNewRow()
        self.generateNewRow()
        
        return self._get_obs()
        
    def step(self, action):
        '''
        Action changes environment
        '''
        assert self.action_space.low<=action<=self.action_space.high
        #print(r'Action: %.2f pi'%(action/np.pi))
        
        position = np.array([self.state['X'],self.min_y])
        velocity = np.array([float(np.cos(action)*self.ball_speed), float(np.sin(action)*self.ball_speed)])
        done = False
        iteration = 0
        self.pos_list = [position]
        self.collision_map = {}
        self.remove_map = {}    # Remove the block
        

        # Shift one row and randomly generate first row
        if (self.mode in ['train3','train4','train5'] and self.i_step==0) or \
            (self.mode in ['test3','test4','test5'] and self.pattern not in [1,2,3] and self.i_step==0):
            for i in range(4):
                self.generateNewRow()
        elif self.mode == 'train' or (self.mode=='train2' and self.i_step<=5):
            self.generateNewRow()
        
        blockArray = self.state['blocks']
        block_values = np.sum(blockArray)    # Sum of block array
        
        # Update render blocks
        self.render_blocks = self.state['blocks'].copy()
            
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
                            #print('Part a: prev %s | cur %s'%(index_prev_ball,index_ball))
                    else:
                        velocity[0] *= -1
                        #print('Part b: prev %s | cur %s'%(index_prev_ball,index_ball))
                    
                elif (diff[0]<diff[1]): # In the lower or right part
                    
                    if blockArray[index_prev_ball[0],index_ball[1]]>0:
                        if blockArray[index_ball[0],index_prev_ball[1]]>0:       # Hit block is between two other blocks
                            velocity *= -1
                        else:                                                       # Hit block is next to one block
                            velocity[0] *= -1
                            #print('Part c: prev %s | cur %s'%(index_prev_ball,index_ball))
                    else:
                        velocity[1] *= -1
                        #print('Part d: prev %s | cur %s'%(index_prev_ball,index_ball))
                    
                else:                   # In the diangonal part
                    velocity *= -1
                    #print('Hit corner of block')
                
                self.collision_map[len(self.pos_list)] = index_ball
                
                if self.mode not in ['train4','test4']:
                    blockArray[index_ball[0], index_ball[1]] -= 1
                if blockArray[index_ball[0], index_ball[1]] == 0:
                    self.remove_map[len(self.pos_list)] = index_ball
                #print('Hit block (%d,%d) at (%.1f,%.1f)'%(index_ball[0], index_ball[1], position[0],position[1]))
            
            self.pos_list.append(position.copy())
            iteration += 1
            
            # Avoid ball stucking in weird location
            if iteration >= 100000:
                return self._get_obs(), 0, True
        
        # Set reward function
        if self.mode.find('test') != -1:
            reward = len(self.collision_map) if len(self.collision_map)>0 else -1
        else:
            #reward = len(self.collision_map) / block_values if len(self.collision_map)>0 else 0     # (1)
            reward = len(self.collision_map) if len(self.collision_map)>0 else 0                  # (2)
            #reward = 1 if len(self.collision_map)>0 else -1                                        # (3)
        
            '''
            # Prefer center actions
            scale = np.abs(action-self.action_space.low) if action<0.5*np.pi else np.abs(action-self.action_space.high)
            scale = int(scale/np.pi*10) + 1
            reward = len(self.collision_map)*scale if len(self.collision_map)>0 else -1
            '''
            '''
            # Prefer lower blocks
            reward = 0
            for key in self.collision_map:
                reward += self.collision_map[key][0]+1
            reward = reward if len(self.collision_map)>0 else 0
            '''
        
        if self.mode in ['train3','train5']:
            # End game if there's no blocks
            if np.all(blockArray==0):
                reward += 1
                return self._get_obs(), reward, True
        if self.mode == 'train2':
            # End game if there's no blocks
            if np.all(blockArray==0) and self.i_step>=5:
                reward += 20 - self.i_step
                return self._get_obs(), reward, True
        elif self.mode in ['test','test3','test5']:
            # End game if there's no blocks
            if np.all(blockArray==0):
                reward += 1
                return self._get_obs(), reward, True
        else:
            # End game if any block exists in last second row
            if (np.any(blockArray[-2]!=0)):
                reward = reward - 1
                return self._get_obs(), reward, True
        
        
        if self.mode in ['train3','test3','train4','test4']:
            # Update X with the central position
            self.state['X'] = (self.min_x+self.max_x)/2
        else:
            # Update X with last position
            self.state['X'] = position[0]
        
        # Add step
        self.i_step += 1
        
        return self._get_obs(), reward, False
        
    def render(self, mode='human'):
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
                    
        # Repeat positions
        for i in range(1,len(self.pos_list)):
            position = self.coord_transform(self.pos_list[i])
            self.ballTrans.set_translation(*position)
            
            # Remove
            if (i in self.remove_map):
                self.blockImages[self.remove_map[i]]._color.vec4 = (1,1,1,0)
            
            self.viewer.render(return_rgb_array= mode=='rgb_array')
            
        # Update the render blocks
        self.render_blocks = self.state['blocks'].copy()
        
        return frames
            
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
        
    def _get_obs(self):
        
        block_state, X_state = self.state['blocks'].flatten(), self.state['X']
        
        if self.normalized_state:
            block_state = (block_state/(self.max_tile/2)) - 1
            X_state /= self.max_x
            
        return np.append(block_state, X_state)
    
if __name__ == '__main__':
    
    start_time = time.time()
    env = Ballz(1, normalized_state=True, mode='train3')
    
    for i_episode in range(10000):
        print('Episode %d: '%(i_episode))
        env.reset()
        for _ in range(20):
            #action = env.action_space.sample()
            action = 0.25*np.pi
            state, reward, done = env.step(action) # take a random action
            print('Action: %.1f pi, Reward: %f'%(action/np.pi, reward))
            env.render()
            if done:
                break
    env.close()
    
    print('Elapsed %.1f s'%(time.time()-start_time))
