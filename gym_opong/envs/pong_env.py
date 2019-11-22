import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import pygame


def cartesian(arrays, out=None):
    """
    Generate a cartesian product of input arrays.

    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.

    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.

    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])

    """

    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = int(n / arrays[0].size)
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m,1:])
        for j in range(1, arrays[0].size):
            out[j*m:(j+1)*m,1:] = out[0:m,1:]
    return out




class PongEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self,game_name='Pong',enable_render=False):
        self.done = 0
        self.counter = 0
        self.reward = 0
        self.w = 512
        self.h = 256
        self.grid_size = 16
        self.grid_w = self.w//self.grid_size
        self.grid_h = self.h//self.grid_size
        
        
        if enable_render:
            pygame.init()
            self.screen = pygame.display.set_mode((self.w,self.h))
            self.clock = pygame.time.Clock()
        else:
            self.screen = None

        self.ball = Ball(self.w,self.h,self.screen,speed=4)
        self.enemyPaddle = EnemyPaddle(self.w,self.h,self.ball,
                self.screen,speed=4)
        self.aiPaddle = AIPaddle(self.w,self.h,self.ball,
                self.screen,speed=4)
        
        self.action_space = spaces.Discrete(2)
        self.observation_space = self._observation_space()
        self.state = None
        self.score = [0,0]
        self.build_state_encode()
        return

    def build_state_encode(self):
        all_angles = np.array(list(range(16)))
        all_ball_pos = np.array(list(range(256)))
        all_epaddle_pos = np.array(list(range(16)))
        all_aipaddle_pos = np.array(list(range(16)))
        all_states = cartesian((all_angles, all_ball_pos,
            all_epaddle_pos, all_aipaddle_pos))

        self.state_encodes = {}
        for ind in range(all_states.shape[0]):
            self.state_encodes[tuple(all_states[ind].tolist())] = ind
        return state_encodes

    def encode_obs(self,obs):
        obs = np.array(obs)
        ball_angle = obs[0]
        ball_pos = obs[1]
        epaddle_pos = obs[2]
        aipaddle_pos = obs[3]
        return self.state_encode[tuple([ball_angle,ball_pos,
            epaddle_pos,aipaddle_pos])]

    def _del(self):
        if self.screen is not None:
            pygame.display.close()
            pygame.close()
            self.screen = None
        return

    def gridToInd(self,x,y):
        return np.ravel_multi_index([y,x],[self.grid_size,self.grid_w]) 

    def _observation_space(self):
        lows = np.zeros(304)
        highs = np.ones(304)
        return spaces.Box(lows,highs,dtype=np.int32)

    def step(self,action):
        self.aiPaddle.step(action)
        self.enemyPaddle.step()
        self.reward = self.ball.step()
        if self.reward != 0:
            self.ball.reset()
        
        if self.reward==1:
            self.score[0]+=1
        elif self.reward==-1:
            self.score[1]+=1
        self.done = np.maximum(self.score[0],self.score[1]) >= 21 
        
        ballangle_state = self.ball.get_angle()
        ballpos_state = self.gridToInd(self.ball.x//self.grid_w,
            self.ball.y//self.grid_h)
        epaddle_state = self.enemyPaddle.y//self.grid_h
        aipaddle_state = self.aiPaddle.y//self.grid_h

        self.state = self.encode_obs([ballangle_state,
            ballpos_state,epaddle_state,aipaddle_state]) 
        
        info = {}
    
        return self.state, self.reward, self.done, info 

    def reset(self):
        self.score = [0,0]
        self.ball.reset()
        self.enemyPaddle.reset()
        self.aiPaddle.reset()
        self.state = [0] * 304
        ballangle_state = self.ball.get_angle()
        ballpos_state = onehot(self.gridToInd(self.ball.x//self.grid_w,
            self.ball.y//self.grid_h),self.grid_size**2)
        epaddle_state = onehot(self.enemyPaddle.y//self.grid_h,self.grid_size)
        aipaddle_state = onehot(self.aiPaddle.y//self.grid_h,self.grid_size)

        self.state = ballangle_state + ballpos_state + epaddle_state + aipaddle_state 
        return self.state

    def render(self,mode='human',close=False):
        
        if self.screen is not None:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    close = True
            if close:
                pygame.display.close()
                pygame.close()
                self.screen = None

        self.ball.render()
        self.enemyPaddle.render()
        self.aiPaddle.render()
        if self.screen is not None:
            pygame.display.flip()
            self.clock.tick(60)
            self.screen.fill((0,0,0))
        return



def angleToVec(angle):
    angle = degtorad(angle)
    return np.cos(angle), np.sin(angle)

def vecToAngle(vector):
    return radtodeg(np.arctan2(vector[0],vector[1]))

def degtorad(angle):
    return angle * (np.pi/180)

def radtodeg(angle):
    return angle * (180/np.pi)


def onehot(val,num_vals):
    try:
        ret = [0] * num_vals
        ret[int(val)] = 1
    except:
        print(ret)
        print(val)
        print(num_vals)
        input("Stop")
    return ret


class Ball:
    def __init__(self,width,height,_render = None,speed=7):
        self.r = 5 
        self._render = _render
        self.speed = speed
        self.x = width//2
        self.y = np.random.randint(round(height*0.1+self.r),
                                   round(height-(height*0.1)-self.r))
        if np.random.rand() > 0.5:
            self.vector = angleToVec(np.random.randint(-90+22.5,90-22.5))
        else:
            self.vector = angleToVec(np.random.randint(90+22.5,270-22.5))
        self.swidth = width
        self.sheight = height
        return
    
    def get_angle(self):
        ball_angle = (vecToAngle(self.vector)-90)%360
        ball_angle = ball_angle//22.5
        return ball_angle



    def reset(self):
        self.x = self.swidth//2
        self.y = np.random.randint(round(self.sheight*0.1+self.r),
                                   round(self.sheight-(self.sheight*0.1)-self.r))
        if np.random.rand() > 0.5:
            self.vector = angleToVec(np.random.randint(-90+22.5,90-22.5))
        else:
            self.vector = angleToVec(np.random.randint(90+22.5,270-22.5))
        return

    def render(self):
        if self._render:
            pygame.draw.circle(self._render, (255,255,255),
                    (int(self.x),int(self.y)),int(self.r))
        return

    def step(self):
        self.vector = self.vector / np.linalg.norm(self.vector) * self.speed
        new_x = self.x + self.vector[0]
        new_y = self.y + self.vector[1]
        ret = 0
        if (new_y > self.sheight-self.r) or (new_y < self.r):
            self.vector[-1] = -self.vector[-1]
        if (new_x > self.swidth-self.r):
            #Right Hit
            ret = 1
            self.vector[0] = -self.vector[0]
        elif (new_x < self.r):
            #Left hit
            ret = -1
            self.vector[0] = -self.vector[0]
        
        self.x += self.vector[0] 
        self.y += self.vector[1]

        return ret


class Paddle:
    def __init__(self,width,height,_render=None,speed=2):
        self.w = 5 
        self.h = width//18
        self.speed = speed
        self._render = _render
        self.swidth = width
        self.sheight = height

    def step(self):
        self.vector = self.vector / np.linalg.norm(self.vector) * self.speed
        new_y = self.y + self.vector[1]

        if (new_y > self.sheight+self.h) or (new_y < 0):
            self.vector[-1] = 0
        
        self.y += self.vector[1]

        return

    def render(self):
        if self._render:
            pygame.draw.rect(self._render,(255,255,255),pygame.Rect(self.x,self.y,self.w,self.h))
        return

    def collide(self,d):
        if circlerectcollision(self.ball,self):
            relativeIntersectY = (self.y+(self.h//2)) - self.ball.y
            normIntersectY = (relativeIntersectY/(self.h//2))
            bounceAngle = normIntersectY * degtorad(80)
            self.ball.vector[0] = d * np.maximum(np.cos(bounceAngle),0.1)
            self.ball.vector[1] = -np.sin(bounceAngle)
            
        return

class EnemyPaddle(Paddle):
    def __init__(self,width,height,ball,_render=None,speed=4):
        Paddle.__init__(self,width,height,_render,speed)
        self.ball = ball
        self.x = width - (width//8)
        self.y = 0
        self.vector = np.zeros(2)

    def reset(self):
        self.y=0
        self.vector = np.zeros(2)


    def step(self):
        target_y = self.ball.y - (self.y+(self.h//2))
        if (np.abs(target_y) < 5):
            target_y = 0
        self.vector = [0,target_y]
        if self.vector[-1] != 0:
            self.vector = self.vector / np.linalg.norm(self.vector) * self.speed
        new_y = self.y + self.vector[1]

        if (new_y > self.sheight-self.h) or (new_y < 0):
            self.vector[-1] = 0
        self.y += self.vector[1]
        self.collide(-1)
        return


class AIPaddle(Paddle):
    def __init__(self,width,height,ball,_render=None,speed=4):
        Paddle.__init__(self,width,height,_render,speed)
        self.ball = ball
        self.x = width//8
        self.y = 0
        self.vector = np.zeros(2)

    def reset(self):
        self.y = 0
        self.vector = np.zeros(2)

    def step(self,action):
        if action == 1:
            target_y = 1
        else:
            target_y = -1
        self.vector = [0,target_y]
        if self.vector[-1] != 0:
            self.vector = self.vector / np.linalg.norm(self.vector) * self.speed
        new_y = self.y + self.vector[-1]

        if (new_y > self.sheight-self.h) or (new_y < 0):
            self.vector[-1] = 0
        self.y += self.vector[-1]
        self.collide(1)





def circlerectcollision(circle,rect):
    cx = circle.x
    cy = circle.y
    cr = circle.r
    rcx = rect.x+(rect.w//2)
    rcy = rect.y+(rect.h//2)
    rw = rect.w
    rh = rect.h

    nearestX = np.maximum(rect.x,np.minimum(cx,rect.x+rw))
    nearestY = np.maximum(rect.y,np.minimum(cy,rect.y+rh))

    dX = cx - nearestX
    dY = cy - nearestY

    return (dX * dX) + (dY*dY) < (cr * cr)

