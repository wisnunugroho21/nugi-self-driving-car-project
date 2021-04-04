
import gym
from gym import spaces
import numpy as np
from collections import deque
from skimage.transform import resize
from PIL import Image

class WrapperEnv(gym.Wrapper):
    def __init__(self, env, num_stack=4):
        gym.Wrapper.__init__(self, env)
        self.num_stack = num_stack
        shape = env.observation_space.shape
        self.side = 84
        self.observation_space = spaces.Box(low=0, high=255, shape=(shape[0],
            shape[1], self.num_stack))

        self.frames = deque([], maxlen=self.num_stack)

        self.lives = 0
        self.out_of_lives = False
        self.started = False

    def grayscale_resize(self, frame):
        #obs = resize(np.array(frame)[...,:3].dot([0.299, 0.587, 0.114]), (84, 84))
        cropped_frame = frame[34:-16,:,:]
        gray_frame = np.dot(cropped_frame.astype('float32'),
            np.array([0.299, 0.587, 0.114], 'float32'))
        obs = np.array(Image.fromarray(gray_frame).resize((self.side, self.side),
                        resample=Image.BILINEAR), dtype=np.uint8)
        return obs

    def _reset(self):
        if self.out_of_lives or not self.started:
            frame = self.grayscale_resize(self.env.reset())
            for i in range(self.num_stack):
                self.frames.append(frame)
            self.started = True
            self.out_of_lives = False
            obs, reward, done, info = self._step(1)
        # Press fire at the beginning
        else:
            obs, reward, done, info = self._step(1)

        self.lives = self.env.unwrapped.ale.lives()
        return self._observation()

    def _step(self, action):
        frame, reward, done, info = self.env.step(action)
        self.out_of_lives = done
        self.frames.append(self.grayscale_resize(frame))

        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            done = True
        # This if is redundant
        if lives == 0:
            self.out_of_lives = True
        self.lives = lives

        return self._observation(), reward, done, info

    def _observation(self):
        return np.stack(self.frames, axis=-1)