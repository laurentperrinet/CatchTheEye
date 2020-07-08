#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""

The client side.

"""

import numpy as np
import time

# VISUALIZATION ROUTINES
from vispy import app
from vispy import gloo

#--------------------------------------------------------------------------

class Stimulation(object):
    """
    A stimulation is an ensemble of stimuli and its properties. This class
    allows creating and running stimulations and is handled by Client.

    To set positions we will use OpenCV's coordinate system in which the origin
    is at the top-right on the screen and coordinates are normalized.

    """

    def __init__(self, window_w, window_h, stim_type = 'calibration_horizontal'):
        self.stim_type = stim_type
        self.window_h = window_h
        self.window_w = window_w
        self.ratio = self.window_w / self.window_h
        self.stim_details()

    def run(self):
        """
        Provide a continuum-like use of the class
        """
        t0 = time.time()
        i = 0
        while (time.time()-t0 < self.duration or i <= len(self.tabPos)-1):
            if time.time()-t0 < self.transition_lag*(i+1):
                image = self.draw_stimulus(self.tabPos[i])
                #Use image as you like (Feeding a vizualisation tool)
            print ("stim en cours")
            i += 1

    def get_stimulus(self, t0, t):
        """
        Provide a discrete use of the class
        Return a cliche of stimulation at t, t0 the start of stimulation
        """
        i = int((t-t0)//self.transition_lag)
        stim_x, stim_y = self.tabPos[i]
        return self.draw_stimulus((stim_x, stim_y)), stim_x

    def stim_details(self):
        """
        (private) Set stimulation properties according to stimulation desired type
        """
        if self.stim_type == 'calibration':
            self.duration = 10
            #10° excentricity if subject-screen = 57 cm and the display is 60x33 cm
            self.tabPos = [(0.5, 0.5), (0.5, 0.2), (0.33, 0.5), (0.5, 0.5), (0.5, 0.8), (0.67, 0.5), (0.5, 0.5)]
            #self.tabPos = [(0.5, 0.5), (0.5, 0.1), (0.1, 0.5), (0.5, 0.5), (0.5, 0.9), (0.9, 0.5), (0.5, 0.5)]
            self.transition_lag = 1 # how long (in seconds) a calibration dot is shown
            self.stimulus = 'target'
        elif self.stim_type == 'calibration_horizontal':
            self.duration = 9
            self.tabPos = [(0.5, 0.5), (0.67, 0.5), (0.33, 0.5), (0.67, 0.5), (0.5, 0.5), (0.67, 0.5), (0.33, 0.5), (0.5, 0.5), (0.33, 0.5)]
            self.transition_lag = 1
            self.stimulus = 'target'
        else :
            print ("This type of stimulation is not implemented for the moment")

    def draw_stimulus(self, pos):
        """
        (private) Draw a stimulus at a normalized position given.
        pos : the stimulus normalized localization

        """
        img0 = np.zeros((self.window_h, self.window_w, 3)).astype(np.uint8)

        #fixation cross
        img = img0.copy()
        import cv2
        img = cv2.line(img, self.compute_pos((0.470, 0.5)), self.compute_pos((0.530, 0.5)), (255, 255, 255), 2)
        img = cv2.line(img, self.compute_pos((0.5, 0.45)), self.compute_pos((0.5, 0.55)), (255, 255, 255), 2)

        if self.stimulus == 'target':
            img = cv2.circle(img, self.compute_pos(pos), 24, (0, 0, 255), 1)
            img = cv2.circle(img, self.compute_pos(pos), 6, (0, 0, 255), -1)

        return img

    def compute_pos(self, pos):
        """
        (private) Compute position in OpenCV coordinates system
        """
        posX, posY = pos
        posX = int((1-posX) * self.window_w)
        posY = int(posY * self.window_h)
        return (posX, posY)

#---------------------------------------------------------------------------

vertex = """
    attribute vec2 position;
    attribute vec2 texcoord;
    varying vec2 v_texcoord;
    void main()
    {
        gl_Position = vec4(position, 0.0, 1.0);
        v_texcoord = texcoord;
    } """

fragment = """
    uniform sampler2D texture;
    varying vec2 v_texcoord;
    void main()
    {
        gl_FragColor = texture2D(texture, v_texcoord);
    } """


class Client(app.Canvas):
    """
    The client initializes and updates the display where stimulations and
    camera take will occur.
    """
    def __init__(self, et, timeline, downscale=1, stim_type='calibration_horizontal'):
        self.downscale = downscale
        self.et = et
        self.timeline = timeline

        app.use_app('pyglet')
        app.Canvas.__init__(self, keys='interactive', size=(1280, 720))
        self.fullscreen = True
        self.width, self.height = self.physical_size
        print ('window size : ', self.physical_size)
        self.stimulation = Stimulation(self.width//downscale, self.height//downscale, stim_type=stim_type)
        self.program = gloo.Program(vertex, fragment, count=4)
        self.program['position'] = [(-1, -1), (-1, +1), (+1, -1), (+1, +1)]
        self.program['texcoord'] = [(1, 1), (1, 0), (0, 1), (0, 0)]
        self.program['texture'] = (255*np.random.rand(self.height//downscale, self.width//downscale, 3)).astype(np.uint8)
        gloo.set_viewport(0, 0, self.width, self.height)
        self._timer = app.Timer('auto', connect=self.on_timer, start=True)
        self.start = time.time()
        self.stims_X_t = []
        self.eye_x_t = []
        self.native.set_mouse_visible(False)
        self.show()

    def on_resize(self, event):
        self.width, self.height = event.physical_size
        gloo.set_viewport(0, 0, self.width, self.height)

    def on_draw(self, event):
        gloo.clear('black')
        self.program.draw('triangle_strip')

    def on_timer(self, event):
        if time.time()-self.start < self.timeline.max():
            image, x = self.stimulation.get_stimulus(t0 = self.start, t = time.time())
            self.stims_X_t.append((x, time.time()-self.start))
            #feedback cross
            #with_cross = image.copy()
            #with_cross = cv2.line(img, self.compute_pos((0.470, 0.5)), self.compute_pos((0.530, 0.5)), (255, 255, 255), 2)
            #with_cross = cv2.line(img, self.compute_pos((0.5, 0.45)), self.compute_pos((0.5, 0.55)), (255, 255, 255), 2)
            self.program['texture'][...] = image.astype(np.uint8).reshape((self.height//self.downscale, self.width//self.downscale, 3))
            try:
                frame = self.et.cam.grab()
                if not frame is None:
                    img_face = self.et.F.face_extractor(frame[:, :, ::-1])
                    pred = 'DEBUG' #self.ml.classify(img_face, ml.dataset.test_transform)
                    self.eye_x_t.append((pred, time.time() - start))
            except Exception as e:
                if not self.et is None: print('could not grab a frame / detect the eye position', e)
            self.update()
        else:
            self.close()
            print ('target dynamic :\n', self.stims_X_t)
            print ('eye dynamic :\n', self.et.eye_x_t)

if __name__ == '__main__':

    import numpy as np
    from LeCheapEyeTracker import Server

    fps = 30 # the maximum FPS we try in this experiment
    T = 7.
    screen = Client(et=Server(), timeline=np.linspace(0, T, T*fps), downscale=2)
    screen.app.run()
    screen.et.close()
