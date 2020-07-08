#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Using psychopy to perform an experiment on the role of a bias in the direction """

import sys
import os
import numpy as np
import pickle
from LeCheapEyeTracker.EyeTrackerServer import Server
import imageio
# pip3 install git+https://laurentperrinet.github.com/openRetina

class Experiment(object) :

    def __init__(self, timeStr, observer='test') :
        self.observer = observer
        self.timeStr = str(timeStr)
        self.init()

    def init(self) :

        # ---------------------------------------------------
        # Dossier
        # ---------------------------------------------------
        self.datadir = 'dataset'
        targets = ['left', 'blink', 'center', 'right']

        import os
        try: os.mkdir(self.datadir)
        except: pass
        for target in targets :
            print(os.path.join(self.datadir, target))
            try: os.mkdir(os.path.join(self.datadir, target))
            except: pass

        #----------------------------------------------------
        # Caméra
        #----------------------------------------------------
        try :
            et = Server()
            self.camera = et.cam
        except :
            self.camera = None

        #----------------------------------------------------

        # ---------------------------------------------------
        # Présente un dialogue pour changer les paramètres
        # ---------------------------------------------------
        expInfo = {"Sujet":'', "Age":'', "Nomdre de trials":''}
        from psychopy import gui
        dlg = gui.DlgFromDict(expInfo, title=u'aSPEM')

        self.observer = expInfo["Sujet"]
        age = expInfo["Age"]
        N_trials = int(expInfo["Nomdre de trials"])

        # ---------------------------------------------------
        # screen
        # ---------------------------------------------------
        screen_width_px = 1920 #1280 for ordi enregistrement
        screen_height_px = 1080 #1024 for ordi enregistrement
        framerate = 60 #100.for ordi enregistrement
        screen = 0 # 1 pour afficher sur l'écran 2 (ne marche pas pour eyeMvt (mac))

        # ---------------------------------------------------
        # stimulus parameters
        # ---------------------------------------------------
        dot_size = 10 # (0.02*screen_height_px)
        saccade_px = .618*screen_height_px
        offset = 0 #.2*screen_height_px

        # ---------------------------------------------------
        # Trial
        # ---------------------------------------------------
        #seed = 51
        trials = [targets[x%4] for x in range(N_trials)]

        # ---------------------------------------------------
        self.param_exp = dict(N_trials=N_trials, trials=trials, #seed=seed,
                              screen=screen, framerate=framerate,
                              screen_width_px=screen_width_px, screen_height_px=screen_height_px,
                              dot_size=dot_size, saccade_px=saccade_px, offset=offset,
                              age=age, observer=self.observer, timeStr=self.timeStr)

    def exp_name(self) :
        return os.path.join(self.datadir, self.timeStr[:10] + '_' + self.observer + '.pkl')

    def run_experiment(self, verb=True) :


        from psychopy import visual, core, event, logging, prefs

        # ---------------------------------------------------
        win = visual.Window([self.param_exp['screen_width_px'], self.param_exp['screen_height_px']], color=(0, 0, 0),
                            allowGUI=False, fullscr=True, screen=self.param_exp['screen'], units='pix') # enlever fullscr=True pour écran 2

        win.setRecordFrameIntervals(True)
        win._refreshThreshold = 1/self.param_exp['framerate'] + 0.004 # i've got 50Hz monitor and want to allow 4ms tolerance


        # ---------------------------------------------------
        if verb : print('FPS =',  win.getActualFrameRate() , 'framerate =', self.param_exp['framerate'])

        # ---------------------------------------------------
        fixation = visual.GratingStim(win, mask='cross', sf=0, color='white', size=self.param_exp['dot_size'])
        target   = visual.GratingStim(win, mask='circle', sf=0, color='white', size=self.param_exp['dot_size'])

        def escape_possible() :
            if event.getKeys(keyList=['escape', 'a', 'q']):
                win.close()
                core.quit()

        # ---------------------------------------------------
        def presentStimulus(dir_target) :
            escape_possible()
            if dir_target=='blink' :
                escape_possible()
                win.color = (1, 1, 1)
                win.flip()
                win.flip()
                core.wait(0.5)

            else :
                target.setPos((dir_target * (self.param_exp['saccade_px']), self.param_exp['offset']))
                target.draw()
                win.flip()
                core.wait(0.5)

            if self.camera != None : frame = self.camera.grab()
            else :              frame = None
            core.wait(0.5)
            win.color = (0, 0, 0)
            escape_possible()
            win.flip()

            return frame

        # ---------------------------------------------------
        # EXPERIMENT
        # ---------------------------------------------------

        for i, trial in enumerate(self.param_exp['trials']) :

            # ---------------------------------------------------
            # FIXATION
            # ---------------------------------------------------
            fixation.draw()
            escape_possible()
            win.flip()
            core.wait(1)

            # ---------------------------------------------------
            # GAP
            # ---------------------------------------------------
            win.flip()
            escape_possible()
            core.wait(0.3)

            # ---------------------------------------------------
            # Target
            # ---------------------------------------------------
            if   trial == 'left'   : dir_target = -1
            elif trial == 'blink'  : dir_target = 'blink'
            elif trial == 'center' : dir_target = 0
            elif trial == 'right'  : dir_target = 1
            frame = presentStimulus(dir_target)


            if frame != None :
                filename = os.path.join(datadir, self.timeStr[:10] + '_' + self.observer + '_' + '%.3d' % i + '.png')
                imageio.imwrite(filename, frame[:, :, ::-1]) # converting on the fly from BGR > RGB
            else :

                msg_Warning = visual.TextStim(win, text=u"\n\n\nL'image pour le trial %.3d n'a pas pu être enregistré"%i,
                                               font='calibri', height=25, alignHoriz='center')#, alignVert='top')

                win.flip()
                msg_Warning.draw()
                escape_possible()
                win.flip()
                core.wait(2)

                print("L'image pour le trial %.3d n'a pas pu être enregistré"%i)

        #''''''''''''''''''''''''''''''''''''''''''''
        # enregistre param_exp
        #''''''''''''''''''''''''''''''''''''''''''''
        with open(self.exp_name(), 'wb') as fichier:
            f = pickle.Pickler(fichier)
            f.dump(self.param_exp)

        win.update()
        core.wait(0.5)
        win.saveFrameIntervals(fileName=None, clear=True)

        win.close()
        core.quit()


if __name__ == '__main__':

    import time
    timeStr = time.strftime("%Y-%m-%d_%H%M%S", time.localtime())

    e = Experiment(timeStr)
    print('Starting protocol')
    e.run_experiment()
