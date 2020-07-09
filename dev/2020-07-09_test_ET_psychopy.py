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

def parse() :

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-o",       "--observer",           default='no-name',  type=str,       help="nom de l'observateur")
    parser.add_argument("-nb_t",    "--N_trials",           default=120,        type=int,       help="nombre d'essais")
    parser.add_argument("-a",       "--age",                default=20,         type=int,       help="observer's age")
    parser.add_argument("-s_w",     "--screen_width_px",    default=1920,       type=int,       help="largeur de l'écran (en pixel)")
    parser.add_argument("-s_h",     "--screen_height_px",   default=1080,       type=int,       help="hauteur de l'écran (en pixel)")
    parser.add_argument("-f",       "--framerate",          default=60,         type=float,     help="taux de rafraichisement de l'écran")
    parser.add_argument("-t_s",     "--target_size",        default=0.05,       type=float,     help="taille de la cible relative à l'écran'")
    parser.add_argument("-t_dur",   "--target_dur",         default=.7,         type=float,     help="durée de la cible  /!\ doit être superieur a photo_time")
    parser.add_argument("-p_t",     "--photo_time",         default=.35,         type=float,     help="temps où la photo sera prise après l'apparition de la cible")
    parser.add_argument("-f_dur",   "--fix_dur",            default=.15,         type=float,     help="durée de la fixation")
    parser.add_argument("-g_dur",   "--gap_dur",            default=.1,         type=float,     help="durée du GAP")
    parser.add_argument("-v",       "--verb",               default=False,      type=bool,      help="Verbose?")

    return parser.parse_args()


class Experiment(object) :

    def __init__(self, args, timeStr, observer='test') :
        self.args = args
        self.init()

    def init(self) :

        # ---------------------------------------------------
        # Dossier
        # ---------------------------------------------------
        self.datadir = 'dataset'
        targets = ['left', 'blink', 'center', 'right']

        import os
        os.makedirs(self.datadir, exist_ok=True)
        for target in targets :
            if self.args.verb: print('target:', os.path.join(self.datadir, target))
            os.makedirs(os.path.join(self.datadir, target), exist_ok=True)

        #----------------------------------------------------
        # Caméra
        #----------------------------------------------------
        try :
            et = Server()
            self.camera = et.cam
        except :
            self.camera = None

        #----------------------------------------------------


        import time
        timeStr = time.strftime("%Y-%m-%d_%H%M%S", time.localtime())
        self.timeStr = str(timeStr)
        self.datadir = 'dataset'

        if os.path.exists(self.exp_name()) :
            observer = self.args.observer

            i=0
            while os.path.exists(self.exp_name()) :
                self.args.observer = observer + '_' + str(i)
                if self.args.verb: print(observer, self.args.observer)
                i+=1

        self.exp_name = self.exp_name()

        # ---------------------------------------------------
        # screen
        # ---------------------------------------------------
        screen = 0 # 1 pour afficher sur l'écran 2 (ne marche pas pour eyeMvt (mac))

        # ---------------------------------------------------
        # stimulus parameters
        # ---------------------------------------------------
        target_size = 0.05 # (0.02*screen_height_px) - relative to screen size
        saccade_px = .618 #*self.args.screen_height_px
        offset = 0 #.2*screen_height_px

        # ---------------------------------------------------
        # Trial
        # ---------------------------------------------------
        #seed = 51
        trials = [targets[x%len(targets)] for x in range(self.args.N_trials)]
        np.random.shuffle(trials)

        # ---------------------------------------------------
        self.param_exp = dict(N_trials=self.args.N_trials, trials=trials, #seed=seed,
                              screen=screen, framerate=self.args.framerate,
                              screen_width_px=self.args.screen_width_px, screen_height_px=self.args.screen_height_px,
                              target_size=self.args.target_size, saccade_px=saccade_px, offset=offset,
                              observer=self.args.observer, age=self.args.age, timeStr=self.timeStr,
                              target_dur=self.args.target_dur, photo_time=self.args.photo_time,
                              fix_dur=self.args.fix_dur, gap_dur=self.args.gap_dur
                              )

    def exp_name(self, ext='.pkl') :
        return os.path.join(self.datadir, self.timeStr[:10] + '_' + self.args.observer + '_' + str(self.args.age) + ext)

    def save_param_exp(self):
        self.param_exp['N_trials'] = self.trial
        self.param_exp['trials']   = self.param_exp['trials'][:self.trial]
        print(self.param_exp)

        with open(self.exp_name, 'wb') as fichier:
            f = pickle.Pickler(fichier)
            f.dump(self.param_exp)


    def run_experiment(self, verb=True) :

        from psychopy import visual, core, event, logging, prefs

        # ---------------------------------------------------
        win = visual.Window([self.param_exp['screen_width_px'], self.param_exp['screen_height_px']], color=(0, 0, 0),
                            allowGUI=False, fullscr=True, screen=self.param_exp['screen'], units='height') # enlever fullscr=True pour écran 2
        win.setRecordFrameIntervals(True)
        win._refreshThreshold = 1/self.param_exp['framerate'] + 0.004 # i've got 50Hz monitor and want to allow 4ms tolerance


        # ---------------------------------------------------
        if verb : print('FPS =',  win.getActualFrameRate() , 'framerate =', self.param_exp['framerate'])

        # ---------------------------------------------------
        fixation = visual.GratingStim(win, mask='cross', sf=0, color='white', size=self.param_exp['target_size'])
        target   = visual.GratingStim(win, mask='circle', sf=0, color='white', size=self.param_exp['target_size'])

        def possible_escape() :
            if event.getKeys(keyList=['escape', 'a', 'q']):
                self.save_param_exp()
                win.close()
                core.quit()

        # ---------------------------------------------------
        def presentStimulus(trial) :

            if   trial == 'left'   : dir_target = -1
            elif trial == 'center' : dir_target = 0
            elif trial == 'right'  : dir_target = 1
            elif trial == 'blink'  : dir_target = None

            possible_escape()
            if dir_target is None :
                for _ in range(8):
                    win.color = (1, 1, 1)
                    win.flip()
                    win.flip()
                    win.color = (0, 0, 0)
                    win.flip()
                    win.flip()
                core.wait(self.args.photo_time)

            else :
                target.setPos((dir_target * (self.param_exp['saccade_px']), self.param_exp['offset']))
                target.draw()
                win.flip()
                core.wait(self.args.photo_time)

            if not(self.camera is None): frame = self.camera.grab()
            else :                       frame = None
            core.wait(self.args.target_dur - self.args.photo_time)
            win.color = (0, 0, 0)
            possible_escape()
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
            possible_escape()
            win.flip()
            core.wait(self.args.fix_dur)

            # ---------------------------------------------------
            # GAP
            # ---------------------------------------------------
            win.flip()
            possible_escape()
            core.wait(self.args.gap_dur)

            # ---------------------------------------------------
            # Target
            # ---------------------------------------------------

            frame = presentStimulus(trial)

            if not(frame is None):
                filename = os.path.join(self.datadir, trial, self.timeStr[:10] + '_' + self.args.observer + '__' + '%.3d' % i + '.png')
                imageio.imwrite(filename, frame[:, :, ::-1]) # converting on the fly from BGR > RGB
            else :
                msg_Warning = visual.TextStim(win, text=u"\n\n\ntrial %.3d : non enregistré"%i,
                                               font='calibri', height=.02, anchorHoriz='center', anchorVert='bottom')
                win.flip()
                msg_Warning.draw()
                possible_escape()
                win.flip()
                core.wait(2)
                # TODO : break the loop + stop experiment
                print("trial %.3d : non enregistré"%i)


        #''''''''''''''''''''''''''''''''''''''''''''
        # enregistre param_exp
        #''''''''''''''''''''''''''''''''''''''''''''
        self.trial = i+1
        self.save_param_exp()

        win.update()
        win.saveFrameIntervals(fileName=None, clear=True)

        win.close()
        core.quit()


if __name__ == '__main__':

    import time
    timeStr = time.strftime("%Y-%m-%d_%H%M%S", time.localtime())

    args = parse()

    e = Experiment(args, timeStr)
    print('Starting protocol')
    e.run_experiment()
