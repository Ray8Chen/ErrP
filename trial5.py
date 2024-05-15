#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2023.2.3),
    on Wed May  1 10:39:17 2024
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
correct = 1
error = 0
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
prefs.hardware['audioLib'] = 'ptb'
prefs.hardware['audioLatencyMode'] = '3'
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout
from psychopy.tools import environmenttools
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER, priority)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

import psychopy.iohub as io
from psychopy.hardware import keyboard

from pylsl import StreamInfo, StreamOutlet
# Set up LabStreamingLayer stream.
info = StreamInfo(name='PsychoPy_LSL', type='Markers', channel_count=1, nominal_srate=0, channel_format='int32', source_id='psy_marker')
outlet = StreamOutlet(info)  # Broadcast the stream.

# --- Setup global variables (available in all functions) ---
# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# Store info about the experiment session
psychopyVersion = '2023.2.3'
expName = 'trial5'  # from the Builder filename that created this script
expInfo = {
    'participant': f"{randint(0, 999999):06.0f}",
    'session': '001',
    'date': data.getDateStr(),  # add a simple timestamp
    'expName': expName,
    'psychopyVersion': psychopyVersion,
}


def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # temporarily remove keys which the dialog doesn't need to show
    poppedKeys = {
        'date': expInfo.pop('date', data.getDateStr()),
        'expName': expInfo.pop('expName', expName),
        'psychopyVersion': expInfo.pop('psychopyVersion', psychopyVersion),
    }
    # show participant info dialog
    dlg = gui.DlgFromDict(dictionary=expInfo, sortKeys=False, title=expName)
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # restore hidden keys
    expInfo.update(poppedKeys)
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version='',
        extraInfo=expInfo, runtimeInfo=None,
        originPath='/Users/raychen/Desktop/BCI project/psychopy/trial5.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # this outputs to the screen, not a file
    logging.console.setLevel(logging.EXP)
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log', level=logging.EXP)
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=(1024, 768), fullscr=True, screen=0,
            winType='pyglet', allowStencil=False,
            monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height'
        )
        if expInfo is not None:
            # store frame rate of monitor if we can measure it
            expInfo['frameRate'] = win.getActualFrameRate()
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [0,0,0]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    win.mouseVisible = False
    win.hideMessage()
    return win


def setupInputs(expInfo, thisExp, win):
    """
    Setup whatever inputs are available (mouse, keyboard, eyetracker, etc.)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    dict
        Dictionary of input devices by name.
    """
    # --- Setup input devices ---
    inputs = {}
    ioConfig = {}
    
    # Setup iohub keyboard
    ioConfig['Keyboard'] = dict(use_keymap='psychopy')
    
    ioSession = '1'
    if 'session' in expInfo:
        ioSession = str(expInfo['session'])
    ioServer = io.launchHubServer(window=win, **ioConfig)
    eyetracker = None
    
    # create a default keyboard (e.g. to check for escape)
    defaultKeyboard = keyboard.Keyboard(backend='iohub')
    # return inputs dict
    return {
        'ioServer': ioServer,
        'defaultKeyboard': defaultKeyboard,
        'eyetracker': eyetracker,
    }

def pauseExperiment(thisExp, inputs=None, win=None, timers=[], playbackComponents=[]):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    inputs : dict
        Dictionary of input devices by name.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    playbackComponents : list, tuple
        List of any components with a `pause` method which need to be paused.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # pause any playback components
    for comp in playbackComponents:
        comp.pause()
    # prevent components from auto-drawing
    win.stashAutoDraw()
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # make sure we have a keyboard
        if inputs is None:
            inputs = {
                'defaultKeyboard': keyboard.Keyboard(backend='ioHub')
            }
        # check for quit (typically the Esc key)
        if inputs['defaultKeyboard'].getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win, inputs=inputs)
        # flip the screen
        win.flip()
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, inputs=inputs, win=win)
    # resume any playback components
    for comp in playbackComponents:
        comp.play()
    # restore auto-drawn components
    win.retrieveAutoDraw()
    # reset any timers
    for timer in timers:
        timer.reset()


def run(expInfo, thisExp, win, inputs, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    inputs : dict
        Dictionary of input devices by name.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = inputs['ioServer']
    defaultKeyboard = inputs['defaultKeyboard']
    eyetracker = inputs['eyetracker']
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "welcome" ---
    ready = visual.TextStim(win=win, name='ready',
        text='Ready...',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    start = visual.TextStim(win=win, name='start',
        text='Start',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "L_L" ---
    No1left = visual.ShapeStim(
        win=win, name='No1left', vertices='arrow',
        size=(0.5, 0.5),
        ori=-90.0, pos=(-0.4, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='blue',
        opacity=None, depth=0.0, interpolate=True)
    No1right = visual.ShapeStim(
        win=win, name='No1right', vertices='arrow',
        size=(0.5, 0.5),
        ori=90.0, pos=(0.4, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='red',
        opacity=None, depth=-1.0, interpolate=True)
    left_1 = visual.TextStim(win=win, name='left_1',
        text='LEFT',
        font='Open Sans',
        pos=(0, 0.1), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "left" ---
    left1_5 = visual.ShapeStim(
        win=win, name='left1_5', vertices='arrow',
        size=(0.6, 0.6),
        ori=-90.0, pos=(-0.4, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='blue',
        opacity=None, depth=0.0, interpolate=True)
    right1_5 = visual.ShapeStim(
        win=win, name='right1_5', vertices='arrow',
        size=(0.5, 0.5),
        ori=90.0, pos=(0.4, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='red',
        opacity=None, depth=-1.0, interpolate=True)
    
    # --- Initialize components for Routine "R_L" ---
    No1left_4 = visual.ShapeStim(
        win=win, name='No1left_4', vertices='arrow',
        size=(0.5, 0.5),
        ori=-90.0, pos=(-0.4, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='blue',
        opacity=None, depth=0.0, interpolate=True)
    No1right_4 = visual.ShapeStim(
        win=win, name='No1right_4', vertices='arrow',
        size=(0.5, 0.5),
        ori=90.0, pos=(0.4, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='red',
        opacity=None, depth=-1.0, interpolate=True)
    right_2 = visual.TextStim(win=win, name='right_2',
        text='RIGHT',
        font='Open Sans',
        pos=(0, 0.1), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "left" ---
    left1_5 = visual.ShapeStim(
        win=win, name='left1_5', vertices='arrow',
        size=(0.6, 0.6),
        ori=-90.0, pos=(-0.4, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='blue',
        opacity=None, depth=0.0, interpolate=True)
    right1_5 = visual.ShapeStim(
        win=win, name='right1_5', vertices='arrow',
        size=(0.5, 0.5),
        ori=90.0, pos=(0.4, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='red',
        opacity=None, depth=-1.0, interpolate=True)
    
    # --- Initialize components for Routine "L_L" ---
    No1left = visual.ShapeStim(
        win=win, name='No1left', vertices='arrow',
        size=(0.5, 0.5),
        ori=-90.0, pos=(-0.4, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='blue',
        opacity=None, depth=0.0, interpolate=True)
    No1right = visual.ShapeStim(
        win=win, name='No1right', vertices='arrow',
        size=(0.5, 0.5),
        ori=90.0, pos=(0.4, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='red',
        opacity=None, depth=-1.0, interpolate=True)
    left_1 = visual.TextStim(win=win, name='left_1',
        text='LEFT',
        font='Open Sans',
        pos=(0, 0.1), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "left" ---
    left1_5 = visual.ShapeStim(
        win=win, name='left1_5', vertices='arrow',
        size=(0.6, 0.6),
        ori=-90.0, pos=(-0.4, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='blue',
        opacity=None, depth=0.0, interpolate=True)
    right1_5 = visual.ShapeStim(
        win=win, name='right1_5', vertices='arrow',
        size=(0.5, 0.5),
        ori=90.0, pos=(0.4, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='red',
        opacity=None, depth=-1.0, interpolate=True)
    
    # --- Initialize components for Routine "R_L" ---
    No1left_4 = visual.ShapeStim(
        win=win, name='No1left_4', vertices='arrow',
        size=(0.5, 0.5),
        ori=-90.0, pos=(-0.4, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='blue',
        opacity=None, depth=0.0, interpolate=True)
    No1right_4 = visual.ShapeStim(
        win=win, name='No1right_4', vertices='arrow',
        size=(0.5, 0.5),
        ori=90.0, pos=(0.4, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='red',
        opacity=None, depth=-1.0, interpolate=True)
    right_2 = visual.TextStim(win=win, name='right_2',
        text='RIGHT',
        font='Open Sans',
        pos=(0, 0.1), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "left" ---
    left1_5 = visual.ShapeStim(
        win=win, name='left1_5', vertices='arrow',
        size=(0.6, 0.6),
        ori=-90.0, pos=(-0.4, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='blue',
        opacity=None, depth=0.0, interpolate=True)
    right1_5 = visual.ShapeStim(
        win=win, name='right1_5', vertices='arrow',
        size=(0.5, 0.5),
        ori=90.0, pos=(0.4, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='red',
        opacity=None, depth=-1.0, interpolate=True)
    
    # --- Initialize components for Routine "L_L" ---
    No1left = visual.ShapeStim(
        win=win, name='No1left', vertices='arrow',
        size=(0.5, 0.5),
        ori=-90.0, pos=(-0.4, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='blue',
        opacity=None, depth=0.0, interpolate=True)
    No1right = visual.ShapeStim(
        win=win, name='No1right', vertices='arrow',
        size=(0.5, 0.5),
        ori=90.0, pos=(0.4, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='red',
        opacity=None, depth=-1.0, interpolate=True)
    left_1 = visual.TextStim(win=win, name='left_1',
        text='LEFT',
        font='Open Sans',
        pos=(0, 0.1), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "left" ---
    left1_5 = visual.ShapeStim(
        win=win, name='left1_5', vertices='arrow',
        size=(0.6, 0.6),
        ori=-90.0, pos=(-0.4, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='blue',
        opacity=None, depth=0.0, interpolate=True)
    right1_5 = visual.ShapeStim(
        win=win, name='right1_5', vertices='arrow',
        size=(0.5, 0.5),
        ori=90.0, pos=(0.4, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='red',
        opacity=None, depth=-1.0, interpolate=True)
    
    # --- Initialize components for Routine "R_L" ---
    No1left_4 = visual.ShapeStim(
        win=win, name='No1left_4', vertices='arrow',
        size=(0.5, 0.5),
        ori=-90.0, pos=(-0.4, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='blue',
        opacity=None, depth=0.0, interpolate=True)
    No1right_4 = visual.ShapeStim(
        win=win, name='No1right_4', vertices='arrow',
        size=(0.5, 0.5),
        ori=90.0, pos=(0.4, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='red',
        opacity=None, depth=-1.0, interpolate=True)
    right_2 = visual.TextStim(win=win, name='right_2',
        text='RIGHT',
        font='Open Sans',
        pos=(0, 0.1), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "left" ---
    left1_5 = visual.ShapeStim(
        win=win, name='left1_5', vertices='arrow',
        size=(0.6, 0.6),
        ori=-90.0, pos=(-0.4, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='blue',
        opacity=None, depth=0.0, interpolate=True)
    right1_5 = visual.ShapeStim(
        win=win, name='right1_5', vertices='arrow',
        size=(0.5, 0.5),
        ori=90.0, pos=(0.4, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='red',
        opacity=None, depth=-1.0, interpolate=True)
    
    # --- Initialize components for Routine "L_R" ---
    No1left_3 = visual.ShapeStim(
        win=win, name='No1left_3', vertices='arrow',
        size=(0.5, 0.5),
        ori=-90.0, pos=(-0.4, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='blue',
        opacity=None, depth=0.0, interpolate=True)
    No1right_3 = visual.ShapeStim(
        win=win, name='No1right_3', vertices='arrow',
        size=(0.5, 0.5),
        ori=90.0, pos=(0.4, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='red',
        opacity=None, depth=-1.0, interpolate=True)
    left_2 = visual.TextStim(win=win, name='left_2',
        text='LEFT',
        font='Open Sans',
        pos=(0, 0.1), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "Right" ---
    left1_6 = visual.ShapeStim(
        win=win, name='left1_6', vertices='arrow',
        size=(0.5, 0.5),
        ori=-90.0, pos=(-0.4, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='blue',
        opacity=None, depth=0.0, interpolate=True)
    right1_6 = visual.ShapeStim(
        win=win, name='right1_6', vertices='arrow',
        size=(0.6, 0.6),
        ori=90.0, pos=(0.4, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='red',
        opacity=None, depth=-1.0, interpolate=True)
    
    # --- Initialize components for Routine "R_R" ---
    No1left_2 = visual.ShapeStim(
        win=win, name='No1left_2', vertices='arrow',
        size=(0.5, 0.5),
        ori=-90.0, pos=(-0.4, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='blue',
        opacity=None, depth=0.0, interpolate=True)
    No1right_2 = visual.ShapeStim(
        win=win, name='No1right_2', vertices='arrow',
        size=(0.5, 0.5),
        ori=90.0, pos=(0.4, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='red',
        opacity=None, depth=-1.0, interpolate=True)
    right = visual.TextStim(win=win, name='right',
        text='RIGHT',
        font='Open Sans',
        pos=(0, 0.1), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "Right" ---
    left1_6 = visual.ShapeStim(
        win=win, name='left1_6', vertices='arrow',
        size=(0.5, 0.5),
        ori=-90.0, pos=(-0.4, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='blue',
        opacity=None, depth=0.0, interpolate=True)
    right1_6 = visual.ShapeStim(
        win=win, name='right1_6', vertices='arrow',
        size=(0.6, 0.6),
        ori=90.0, pos=(0.4, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='red',
        opacity=None, depth=-1.0, interpolate=True)
    
    # --- Initialize components for Routine "L_R" ---
    No1left_3 = visual.ShapeStim(
        win=win, name='No1left_3', vertices='arrow',
        size=(0.5, 0.5),
        ori=-90.0, pos=(-0.4, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='blue',
        opacity=None, depth=0.0, interpolate=True)
    No1right_3 = visual.ShapeStim(
        win=win, name='No1right_3', vertices='arrow',
        size=(0.5, 0.5),
        ori=90.0, pos=(0.4, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='red',
        opacity=None, depth=-1.0, interpolate=True)
    left_2 = visual.TextStim(win=win, name='left_2',
        text='LEFT',
        font='Open Sans',
        pos=(0, 0.1), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "Right" ---
    left1_6 = visual.ShapeStim(
        win=win, name='left1_6', vertices='arrow',
        size=(0.5, 0.5),
        ori=-90.0, pos=(-0.4, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='blue',
        opacity=None, depth=0.0, interpolate=True)
    right1_6 = visual.ShapeStim(
        win=win, name='right1_6', vertices='arrow',
        size=(0.6, 0.6),
        ori=90.0, pos=(0.4, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='red',
        opacity=None, depth=-1.0, interpolate=True)
    
    # --- Initialize components for Routine "R_R" ---
    No1left_2 = visual.ShapeStim(
        win=win, name='No1left_2', vertices='arrow',
        size=(0.5, 0.5),
        ori=-90.0, pos=(-0.4, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='blue',
        opacity=None, depth=0.0, interpolate=True)
    No1right_2 = visual.ShapeStim(
        win=win, name='No1right_2', vertices='arrow',
        size=(0.5, 0.5),
        ori=90.0, pos=(0.4, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='red',
        opacity=None, depth=-1.0, interpolate=True)
    right = visual.TextStim(win=win, name='right',
        text='RIGHT',
        font='Open Sans',
        pos=(0, 0.1), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "Right" ---
    left1_6 = visual.ShapeStim(
        win=win, name='left1_6', vertices='arrow',
        size=(0.5, 0.5),
        ori=-90.0, pos=(-0.4, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='blue',
        opacity=None, depth=0.0, interpolate=True)
    right1_6 = visual.ShapeStim(
        win=win, name='right1_6', vertices='arrow',
        size=(0.6, 0.6),
        ori=90.0, pos=(0.4, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='red',
        opacity=None, depth=-1.0, interpolate=True)
    
    # --- Initialize components for Routine "L_L" ---
    No1left = visual.ShapeStim(
        win=win, name='No1left', vertices='arrow',
        size=(0.5, 0.5),
        ori=-90.0, pos=(-0.4, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='blue',
        opacity=None, depth=0.0, interpolate=True)
    No1right = visual.ShapeStim(
        win=win, name='No1right', vertices='arrow',
        size=(0.5, 0.5),
        ori=90.0, pos=(0.4, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='red',
        opacity=None, depth=-1.0, interpolate=True)
    left_1 = visual.TextStim(win=win, name='left_1',
        text='LEFT',
        font='Open Sans',
        pos=(0, 0.1), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "left" ---
    left1_5 = visual.ShapeStim(
        win=win, name='left1_5', vertices='arrow',
        size=(0.6, 0.6),
        ori=-90.0, pos=(-0.4, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='blue',
        opacity=None, depth=0.0, interpolate=True)
    right1_5 = visual.ShapeStim(
        win=win, name='right1_5', vertices='arrow',
        size=(0.5, 0.5),
        ori=90.0, pos=(0.4, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='red',
        opacity=None, depth=-1.0, interpolate=True)
    
    # --- Initialize components for Routine "R_R" ---
    No1left_2 = visual.ShapeStim(
        win=win, name='No1left_2', vertices='arrow',
        size=(0.5, 0.5),
        ori=-90.0, pos=(-0.4, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='blue',
        opacity=None, depth=0.0, interpolate=True)
    No1right_2 = visual.ShapeStim(
        win=win, name='No1right_2', vertices='arrow',
        size=(0.5, 0.5),
        ori=90.0, pos=(0.4, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='red',
        opacity=None, depth=-1.0, interpolate=True)
    right = visual.TextStim(win=win, name='right',
        text='RIGHT',
        font='Open Sans',
        pos=(0, 0.1), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "Right" ---
    left1_6 = visual.ShapeStim(
        win=win, name='left1_6', vertices='arrow',
        size=(0.5, 0.5),
        ori=-90.0, pos=(-0.4, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='blue',
        opacity=None, depth=0.0, interpolate=True)
    right1_6 = visual.ShapeStim(
        win=win, name='right1_6', vertices='arrow',
        size=(0.6, 0.6),
        ori=90.0, pos=(0.4, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='red',
        opacity=None, depth=-1.0, interpolate=True)
    
    # --- Initialize components for Routine "L_R" ---
    No1left_3 = visual.ShapeStim(
        win=win, name='No1left_3', vertices='arrow',
        size=(0.5, 0.5),
        ori=-90.0, pos=(-0.4, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='blue',
        opacity=None, depth=0.0, interpolate=True)
    No1right_3 = visual.ShapeStim(
        win=win, name='No1right_3', vertices='arrow',
        size=(0.5, 0.5),
        ori=90.0, pos=(0.4, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='red',
        opacity=None, depth=-1.0, interpolate=True)
    left_2 = visual.TextStim(win=win, name='left_2',
        text='LEFT',
        font='Open Sans',
        pos=(0, 0.1), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "Right" ---
    left1_6 = visual.ShapeStim(
        win=win, name='left1_6', vertices='arrow',
        size=(0.5, 0.5),
        ori=-90.0, pos=(-0.4, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='blue',
        opacity=None, depth=0.0, interpolate=True)
    right1_6 = visual.ShapeStim(
        win=win, name='right1_6', vertices='arrow',
        size=(0.6, 0.6),
        ori=90.0, pos=(0.4, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='red',
        opacity=None, depth=-1.0, interpolate=True)
    
    # --- Initialize components for Routine "R_R" ---
    No1left_2 = visual.ShapeStim(
        win=win, name='No1left_2', vertices='arrow',
        size=(0.5, 0.5),
        ori=-90.0, pos=(-0.4, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='blue',
        opacity=None, depth=0.0, interpolate=True)
    No1right_2 = visual.ShapeStim(
        win=win, name='No1right_2', vertices='arrow',
        size=(0.5, 0.5),
        ori=90.0, pos=(0.4, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='red',
        opacity=None, depth=-1.0, interpolate=True)
    right = visual.TextStim(win=win, name='right',
        text='RIGHT',
        font='Open Sans',
        pos=(0, 0.1), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "Right" ---
    left1_6 = visual.ShapeStim(
        win=win, name='left1_6', vertices='arrow',
        size=(0.5, 0.5),
        ori=-90.0, pos=(-0.4, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='blue',
        opacity=None, depth=0.0, interpolate=True)
    right1_6 = visual.ShapeStim(
        win=win, name='right1_6', vertices='arrow',
        size=(0.6, 0.6),
        ori=90.0, pos=(0.4, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='red',
        opacity=None, depth=-1.0, interpolate=True)
    
    # --- Initialize components for Routine "L_L" ---
    No1left = visual.ShapeStim(
        win=win, name='No1left', vertices='arrow',
        size=(0.5, 0.5),
        ori=-90.0, pos=(-0.4, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='blue',
        opacity=None, depth=0.0, interpolate=True)
    No1right = visual.ShapeStim(
        win=win, name='No1right', vertices='arrow',
        size=(0.5, 0.5),
        ori=90.0, pos=(0.4, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='red',
        opacity=None, depth=-1.0, interpolate=True)
    left_1 = visual.TextStim(win=win, name='left_1',
        text='LEFT',
        font='Open Sans',
        pos=(0, 0.1), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "left" ---
    left1_5 = visual.ShapeStim(
        win=win, name='left1_5', vertices='arrow',
        size=(0.6, 0.6),
        ori=-90.0, pos=(-0.4, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='blue',
        opacity=None, depth=0.0, interpolate=True)
    right1_5 = visual.ShapeStim(
        win=win, name='right1_5', vertices='arrow',
        size=(0.5, 0.5),
        ori=90.0, pos=(0.4, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='red',
        opacity=None, depth=-1.0, interpolate=True)
    
    # --- Initialize components for Routine "R_L" ---
    No1left_4 = visual.ShapeStim(
        win=win, name='No1left_4', vertices='arrow',
        size=(0.5, 0.5),
        ori=-90.0, pos=(-0.4, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='blue',
        opacity=None, depth=0.0, interpolate=True)
    No1right_4 = visual.ShapeStim(
        win=win, name='No1right_4', vertices='arrow',
        size=(0.5, 0.5),
        ori=90.0, pos=(0.4, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='red',
        opacity=None, depth=-1.0, interpolate=True)
    right_2 = visual.TextStim(win=win, name='right_2',
        text='RIGHT',
        font='Open Sans',
        pos=(0, 0.1), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "left" ---
    left1_5 = visual.ShapeStim(
        win=win, name='left1_5', vertices='arrow',
        size=(0.6, 0.6),
        ori=-90.0, pos=(-0.4, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='blue',
        opacity=None, depth=0.0, interpolate=True)
    right1_5 = visual.ShapeStim(
        win=win, name='right1_5', vertices='arrow',
        size=(0.5, 0.5),
        ori=90.0, pos=(0.4, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='red',
        opacity=None, depth=-1.0, interpolate=True)
    
    # --- Initialize components for Routine "L_L" ---
    No1left = visual.ShapeStim(
        win=win, name='No1left', vertices='arrow',
        size=(0.5, 0.5),
        ori=-90.0, pos=(-0.4, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='blue',
        opacity=None, depth=0.0, interpolate=True)
    No1right = visual.ShapeStim(
        win=win, name='No1right', vertices='arrow',
        size=(0.5, 0.5),
        ori=90.0, pos=(0.4, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='red',
        opacity=None, depth=-1.0, interpolate=True)
    left_1 = visual.TextStim(win=win, name='left_1',
        text='LEFT',
        font='Open Sans',
        pos=(0, 0.1), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "left" ---
    left1_5 = visual.ShapeStim(
        win=win, name='left1_5', vertices='arrow',
        size=(0.6, 0.6),
        ori=-90.0, pos=(-0.4, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='blue',
        opacity=None, depth=0.0, interpolate=True)
    right1_5 = visual.ShapeStim(
        win=win, name='right1_5', vertices='arrow',
        size=(0.5, 0.5),
        ori=90.0, pos=(0.4, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='red',
        opacity=None, depth=-1.0, interpolate=True)
    
    # --- Initialize components for Routine "R_R" ---
    No1left_2 = visual.ShapeStim(
        win=win, name='No1left_2', vertices='arrow',
        size=(0.5, 0.5),
        ori=-90.0, pos=(-0.4, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='blue',
        opacity=None, depth=0.0, interpolate=True)
    No1right_2 = visual.ShapeStim(
        win=win, name='No1right_2', vertices='arrow',
        size=(0.5, 0.5),
        ori=90.0, pos=(0.4, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='red',
        opacity=None, depth=-1.0, interpolate=True)
    right = visual.TextStim(win=win, name='right',
        text='RIGHT',
        font='Open Sans',
        pos=(0, 0.1), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "Right" ---
    left1_6 = visual.ShapeStim(
        win=win, name='left1_6', vertices='arrow',
        size=(0.5, 0.5),
        ori=-90.0, pos=(-0.4, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='blue',
        opacity=None, depth=0.0, interpolate=True)
    right1_6 = visual.ShapeStim(
        win=win, name='right1_6', vertices='arrow',
        size=(0.6, 0.6),
        ori=90.0, pos=(0.4, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='red',
        opacity=None, depth=-1.0, interpolate=True)
    
    # --- Initialize components for Routine "L_L" ---
    No1left = visual.ShapeStim(
        win=win, name='No1left', vertices='arrow',
        size=(0.5, 0.5),
        ori=-90.0, pos=(-0.4, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='blue',
        opacity=None, depth=0.0, interpolate=True)
    No1right = visual.ShapeStim(
        win=win, name='No1right', vertices='arrow',
        size=(0.5, 0.5),
        ori=90.0, pos=(0.4, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='red',
        opacity=None, depth=-1.0, interpolate=True)
    left_1 = visual.TextStim(win=win, name='left_1',
        text='LEFT',
        font='Open Sans',
        pos=(0, 0.1), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "left" ---
    left1_5 = visual.ShapeStim(
        win=win, name='left1_5', vertices='arrow',
        size=(0.6, 0.6),
        ori=-90.0, pos=(-0.4, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='blue',
        opacity=None, depth=0.0, interpolate=True)
    right1_5 = visual.ShapeStim(
        win=win, name='right1_5', vertices='arrow',
        size=(0.5, 0.5),
        ori=90.0, pos=(0.4, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='red',
        opacity=None, depth=-1.0, interpolate=True)
    
    # --- Initialize components for Routine "R_R" ---
    No1left_2 = visual.ShapeStim(
        win=win, name='No1left_2', vertices='arrow',
        size=(0.5, 0.5),
        ori=-90.0, pos=(-0.4, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='blue',
        opacity=None, depth=0.0, interpolate=True)
    No1right_2 = visual.ShapeStim(
        win=win, name='No1right_2', vertices='arrow',
        size=(0.5, 0.5),
        ori=90.0, pos=(0.4, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='red',
        opacity=None, depth=-1.0, interpolate=True)
    right = visual.TextStim(win=win, name='right',
        text='RIGHT',
        font='Open Sans',
        pos=(0, 0.1), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "Right" ---
    left1_6 = visual.ShapeStim(
        win=win, name='left1_6', vertices='arrow',
        size=(0.5, 0.5),
        ori=-90.0, pos=(-0.4, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='blue',
        opacity=None, depth=0.0, interpolate=True)
    right1_6 = visual.ShapeStim(
        win=win, name='right1_6', vertices='arrow',
        size=(0.6, 0.6),
        ori=90.0, pos=(0.4, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='red',
        opacity=None, depth=-1.0, interpolate=True)
    
    # --- Initialize components for Routine "end" ---
    finish = visual.TextStim(win=win, name='finish',
        text='FINISHED',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # create some handy timers
    if globalClock is None:
        globalClock = core.Clock()  # to track the time since experiment started
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    routineTimer = core.Clock()  # to track time remaining of each (possibly non-slip) routine
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6)
    
    # --- Prepare to start Routine "welcome" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('welcome.started', globalClock.getTime())
    # keep track of which components have finished
    welcomeComponents = [ready, start]
    for thisComponent in welcomeComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "welcome" ---
    routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 6.5:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *ready* updates
        
        # if ready is starting this frame...
        if ready.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            ready.frameNStart = frameN  # exact frame index
            ready.tStart = t  # local t and not account for scr refresh
            ready.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(ready, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'ready.started')
            # update status
            ready.status = STARTED
            ready.setAutoDraw(True)
        
        # if ready is active this frame...
        if ready.status == STARTED:
            # update params
            pass
        
        # if ready is stopping this frame...
        if ready.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > ready.tStartRefresh + 3.0-frameTolerance:
                # keep track of stop time/frame for later
                ready.tStop = t  # not accounting for scr refresh
                ready.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'ready.stopped')
                # update status
                ready.status = FINISHED
                ready.setAutoDraw(False)
        
        # *start* updates
        
        # if start is starting this frame...
        if start.status == NOT_STARTED and tThisFlip >= 3.5-frameTolerance:
            # keep track of start time/frame for later
            start.frameNStart = frameN  # exact frame index
            start.tStart = t  # local t and not account for scr refresh
            start.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(start, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'start.started')
            # update status
            start.status = STARTED
            start.setAutoDraw(True)
        
        # if start is active this frame...
        if start.status == STARTED:
            # update params
            pass
        
        # if start is stopping this frame...
        if start.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > start.tStartRefresh + 3.0-frameTolerance:
                # keep track of stop time/frame for later
                start.tStop = t  # not accounting for scr refresh
                start.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'start.stopped')
                # update status
                start.status = FINISHED
                start.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in welcomeComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "welcome" ---
    for thisComponent in welcomeComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('welcome.stopped', globalClock.getTime())
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if routineForceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-6.500000)
    
    # --- Prepare to start Routine "L_L" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('L_L.started', globalClock.getTime())
    # keep track of which components have finished
    L_LComponents = [No1left, No1right, left_1]
    for thisComponent in L_LComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "L_L" ---
    routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 8.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *No1left* updates
        
        # if No1left is starting this frame...
        if No1left.status == NOT_STARTED and tThisFlip >= 5.0-frameTolerance:
            # keep track of start time/frame for later
            No1left.frameNStart = frameN  # exact frame index
            No1left.tStart = t  # local t and not account for scr refresh
            No1left.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(No1left, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'No1left.started')
            # update status
            No1left.status = STARTED
            No1left.setAutoDraw(True)
        
        # if No1left is active this frame...
        if No1left.status == STARTED:
            # update params
            pass
        
        # if No1left is stopping this frame...
        if No1left.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > No1left.tStartRefresh + 3.0-frameTolerance:
                # keep track of stop time/frame for later
                No1left.tStop = t  # not accounting for scr refresh
                No1left.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'No1left.stopped')
                # update status
                No1left.status = FINISHED
                No1left.setAutoDraw(False)
        
        # *No1right* updates
        
        # if No1right is starting this frame...
        if No1right.status == NOT_STARTED and tThisFlip >= 5.0-frameTolerance:
            # keep track of start time/frame for later
            No1right.frameNStart = frameN  # exact frame index
            No1right.tStart = t  # local t and not account for scr refresh
            No1right.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(No1right, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'No1right.started')
            # update status
            No1right.status = STARTED
            No1right.setAutoDraw(True)
        
        # if No1right is active this frame...
        if No1right.status == STARTED:
            # update params
            pass
        
        # if No1right is stopping this frame...
        if No1right.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > No1right.tStartRefresh + 3.0-frameTolerance:
                # keep track of stop time/frame for later
                No1right.tStop = t  # not accounting for scr refresh
                No1right.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'No1right.stopped')
                # update status
                No1right.status = FINISHED
                No1right.setAutoDraw(False)
        
        # *left_1* updates
        
        # if left_1 is starting this frame...
        if left_1.status == NOT_STARTED and tThisFlip >= 5.0-frameTolerance:
            # keep track of start time/frame for later
            left_1.frameNStart = frameN  # exact frame index
            left_1.tStart = t  # local t and not account for scr refresh
            left_1.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(left_1, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'left_1.started')
            # update status
            left_1.status = STARTED
            left_1.setAutoDraw(True)
        
        # if left_1 is active this frame...
        if left_1.status == STARTED:
            # update params
            pass
        
        # if left_1 is stopping this frame...
        if left_1.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > left_1.tStartRefresh + 3.0-frameTolerance:
                # keep track of stop time/frame for later
                left_1.tStop = t  # not accounting for scr refresh
                left_1.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'left_1.stopped')
                # update status
                left_1.status = FINISHED
                left_1.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in L_LComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "L_L" ---
    for thisComponent in L_LComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('L_L.stopped', globalClock.getTime())
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if routineForceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-8.000000)
    
    # --- Prepare to start Routine "left" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('left.started', globalClock.getTime())
    # keep track of which components have finished
    leftComponents = [left1_5, right1_5]
    for thisComponent in leftComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    outlet.push_sample([correct])  # Push event marker.
    print("correct")
    # --- Run Routine "left" ---
    routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 3.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *left1_5* updates
        
        # if left1_5 is starting this frame...
        if left1_5.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            left1_5.frameNStart = frameN  # exact frame index
            left1_5.tStart = t  # local t and not account for scr refresh
            left1_5.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(left1_5, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'left1_5.started')
            # update status
            left1_5.status = STARTED
            left1_5.setAutoDraw(True)
        
        # if left1_5 is active this frame...
        if left1_5.status == STARTED:
            # update params
            pass
        
        # if left1_5 is stopping this frame...
        if left1_5.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > left1_5.tStartRefresh + 3.0-frameTolerance:
                # keep track of stop time/frame for later
                left1_5.tStop = t  # not accounting for scr refresh
                left1_5.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'left1_5.stopped')
                # update status
                left1_5.status = FINISHED
                left1_5.setAutoDraw(False)
        
        # *right1_5* updates
        
        # if right1_5 is starting this frame...
        if right1_5.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            right1_5.frameNStart = frameN  # exact frame index
            right1_5.tStart = t  # local t and not account for scr refresh
            right1_5.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(right1_5, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'right1_5.started')
            # update status
            right1_5.status = STARTED
            right1_5.setAutoDraw(True)
        
        # if right1_5 is active this frame...
        if right1_5.status == STARTED:
            # update params
            pass
        
        # if right1_5 is stopping this frame...
        if right1_5.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > right1_5.tStartRefresh + 3.0-frameTolerance:
                # keep track of stop time/frame for later
                right1_5.tStop = t  # not accounting for scr refresh
                right1_5.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'right1_5.stopped')
                # update status
                right1_5.status = FINISHED
                right1_5.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in leftComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "left" ---
    for thisComponent in leftComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('left.stopped', globalClock.getTime())
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if routineForceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-3.000000)
    
    # --- Prepare to start Routine "R_L" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('R_L.started', globalClock.getTime())
    # keep track of which components have finished
    R_LComponents = [No1left_4, No1right_4, right_2]
    for thisComponent in R_LComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "R_L" ---
    routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 8.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *No1left_4* updates
        
        # if No1left_4 is starting this frame...
        if No1left_4.status == NOT_STARTED and tThisFlip >= 5.0-frameTolerance:
            # keep track of start time/frame for later
            No1left_4.frameNStart = frameN  # exact frame index
            No1left_4.tStart = t  # local t and not account for scr refresh
            No1left_4.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(No1left_4, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'No1left_4.started')
            # update status
            No1left_4.status = STARTED
            No1left_4.setAutoDraw(True)
        
        # if No1left_4 is active this frame...
        if No1left_4.status == STARTED:
            # update params
            pass
        
        # if No1left_4 is stopping this frame...
        if No1left_4.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > No1left_4.tStartRefresh + 3.0-frameTolerance:
                # keep track of stop time/frame for later
                No1left_4.tStop = t  # not accounting for scr refresh
                No1left_4.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'No1left_4.stopped')
                # update status
                No1left_4.status = FINISHED
                No1left_4.setAutoDraw(False)
        
        # *No1right_4* updates
        
        # if No1right_4 is starting this frame...
        if No1right_4.status == NOT_STARTED and tThisFlip >= 5.0-frameTolerance:
            # keep track of start time/frame for later
            No1right_4.frameNStart = frameN  # exact frame index
            No1right_4.tStart = t  # local t and not account for scr refresh
            No1right_4.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(No1right_4, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'No1right_4.started')
            # update status
            No1right_4.status = STARTED
            No1right_4.setAutoDraw(True)
        
        # if No1right_4 is active this frame...
        if No1right_4.status == STARTED:
            # update params
            pass
        
        # if No1right_4 is stopping this frame...
        if No1right_4.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > No1right_4.tStartRefresh + 3.0-frameTolerance:
                # keep track of stop time/frame for later
                No1right_4.tStop = t  # not accounting for scr refresh
                No1right_4.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'No1right_4.stopped')
                # update status
                No1right_4.status = FINISHED
                No1right_4.setAutoDraw(False)
        
        # *right_2* updates
        
        # if right_2 is starting this frame...
        if right_2.status == NOT_STARTED and tThisFlip >= 5.0-frameTolerance:
            # keep track of start time/frame for later
            right_2.frameNStart = frameN  # exact frame index
            right_2.tStart = t  # local t and not account for scr refresh
            right_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(right_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'right_2.started')
            # update status
            right_2.status = STARTED
            right_2.setAutoDraw(True)
        
        # if right_2 is active this frame...
        if right_2.status == STARTED:
            # update params
            pass
        
        # if right_2 is stopping this frame...
        if right_2.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > right_2.tStartRefresh + 3.0-frameTolerance:
                # keep track of stop time/frame for later
                right_2.tStop = t  # not accounting for scr refresh
                right_2.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'right_2.stopped')
                # update status
                right_2.status = FINISHED
                right_2.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in R_LComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "R_L" ---
    for thisComponent in R_LComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('R_L.stopped', globalClock.getTime())
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if routineForceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-8.000000)
    
    # --- Prepare to start Routine "left" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('left.started', globalClock.getTime())
    # keep track of which components have finished
    leftComponents = [left1_5, right1_5]
    for thisComponent in leftComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    outlet.push_sample([error])  # Push event marker.
    print("error")
    # --- Run Routine "left" ---
    routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 3.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *left1_5* updates
        
        # if left1_5 is starting this frame...
        if left1_5.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            left1_5.frameNStart = frameN  # exact frame index
            left1_5.tStart = t  # local t and not account for scr refresh
            left1_5.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(left1_5, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'left1_5.started')
            # update status
            left1_5.status = STARTED
            left1_5.setAutoDraw(True)
        
        # if left1_5 is active this frame...
        if left1_5.status == STARTED:
            # update params
            pass
        
        # if left1_5 is stopping this frame...
        if left1_5.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > left1_5.tStartRefresh + 3.0-frameTolerance:
                # keep track of stop time/frame for later
                left1_5.tStop = t  # not accounting for scr refresh
                left1_5.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'left1_5.stopped')
                # update status
                left1_5.status = FINISHED
                left1_5.setAutoDraw(False)
        
        # *right1_5* updates
        
        # if right1_5 is starting this frame...
        if right1_5.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            right1_5.frameNStart = frameN  # exact frame index
            right1_5.tStart = t  # local t and not account for scr refresh
            right1_5.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(right1_5, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'right1_5.started')
            # update status
            right1_5.status = STARTED
            right1_5.setAutoDraw(True)
        
        # if right1_5 is active this frame...
        if right1_5.status == STARTED:
            # update params
            pass
        
        # if right1_5 is stopping this frame...
        if right1_5.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > right1_5.tStartRefresh + 3.0-frameTolerance:
                # keep track of stop time/frame for later
                right1_5.tStop = t  # not accounting for scr refresh
                right1_5.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'right1_5.stopped')
                # update status
                right1_5.status = FINISHED
                right1_5.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in leftComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "left" ---
    for thisComponent in leftComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('left.stopped', globalClock.getTime())
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if routineForceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-3.000000)
    
    # --- Prepare to start Routine "L_L" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('L_L.started', globalClock.getTime())
    # keep track of which components have finished
    L_LComponents = [No1left, No1right, left_1]
    for thisComponent in L_LComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "L_L" ---
    routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 8.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *No1left* updates
        
        # if No1left is starting this frame...
        if No1left.status == NOT_STARTED and tThisFlip >= 5.0-frameTolerance:
            # keep track of start time/frame for later
            No1left.frameNStart = frameN  # exact frame index
            No1left.tStart = t  # local t and not account for scr refresh
            No1left.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(No1left, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'No1left.started')
            # update status
            No1left.status = STARTED
            No1left.setAutoDraw(True)
        
        # if No1left is active this frame...
        if No1left.status == STARTED:
            # update params
            pass
        
        # if No1left is stopping this frame...
        if No1left.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > No1left.tStartRefresh + 3.0-frameTolerance:
                # keep track of stop time/frame for later
                No1left.tStop = t  # not accounting for scr refresh
                No1left.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'No1left.stopped')
                # update status
                No1left.status = FINISHED
                No1left.setAutoDraw(False)
        
        # *No1right* updates
        
        # if No1right is starting this frame...
        if No1right.status == NOT_STARTED and tThisFlip >= 5.0-frameTolerance:
            # keep track of start time/frame for later
            No1right.frameNStart = frameN  # exact frame index
            No1right.tStart = t  # local t and not account for scr refresh
            No1right.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(No1right, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'No1right.started')
            # update status
            No1right.status = STARTED
            No1right.setAutoDraw(True)
        
        # if No1right is active this frame...
        if No1right.status == STARTED:
            # update params
            pass
        
        # if No1right is stopping this frame...
        if No1right.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > No1right.tStartRefresh + 3.0-frameTolerance:
                # keep track of stop time/frame for later
                No1right.tStop = t  # not accounting for scr refresh
                No1right.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'No1right.stopped')
                # update status
                No1right.status = FINISHED
                No1right.setAutoDraw(False)
        
        # *left_1* updates
        
        # if left_1 is starting this frame...
        if left_1.status == NOT_STARTED and tThisFlip >= 5.0-frameTolerance:
            # keep track of start time/frame for later
            left_1.frameNStart = frameN  # exact frame index
            left_1.tStart = t  # local t and not account for scr refresh
            left_1.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(left_1, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'left_1.started')
            # update status
            left_1.status = STARTED
            left_1.setAutoDraw(True)
        
        # if left_1 is active this frame...
        if left_1.status == STARTED:
            # update params
            pass
        
        # if left_1 is stopping this frame...
        if left_1.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > left_1.tStartRefresh + 3.0-frameTolerance:
                # keep track of stop time/frame for later
                left_1.tStop = t  # not accounting for scr refresh
                left_1.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'left_1.stopped')
                # update status
                left_1.status = FINISHED
                left_1.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in L_LComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "L_L" ---
    for thisComponent in L_LComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('L_L.stopped', globalClock.getTime())
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if routineForceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-8.000000)
    
    # --- Prepare to start Routine "left" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('left.started', globalClock.getTime())
    # keep track of which components have finished
    leftComponents = [left1_5, right1_5]
    for thisComponent in leftComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    outlet.push_sample([correct])  # Push event marker.
    print("correct")
    # --- Run Routine "left" ---
    routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 3.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *left1_5* updates
        
        # if left1_5 is starting this frame...
        if left1_5.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            left1_5.frameNStart = frameN  # exact frame index
            left1_5.tStart = t  # local t and not account for scr refresh
            left1_5.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(left1_5, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'left1_5.started')
            # update status
            left1_5.status = STARTED
            left1_5.setAutoDraw(True)
        
        # if left1_5 is active this frame...
        if left1_5.status == STARTED:
            # update params
            pass
        
        # if left1_5 is stopping this frame...
        if left1_5.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > left1_5.tStartRefresh + 3.0-frameTolerance:
                # keep track of stop time/frame for later
                left1_5.tStop = t  # not accounting for scr refresh
                left1_5.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'left1_5.stopped')
                # update status
                left1_5.status = FINISHED
                left1_5.setAutoDraw(False)
        
        # *right1_5* updates
        
        # if right1_5 is starting this frame...
        if right1_5.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            right1_5.frameNStart = frameN  # exact frame index
            right1_5.tStart = t  # local t and not account for scr refresh
            right1_5.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(right1_5, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'right1_5.started')
            # update status
            right1_5.status = STARTED
            right1_5.setAutoDraw(True)
        
        # if right1_5 is active this frame...
        if right1_5.status == STARTED:
            # update params
            pass
        
        # if right1_5 is stopping this frame...
        if right1_5.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > right1_5.tStartRefresh + 3.0-frameTolerance:
                # keep track of stop time/frame for later
                right1_5.tStop = t  # not accounting for scr refresh
                right1_5.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'right1_5.stopped')
                # update status
                right1_5.status = FINISHED
                right1_5.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in leftComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "left" ---
    for thisComponent in leftComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('left.stopped', globalClock.getTime())
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if routineForceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-3.000000)
    
    # --- Prepare to start Routine "R_L" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('R_L.started', globalClock.getTime())
    # keep track of which components have finished
    R_LComponents = [No1left_4, No1right_4, right_2]
    for thisComponent in R_LComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "R_L" ---
    routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 8.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *No1left_4* updates
        
        # if No1left_4 is starting this frame...
        if No1left_4.status == NOT_STARTED and tThisFlip >= 5.0-frameTolerance:
            # keep track of start time/frame for later
            No1left_4.frameNStart = frameN  # exact frame index
            No1left_4.tStart = t  # local t and not account for scr refresh
            No1left_4.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(No1left_4, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'No1left_4.started')
            # update status
            No1left_4.status = STARTED
            No1left_4.setAutoDraw(True)
        
        # if No1left_4 is active this frame...
        if No1left_4.status == STARTED:
            # update params
            pass
        
        # if No1left_4 is stopping this frame...
        if No1left_4.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > No1left_4.tStartRefresh + 3.0-frameTolerance:
                # keep track of stop time/frame for later
                No1left_4.tStop = t  # not accounting for scr refresh
                No1left_4.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'No1left_4.stopped')
                # update status
                No1left_4.status = FINISHED
                No1left_4.setAutoDraw(False)
        
        # *No1right_4* updates
        
        # if No1right_4 is starting this frame...
        if No1right_4.status == NOT_STARTED and tThisFlip >= 5.0-frameTolerance:
            # keep track of start time/frame for later
            No1right_4.frameNStart = frameN  # exact frame index
            No1right_4.tStart = t  # local t and not account for scr refresh
            No1right_4.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(No1right_4, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'No1right_4.started')
            # update status
            No1right_4.status = STARTED
            No1right_4.setAutoDraw(True)
        
        # if No1right_4 is active this frame...
        if No1right_4.status == STARTED:
            # update params
            pass
        
        # if No1right_4 is stopping this frame...
        if No1right_4.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > No1right_4.tStartRefresh + 3.0-frameTolerance:
                # keep track of stop time/frame for later
                No1right_4.tStop = t  # not accounting for scr refresh
                No1right_4.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'No1right_4.stopped')
                # update status
                No1right_4.status = FINISHED
                No1right_4.setAutoDraw(False)
        
        # *right_2* updates
        
        # if right_2 is starting this frame...
        if right_2.status == NOT_STARTED and tThisFlip >= 5.0-frameTolerance:
            # keep track of start time/frame for later
            right_2.frameNStart = frameN  # exact frame index
            right_2.tStart = t  # local t and not account for scr refresh
            right_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(right_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'right_2.started')
            # update status
            right_2.status = STARTED
            right_2.setAutoDraw(True)
        
        # if right_2 is active this frame...
        if right_2.status == STARTED:
            # update params
            pass
        
        # if right_2 is stopping this frame...
        if right_2.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > right_2.tStartRefresh + 3.0-frameTolerance:
                # keep track of stop time/frame for later
                right_2.tStop = t  # not accounting for scr refresh
                right_2.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'right_2.stopped')
                # update status
                right_2.status = FINISHED
                right_2.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in R_LComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "R_L" ---
    for thisComponent in R_LComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('R_L.stopped', globalClock.getTime())
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if routineForceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-8.000000)
    
    # --- Prepare to start Routine "left" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('left.started', globalClock.getTime())
    # keep track of which components have finished
    leftComponents = [left1_5, right1_5]
    for thisComponent in leftComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    outlet.push_sample([error])  # Push event marker.
    print("error")
    # --- Run Routine "left" ---
    routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 3.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *left1_5* updates
        
        # if left1_5 is starting this frame...
        if left1_5.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            left1_5.frameNStart = frameN  # exact frame index
            left1_5.tStart = t  # local t and not account for scr refresh
            left1_5.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(left1_5, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'left1_5.started')
            # update status
            left1_5.status = STARTED
            left1_5.setAutoDraw(True)
        
        # if left1_5 is active this frame...
        if left1_5.status == STARTED:
            # update params
            pass
        
        # if left1_5 is stopping this frame...
        if left1_5.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > left1_5.tStartRefresh + 3.0-frameTolerance:
                # keep track of stop time/frame for later
                left1_5.tStop = t  # not accounting for scr refresh
                left1_5.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'left1_5.stopped')
                # update status
                left1_5.status = FINISHED
                left1_5.setAutoDraw(False)
        
        # *right1_5* updates
        
        # if right1_5 is starting this frame...
        if right1_5.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            right1_5.frameNStart = frameN  # exact frame index
            right1_5.tStart = t  # local t and not account for scr refresh
            right1_5.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(right1_5, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'right1_5.started')
            # update status
            right1_5.status = STARTED
            right1_5.setAutoDraw(True)
        
        # if right1_5 is active this frame...
        if right1_5.status == STARTED:
            # update params
            pass
        
        # if right1_5 is stopping this frame...
        if right1_5.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > right1_5.tStartRefresh + 3.0-frameTolerance:
                # keep track of stop time/frame for later
                right1_5.tStop = t  # not accounting for scr refresh
                right1_5.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'right1_5.stopped')
                # update status
                right1_5.status = FINISHED
                right1_5.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in leftComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "left" ---
    for thisComponent in leftComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('left.stopped', globalClock.getTime())
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if routineForceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-3.000000)
    
    # --- Prepare to start Routine "L_L" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('L_L.started', globalClock.getTime())
    # keep track of which components have finished
    L_LComponents = [No1left, No1right, left_1]
    for thisComponent in L_LComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "L_L" ---
    routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 8.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *No1left* updates
        
        # if No1left is starting this frame...
        if No1left.status == NOT_STARTED and tThisFlip >= 5.0-frameTolerance:
            # keep track of start time/frame for later
            No1left.frameNStart = frameN  # exact frame index
            No1left.tStart = t  # local t and not account for scr refresh
            No1left.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(No1left, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'No1left.started')
            # update status
            No1left.status = STARTED
            No1left.setAutoDraw(True)
        
        # if No1left is active this frame...
        if No1left.status == STARTED:
            # update params
            pass
        
        # if No1left is stopping this frame...
        if No1left.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > No1left.tStartRefresh + 3.0-frameTolerance:
                # keep track of stop time/frame for later
                No1left.tStop = t  # not accounting for scr refresh
                No1left.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'No1left.stopped')
                # update status
                No1left.status = FINISHED
                No1left.setAutoDraw(False)
        
        # *No1right* updates
        
        # if No1right is starting this frame...
        if No1right.status == NOT_STARTED and tThisFlip >= 5.0-frameTolerance:
            # keep track of start time/frame for later
            No1right.frameNStart = frameN  # exact frame index
            No1right.tStart = t  # local t and not account for scr refresh
            No1right.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(No1right, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'No1right.started')
            # update status
            No1right.status = STARTED
            No1right.setAutoDraw(True)
        
        # if No1right is active this frame...
        if No1right.status == STARTED:
            # update params
            pass
        
        # if No1right is stopping this frame...
        if No1right.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > No1right.tStartRefresh + 3.0-frameTolerance:
                # keep track of stop time/frame for later
                No1right.tStop = t  # not accounting for scr refresh
                No1right.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'No1right.stopped')
                # update status
                No1right.status = FINISHED
                No1right.setAutoDraw(False)
        
        # *left_1* updates
        
        # if left_1 is starting this frame...
        if left_1.status == NOT_STARTED and tThisFlip >= 5.0-frameTolerance:
            # keep track of start time/frame for later
            left_1.frameNStart = frameN  # exact frame index
            left_1.tStart = t  # local t and not account for scr refresh
            left_1.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(left_1, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'left_1.started')
            # update status
            left_1.status = STARTED
            left_1.setAutoDraw(True)
        
        # if left_1 is active this frame...
        if left_1.status == STARTED:
            # update params
            pass
        
        # if left_1 is stopping this frame...
        if left_1.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > left_1.tStartRefresh + 3.0-frameTolerance:
                # keep track of stop time/frame for later
                left_1.tStop = t  # not accounting for scr refresh
                left_1.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'left_1.stopped')
                # update status
                left_1.status = FINISHED
                left_1.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in L_LComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "L_L" ---
    for thisComponent in L_LComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('L_L.stopped', globalClock.getTime())
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if routineForceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-8.000000)
    
    # --- Prepare to start Routine "left" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('left.started', globalClock.getTime())
    # keep track of which components have finished
    leftComponents = [left1_5, right1_5]
    for thisComponent in leftComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    outlet.push_sample([correct])  # Push event marker.
    print("correct")
    # --- Run Routine "left" ---
    routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 3.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *left1_5* updates
        
        # if left1_5 is starting this frame...
        if left1_5.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            left1_5.frameNStart = frameN  # exact frame index
            left1_5.tStart = t  # local t and not account for scr refresh
            left1_5.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(left1_5, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'left1_5.started')
            # update status
            left1_5.status = STARTED
            left1_5.setAutoDraw(True)
        
        # if left1_5 is active this frame...
        if left1_5.status == STARTED:
            # update params
            pass
        
        # if left1_5 is stopping this frame...
        if left1_5.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > left1_5.tStartRefresh + 3.0-frameTolerance:
                # keep track of stop time/frame for later
                left1_5.tStop = t  # not accounting for scr refresh
                left1_5.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'left1_5.stopped')
                # update status
                left1_5.status = FINISHED
                left1_5.setAutoDraw(False)
        
        # *right1_5* updates
        
        # if right1_5 is starting this frame...
        if right1_5.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            right1_5.frameNStart = frameN  # exact frame index
            right1_5.tStart = t  # local t and not account for scr refresh
            right1_5.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(right1_5, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'right1_5.started')
            # update status
            right1_5.status = STARTED
            right1_5.setAutoDraw(True)
        
        # if right1_5 is active this frame...
        if right1_5.status == STARTED:
            # update params
            pass
        
        # if right1_5 is stopping this frame...
        if right1_5.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > right1_5.tStartRefresh + 3.0-frameTolerance:
                # keep track of stop time/frame for later
                right1_5.tStop = t  # not accounting for scr refresh
                right1_5.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'right1_5.stopped')
                # update status
                right1_5.status = FINISHED
                right1_5.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in leftComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "left" ---
    for thisComponent in leftComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('left.stopped', globalClock.getTime())
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if routineForceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-3.000000)
    
    # --- Prepare to start Routine "R_L" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('R_L.started', globalClock.getTime())
    # keep track of which components have finished
    R_LComponents = [No1left_4, No1right_4, right_2]
    for thisComponent in R_LComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "R_L" ---
    routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 8.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *No1left_4* updates
        
        # if No1left_4 is starting this frame...
        if No1left_4.status == NOT_STARTED and tThisFlip >= 5.0-frameTolerance:
            # keep track of start time/frame for later
            No1left_4.frameNStart = frameN  # exact frame index
            No1left_4.tStart = t  # local t and not account for scr refresh
            No1left_4.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(No1left_4, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'No1left_4.started')
            # update status
            No1left_4.status = STARTED
            No1left_4.setAutoDraw(True)
        
        # if No1left_4 is active this frame...
        if No1left_4.status == STARTED:
            # update params
            pass
        
        # if No1left_4 is stopping this frame...
        if No1left_4.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > No1left_4.tStartRefresh + 3.0-frameTolerance:
                # keep track of stop time/frame for later
                No1left_4.tStop = t  # not accounting for scr refresh
                No1left_4.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'No1left_4.stopped')
                # update status
                No1left_4.status = FINISHED
                No1left_4.setAutoDraw(False)
        
        # *No1right_4* updates
        
        # if No1right_4 is starting this frame...
        if No1right_4.status == NOT_STARTED and tThisFlip >= 5.0-frameTolerance:
            # keep track of start time/frame for later
            No1right_4.frameNStart = frameN  # exact frame index
            No1right_4.tStart = t  # local t and not account for scr refresh
            No1right_4.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(No1right_4, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'No1right_4.started')
            # update status
            No1right_4.status = STARTED
            No1right_4.setAutoDraw(True)
        
        # if No1right_4 is active this frame...
        if No1right_4.status == STARTED:
            # update params
            pass
        
        # if No1right_4 is stopping this frame...
        if No1right_4.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > No1right_4.tStartRefresh + 3.0-frameTolerance:
                # keep track of stop time/frame for later
                No1right_4.tStop = t  # not accounting for scr refresh
                No1right_4.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'No1right_4.stopped')
                # update status
                No1right_4.status = FINISHED
                No1right_4.setAutoDraw(False)
        
        # *right_2* updates
        
        # if right_2 is starting this frame...
        if right_2.status == NOT_STARTED and tThisFlip >= 5.0-frameTolerance:
            # keep track of start time/frame for later
            right_2.frameNStart = frameN  # exact frame index
            right_2.tStart = t  # local t and not account for scr refresh
            right_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(right_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'right_2.started')
            # update status
            right_2.status = STARTED
            right_2.setAutoDraw(True)
        
        # if right_2 is active this frame...
        if right_2.status == STARTED:
            # update params
            pass
        
        # if right_2 is stopping this frame...
        if right_2.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > right_2.tStartRefresh + 3.0-frameTolerance:
                # keep track of stop time/frame for later
                right_2.tStop = t  # not accounting for scr refresh
                right_2.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'right_2.stopped')
                # update status
                right_2.status = FINISHED
                right_2.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in R_LComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "R_L" ---
    for thisComponent in R_LComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('R_L.stopped', globalClock.getTime())
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if routineForceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-8.000000)
    
    # --- Prepare to start Routine "left" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('left.started', globalClock.getTime())
    # keep track of which components have finished
    leftComponents = [left1_5, right1_5]
    for thisComponent in leftComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    outlet.push_sample([error])  # Push event marker.
    print("error")
    # --- Run Routine "left" ---
    routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 3.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *left1_5* updates
        
        # if left1_5 is starting this frame...
        if left1_5.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            left1_5.frameNStart = frameN  # exact frame index
            left1_5.tStart = t  # local t and not account for scr refresh
            left1_5.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(left1_5, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'left1_5.started')
            # update status
            left1_5.status = STARTED
            left1_5.setAutoDraw(True)
        
        # if left1_5 is active this frame...
        if left1_5.status == STARTED:
            # update params
            pass
        
        # if left1_5 is stopping this frame...
        if left1_5.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > left1_5.tStartRefresh + 3.0-frameTolerance:
                # keep track of stop time/frame for later
                left1_5.tStop = t  # not accounting for scr refresh
                left1_5.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'left1_5.stopped')
                # update status
                left1_5.status = FINISHED
                left1_5.setAutoDraw(False)
        
        # *right1_5* updates
        
        # if right1_5 is starting this frame...
        if right1_5.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            right1_5.frameNStart = frameN  # exact frame index
            right1_5.tStart = t  # local t and not account for scr refresh
            right1_5.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(right1_5, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'right1_5.started')
            # update status
            right1_5.status = STARTED
            right1_5.setAutoDraw(True)
        
        # if right1_5 is active this frame...
        if right1_5.status == STARTED:
            # update params
            pass
        
        # if right1_5 is stopping this frame...
        if right1_5.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > right1_5.tStartRefresh + 3.0-frameTolerance:
                # keep track of stop time/frame for later
                right1_5.tStop = t  # not accounting for scr refresh
                right1_5.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'right1_5.stopped')
                # update status
                right1_5.status = FINISHED
                right1_5.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in leftComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "left" ---
    for thisComponent in leftComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('left.stopped', globalClock.getTime())
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if routineForceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-3.000000)
    
    # --- Prepare to start Routine "L_R" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('L_R.started', globalClock.getTime())
    # keep track of which components have finished
    L_RComponents = [No1left_3, No1right_3, left_2]
    for thisComponent in L_RComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "L_R" ---
    routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 8.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *No1left_3* updates
        
        # if No1left_3 is starting this frame...
        if No1left_3.status == NOT_STARTED and tThisFlip >= 5.0-frameTolerance:
            # keep track of start time/frame for later
            No1left_3.frameNStart = frameN  # exact frame index
            No1left_3.tStart = t  # local t and not account for scr refresh
            No1left_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(No1left_3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'No1left_3.started')
            # update status
            No1left_3.status = STARTED
            No1left_3.setAutoDraw(True)
        
        # if No1left_3 is active this frame...
        if No1left_3.status == STARTED:
            # update params
            pass
        
        # if No1left_3 is stopping this frame...
        if No1left_3.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > No1left_3.tStartRefresh + 3.0-frameTolerance:
                # keep track of stop time/frame for later
                No1left_3.tStop = t  # not accounting for scr refresh
                No1left_3.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'No1left_3.stopped')
                # update status
                No1left_3.status = FINISHED
                No1left_3.setAutoDraw(False)
        
        # *No1right_3* updates
        
        # if No1right_3 is starting this frame...
        if No1right_3.status == NOT_STARTED and tThisFlip >= 5.0-frameTolerance:
            # keep track of start time/frame for later
            No1right_3.frameNStart = frameN  # exact frame index
            No1right_3.tStart = t  # local t and not account for scr refresh
            No1right_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(No1right_3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'No1right_3.started')
            # update status
            No1right_3.status = STARTED
            No1right_3.setAutoDraw(True)
        
        # if No1right_3 is active this frame...
        if No1right_3.status == STARTED:
            # update params
            pass
        
        # if No1right_3 is stopping this frame...
        if No1right_3.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > No1right_3.tStartRefresh + 3.0-frameTolerance:
                # keep track of stop time/frame for later
                No1right_3.tStop = t  # not accounting for scr refresh
                No1right_3.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'No1right_3.stopped')
                # update status
                No1right_3.status = FINISHED
                No1right_3.setAutoDraw(False)
        
        # *left_2* updates
        
        # if left_2 is starting this frame...
        if left_2.status == NOT_STARTED and tThisFlip >= 5.0-frameTolerance:
            # keep track of start time/frame for later
            left_2.frameNStart = frameN  # exact frame index
            left_2.tStart = t  # local t and not account for scr refresh
            left_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(left_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'left_2.started')
            # update status
            left_2.status = STARTED
            left_2.setAutoDraw(True)
        
        # if left_2 is active this frame...
        if left_2.status == STARTED:
            # update params
            pass
        
        # if left_2 is stopping this frame...
        if left_2.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > left_2.tStartRefresh + 3.0-frameTolerance:
                # keep track of stop time/frame for later
                left_2.tStop = t  # not accounting for scr refresh
                left_2.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'left_2.stopped')
                # update status
                left_2.status = FINISHED
                left_2.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in L_RComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "L_R" ---
    for thisComponent in L_RComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('L_R.stopped', globalClock.getTime())
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if routineForceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-8.000000)
    
    # --- Prepare to start Routine "Right" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('Right.started', globalClock.getTime())
    # keep track of which components have finished
    RightComponents = [left1_6, right1_6]
    for thisComponent in RightComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    outlet.push_sample([error])  # Push event marker.
    print("error")
    # --- Run Routine "Right" ---
    routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 3.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *left1_6* updates
        
        # if left1_6 is starting this frame...
        if left1_6.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            left1_6.frameNStart = frameN  # exact frame index
            left1_6.tStart = t  # local t and not account for scr refresh
            left1_6.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(left1_6, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'left1_6.started')
            # update status
            left1_6.status = STARTED
            left1_6.setAutoDraw(True)
        
        # if left1_6 is active this frame...
        if left1_6.status == STARTED:
            # update params
            pass
        
        # if left1_6 is stopping this frame...
        if left1_6.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > left1_6.tStartRefresh + 3.0-frameTolerance:
                # keep track of stop time/frame for later
                left1_6.tStop = t  # not accounting for scr refresh
                left1_6.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'left1_6.stopped')
                # update status
                left1_6.status = FINISHED
                left1_6.setAutoDraw(False)
        
        # *right1_6* updates
        
        # if right1_6 is starting this frame...
        if right1_6.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            right1_6.frameNStart = frameN  # exact frame index
            right1_6.tStart = t  # local t and not account for scr refresh
            right1_6.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(right1_6, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'right1_6.started')
            # update status
            right1_6.status = STARTED
            right1_6.setAutoDraw(True)
        
        # if right1_6 is active this frame...
        if right1_6.status == STARTED:
            # update params
            pass
        
        # if right1_6 is stopping this frame...
        if right1_6.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > right1_6.tStartRefresh + 3.0-frameTolerance:
                # keep track of stop time/frame for later
                right1_6.tStop = t  # not accounting for scr refresh
                right1_6.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'right1_6.stopped')
                # update status
                right1_6.status = FINISHED
                right1_6.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in RightComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Right" ---
    for thisComponent in RightComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('Right.stopped', globalClock.getTime())
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if routineForceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-3.000000)
    
    # --- Prepare to start Routine "R_R" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('R_R.started', globalClock.getTime())
    # keep track of which components have finished
    R_RComponents = [No1left_2, No1right_2, right]
    for thisComponent in R_RComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "R_R" ---
    routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 8.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *No1left_2* updates
        
        # if No1left_2 is starting this frame...
        if No1left_2.status == NOT_STARTED and tThisFlip >= 5.0-frameTolerance:
            # keep track of start time/frame for later
            No1left_2.frameNStart = frameN  # exact frame index
            No1left_2.tStart = t  # local t and not account for scr refresh
            No1left_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(No1left_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'No1left_2.started')
            # update status
            No1left_2.status = STARTED
            No1left_2.setAutoDraw(True)
        
        # if No1left_2 is active this frame...
        if No1left_2.status == STARTED:
            # update params
            pass
        
        # if No1left_2 is stopping this frame...
        if No1left_2.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > No1left_2.tStartRefresh + 3.0-frameTolerance:
                # keep track of stop time/frame for later
                No1left_2.tStop = t  # not accounting for scr refresh
                No1left_2.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'No1left_2.stopped')
                # update status
                No1left_2.status = FINISHED
                No1left_2.setAutoDraw(False)
        
        # *No1right_2* updates
        
        # if No1right_2 is starting this frame...
        if No1right_2.status == NOT_STARTED and tThisFlip >= 5.0-frameTolerance:
            # keep track of start time/frame for later
            No1right_2.frameNStart = frameN  # exact frame index
            No1right_2.tStart = t  # local t and not account for scr refresh
            No1right_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(No1right_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'No1right_2.started')
            # update status
            No1right_2.status = STARTED
            No1right_2.setAutoDraw(True)
        
        # if No1right_2 is active this frame...
        if No1right_2.status == STARTED:
            # update params
            pass
        
        # if No1right_2 is stopping this frame...
        if No1right_2.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > No1right_2.tStartRefresh + 3.0-frameTolerance:
                # keep track of stop time/frame for later
                No1right_2.tStop = t  # not accounting for scr refresh
                No1right_2.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'No1right_2.stopped')
                # update status
                No1right_2.status = FINISHED
                No1right_2.setAutoDraw(False)
        
        # *right* updates
        
        # if right is starting this frame...
        if right.status == NOT_STARTED and tThisFlip >= 5.0-frameTolerance:
            # keep track of start time/frame for later
            right.frameNStart = frameN  # exact frame index
            right.tStart = t  # local t and not account for scr refresh
            right.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(right, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'right.started')
            # update status
            right.status = STARTED
            right.setAutoDraw(True)
        
        # if right is active this frame...
        if right.status == STARTED:
            # update params
            pass
        
        # if right is stopping this frame...
        if right.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > right.tStartRefresh + 3.0-frameTolerance:
                # keep track of stop time/frame for later
                right.tStop = t  # not accounting for scr refresh
                right.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'right.stopped')
                # update status
                right.status = FINISHED
                right.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in R_RComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "R_R" ---
    for thisComponent in R_RComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('R_R.stopped', globalClock.getTime())
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if routineForceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-8.000000)
    
    # --- Prepare to start Routine "Right" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('Right.started', globalClock.getTime())
    # keep track of which components have finished
    RightComponents = [left1_6, right1_6]
    for thisComponent in RightComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    outlet.push_sample([correct])  # Push event marker.
    print("correct")
    # --- Run Routine "Right" ---
    routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 3.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *left1_6* updates
        
        # if left1_6 is starting this frame...
        if left1_6.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            left1_6.frameNStart = frameN  # exact frame index
            left1_6.tStart = t  # local t and not account for scr refresh
            left1_6.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(left1_6, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'left1_6.started')
            # update status
            left1_6.status = STARTED
            left1_6.setAutoDraw(True)
        
        # if left1_6 is active this frame...
        if left1_6.status == STARTED:
            # update params
            pass
        
        # if left1_6 is stopping this frame...
        if left1_6.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > left1_6.tStartRefresh + 3.0-frameTolerance:
                # keep track of stop time/frame for later
                left1_6.tStop = t  # not accounting for scr refresh
                left1_6.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'left1_6.stopped')
                # update status
                left1_6.status = FINISHED
                left1_6.setAutoDraw(False)
        
        # *right1_6* updates
        
        # if right1_6 is starting this frame...
        if right1_6.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            right1_6.frameNStart = frameN  # exact frame index
            right1_6.tStart = t  # local t and not account for scr refresh
            right1_6.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(right1_6, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'right1_6.started')
            # update status
            right1_6.status = STARTED
            right1_6.setAutoDraw(True)
        
        # if right1_6 is active this frame...
        if right1_6.status == STARTED:
            # update params
            pass
        
        # if right1_6 is stopping this frame...
        if right1_6.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > right1_6.tStartRefresh + 3.0-frameTolerance:
                # keep track of stop time/frame for later
                right1_6.tStop = t  # not accounting for scr refresh
                right1_6.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'right1_6.stopped')
                # update status
                right1_6.status = FINISHED
                right1_6.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in RightComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Right" ---
    for thisComponent in RightComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('Right.stopped', globalClock.getTime())
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if routineForceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-3.000000)
    
    # --- Prepare to start Routine "L_R" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('L_R.started', globalClock.getTime())
    # keep track of which components have finished
    L_RComponents = [No1left_3, No1right_3, left_2]
    for thisComponent in L_RComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "L_R" ---
    routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 8.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *No1left_3* updates
        
        # if No1left_3 is starting this frame...
        if No1left_3.status == NOT_STARTED and tThisFlip >= 5.0-frameTolerance:
            # keep track of start time/frame for later
            No1left_3.frameNStart = frameN  # exact frame index
            No1left_3.tStart = t  # local t and not account for scr refresh
            No1left_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(No1left_3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'No1left_3.started')
            # update status
            No1left_3.status = STARTED
            No1left_3.setAutoDraw(True)
        
        # if No1left_3 is active this frame...
        if No1left_3.status == STARTED:
            # update params
            pass
        
        # if No1left_3 is stopping this frame...
        if No1left_3.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > No1left_3.tStartRefresh + 3.0-frameTolerance:
                # keep track of stop time/frame for later
                No1left_3.tStop = t  # not accounting for scr refresh
                No1left_3.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'No1left_3.stopped')
                # update status
                No1left_3.status = FINISHED
                No1left_3.setAutoDraw(False)
        
        # *No1right_3* updates
        
        # if No1right_3 is starting this frame...
        if No1right_3.status == NOT_STARTED and tThisFlip >= 5.0-frameTolerance:
            # keep track of start time/frame for later
            No1right_3.frameNStart = frameN  # exact frame index
            No1right_3.tStart = t  # local t and not account for scr refresh
            No1right_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(No1right_3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'No1right_3.started')
            # update status
            No1right_3.status = STARTED
            No1right_3.setAutoDraw(True)
        
        # if No1right_3 is active this frame...
        if No1right_3.status == STARTED:
            # update params
            pass
        
        # if No1right_3 is stopping this frame...
        if No1right_3.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > No1right_3.tStartRefresh + 3.0-frameTolerance:
                # keep track of stop time/frame for later
                No1right_3.tStop = t  # not accounting for scr refresh
                No1right_3.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'No1right_3.stopped')
                # update status
                No1right_3.status = FINISHED
                No1right_3.setAutoDraw(False)
        
        # *left_2* updates
        
        # if left_2 is starting this frame...
        if left_2.status == NOT_STARTED and tThisFlip >= 5.0-frameTolerance:
            # keep track of start time/frame for later
            left_2.frameNStart = frameN  # exact frame index
            left_2.tStart = t  # local t and not account for scr refresh
            left_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(left_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'left_2.started')
            # update status
            left_2.status = STARTED
            left_2.setAutoDraw(True)
        
        # if left_2 is active this frame...
        if left_2.status == STARTED:
            # update params
            pass
        
        # if left_2 is stopping this frame...
        if left_2.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > left_2.tStartRefresh + 3.0-frameTolerance:
                # keep track of stop time/frame for later
                left_2.tStop = t  # not accounting for scr refresh
                left_2.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'left_2.stopped')
                # update status
                left_2.status = FINISHED
                left_2.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in L_RComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "L_R" ---
    for thisComponent in L_RComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('L_R.stopped', globalClock.getTime())
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if routineForceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-8.000000)
    
    # --- Prepare to start Routine "Right" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('Right.started', globalClock.getTime())
    # keep track of which components have finished
    RightComponents = [left1_6, right1_6]
    for thisComponent in RightComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    outlet.push_sample([error])  # Push event marker.
    print("error")
    # --- Run Routine "Right" ---
    routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 3.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *left1_6* updates
        
        # if left1_6 is starting this frame...
        if left1_6.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            left1_6.frameNStart = frameN  # exact frame index
            left1_6.tStart = t  # local t and not account for scr refresh
            left1_6.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(left1_6, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'left1_6.started')
            # update status
            left1_6.status = STARTED
            left1_6.setAutoDraw(True)
        
        # if left1_6 is active this frame...
        if left1_6.status == STARTED:
            # update params
            pass
        
        # if left1_6 is stopping this frame...
        if left1_6.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > left1_6.tStartRefresh + 3.0-frameTolerance:
                # keep track of stop time/frame for later
                left1_6.tStop = t  # not accounting for scr refresh
                left1_6.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'left1_6.stopped')
                # update status
                left1_6.status = FINISHED
                left1_6.setAutoDraw(False)
        
        # *right1_6* updates
        
        # if right1_6 is starting this frame...
        if right1_6.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            right1_6.frameNStart = frameN  # exact frame index
            right1_6.tStart = t  # local t and not account for scr refresh
            right1_6.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(right1_6, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'right1_6.started')
            # update status
            right1_6.status = STARTED
            right1_6.setAutoDraw(True)
        
        # if right1_6 is active this frame...
        if right1_6.status == STARTED:
            # update params
            pass
        
        # if right1_6 is stopping this frame...
        if right1_6.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > right1_6.tStartRefresh + 3.0-frameTolerance:
                # keep track of stop time/frame for later
                right1_6.tStop = t  # not accounting for scr refresh
                right1_6.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'right1_6.stopped')
                # update status
                right1_6.status = FINISHED
                right1_6.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in RightComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Right" ---
    for thisComponent in RightComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('Right.stopped', globalClock.getTime())
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if routineForceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-3.000000)
    
    # --- Prepare to start Routine "R_R" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('R_R.started', globalClock.getTime())
    # keep track of which components have finished
    R_RComponents = [No1left_2, No1right_2, right]
    for thisComponent in R_RComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "R_R" ---
    routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 8.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *No1left_2* updates
        
        # if No1left_2 is starting this frame...
        if No1left_2.status == NOT_STARTED and tThisFlip >= 5.0-frameTolerance:
            # keep track of start time/frame for later
            No1left_2.frameNStart = frameN  # exact frame index
            No1left_2.tStart = t  # local t and not account for scr refresh
            No1left_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(No1left_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'No1left_2.started')
            # update status
            No1left_2.status = STARTED
            No1left_2.setAutoDraw(True)
        
        # if No1left_2 is active this frame...
        if No1left_2.status == STARTED:
            # update params
            pass
        
        # if No1left_2 is stopping this frame...
        if No1left_2.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > No1left_2.tStartRefresh + 3.0-frameTolerance:
                # keep track of stop time/frame for later
                No1left_2.tStop = t  # not accounting for scr refresh
                No1left_2.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'No1left_2.stopped')
                # update status
                No1left_2.status = FINISHED
                No1left_2.setAutoDraw(False)
        
        # *No1right_2* updates
        
        # if No1right_2 is starting this frame...
        if No1right_2.status == NOT_STARTED and tThisFlip >= 5.0-frameTolerance:
            # keep track of start time/frame for later
            No1right_2.frameNStart = frameN  # exact frame index
            No1right_2.tStart = t  # local t and not account for scr refresh
            No1right_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(No1right_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'No1right_2.started')
            # update status
            No1right_2.status = STARTED
            No1right_2.setAutoDraw(True)
        
        # if No1right_2 is active this frame...
        if No1right_2.status == STARTED:
            # update params
            pass
        
        # if No1right_2 is stopping this frame...
        if No1right_2.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > No1right_2.tStartRefresh + 3.0-frameTolerance:
                # keep track of stop time/frame for later
                No1right_2.tStop = t  # not accounting for scr refresh
                No1right_2.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'No1right_2.stopped')
                # update status
                No1right_2.status = FINISHED
                No1right_2.setAutoDraw(False)
        
        # *right* updates
        
        # if right is starting this frame...
        if right.status == NOT_STARTED and tThisFlip >= 5.0-frameTolerance:
            # keep track of start time/frame for later
            right.frameNStart = frameN  # exact frame index
            right.tStart = t  # local t and not account for scr refresh
            right.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(right, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'right.started')
            # update status
            right.status = STARTED
            right.setAutoDraw(True)
        
        # if right is active this frame...
        if right.status == STARTED:
            # update params
            pass
        
        # if right is stopping this frame...
        if right.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > right.tStartRefresh + 3.0-frameTolerance:
                # keep track of stop time/frame for later
                right.tStop = t  # not accounting for scr refresh
                right.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'right.stopped')
                # update status
                right.status = FINISHED
                right.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in R_RComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "R_R" ---
    for thisComponent in R_RComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('R_R.stopped', globalClock.getTime())
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if routineForceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-8.000000)
    
    # --- Prepare to start Routine "Right" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('Right.started', globalClock.getTime())
    # keep track of which components have finished
    RightComponents = [left1_6, right1_6]
    for thisComponent in RightComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    outlet.push_sample([correct])  # Push event marker.
    print("correct")
    # --- Run Routine "Right" ---
    routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 3.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *left1_6* updates
        
        # if left1_6 is starting this frame...
        if left1_6.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            left1_6.frameNStart = frameN  # exact frame index
            left1_6.tStart = t  # local t and not account for scr refresh
            left1_6.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(left1_6, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'left1_6.started')
            # update status
            left1_6.status = STARTED
            left1_6.setAutoDraw(True)
        
        # if left1_6 is active this frame...
        if left1_6.status == STARTED:
            # update params
            pass
        
        # if left1_6 is stopping this frame...
        if left1_6.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > left1_6.tStartRefresh + 3.0-frameTolerance:
                # keep track of stop time/frame for later
                left1_6.tStop = t  # not accounting for scr refresh
                left1_6.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'left1_6.stopped')
                # update status
                left1_6.status = FINISHED
                left1_6.setAutoDraw(False)
        
        # *right1_6* updates
        
        # if right1_6 is starting this frame...
        if right1_6.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            right1_6.frameNStart = frameN  # exact frame index
            right1_6.tStart = t  # local t and not account for scr refresh
            right1_6.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(right1_6, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'right1_6.started')
            # update status
            right1_6.status = STARTED
            right1_6.setAutoDraw(True)
        
        # if right1_6 is active this frame...
        if right1_6.status == STARTED:
            # update params
            pass
        
        # if right1_6 is stopping this frame...
        if right1_6.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > right1_6.tStartRefresh + 3.0-frameTolerance:
                # keep track of stop time/frame for later
                right1_6.tStop = t  # not accounting for scr refresh
                right1_6.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'right1_6.stopped')
                # update status
                right1_6.status = FINISHED
                right1_6.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in RightComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Right" ---
    for thisComponent in RightComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('Right.stopped', globalClock.getTime())
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if routineForceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-3.000000)
    
    # --- Prepare to start Routine "L_L" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('L_L.started', globalClock.getTime())
    # keep track of which components have finished
    L_LComponents = [No1left, No1right, left_1]
    for thisComponent in L_LComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "L_L" ---
    routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 8.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *No1left* updates
        
        # if No1left is starting this frame...
        if No1left.status == NOT_STARTED and tThisFlip >= 5.0-frameTolerance:
            # keep track of start time/frame for later
            No1left.frameNStart = frameN  # exact frame index
            No1left.tStart = t  # local t and not account for scr refresh
            No1left.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(No1left, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'No1left.started')
            # update status
            No1left.status = STARTED
            No1left.setAutoDraw(True)
        
        # if No1left is active this frame...
        if No1left.status == STARTED:
            # update params
            pass
        
        # if No1left is stopping this frame...
        if No1left.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > No1left.tStartRefresh + 3.0-frameTolerance:
                # keep track of stop time/frame for later
                No1left.tStop = t  # not accounting for scr refresh
                No1left.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'No1left.stopped')
                # update status
                No1left.status = FINISHED
                No1left.setAutoDraw(False)
        
        # *No1right* updates
        
        # if No1right is starting this frame...
        if No1right.status == NOT_STARTED and tThisFlip >= 5.0-frameTolerance:
            # keep track of start time/frame for later
            No1right.frameNStart = frameN  # exact frame index
            No1right.tStart = t  # local t and not account for scr refresh
            No1right.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(No1right, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'No1right.started')
            # update status
            No1right.status = STARTED
            No1right.setAutoDraw(True)
        
        # if No1right is active this frame...
        if No1right.status == STARTED:
            # update params
            pass
        
        # if No1right is stopping this frame...
        if No1right.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > No1right.tStartRefresh + 3.0-frameTolerance:
                # keep track of stop time/frame for later
                No1right.tStop = t  # not accounting for scr refresh
                No1right.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'No1right.stopped')
                # update status
                No1right.status = FINISHED
                No1right.setAutoDraw(False)
        
        # *left_1* updates
        
        # if left_1 is starting this frame...
        if left_1.status == NOT_STARTED and tThisFlip >= 5.0-frameTolerance:
            # keep track of start time/frame for later
            left_1.frameNStart = frameN  # exact frame index
            left_1.tStart = t  # local t and not account for scr refresh
            left_1.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(left_1, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'left_1.started')
            # update status
            left_1.status = STARTED
            left_1.setAutoDraw(True)
        
        # if left_1 is active this frame...
        if left_1.status == STARTED:
            # update params
            pass
        
        # if left_1 is stopping this frame...
        if left_1.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > left_1.tStartRefresh + 3.0-frameTolerance:
                # keep track of stop time/frame for later
                left_1.tStop = t  # not accounting for scr refresh
                left_1.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'left_1.stopped')
                # update status
                left_1.status = FINISHED
                left_1.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in L_LComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "L_L" ---
    for thisComponent in L_LComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('L_L.stopped', globalClock.getTime())
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if routineForceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-8.000000)
    
    # --- Prepare to start Routine "left" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('left.started', globalClock.getTime())
    # keep track of which components have finished
    leftComponents = [left1_5, right1_5]
    for thisComponent in leftComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    outlet.push_sample([correct])  # Push event marker.
    print("correct")
    # --- Run Routine "left" ---
    routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 3.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *left1_5* updates
        
        # if left1_5 is starting this frame...
        if left1_5.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            left1_5.frameNStart = frameN  # exact frame index
            left1_5.tStart = t  # local t and not account for scr refresh
            left1_5.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(left1_5, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'left1_5.started')
            # update status
            left1_5.status = STARTED
            left1_5.setAutoDraw(True)
        
        # if left1_5 is active this frame...
        if left1_5.status == STARTED:
            # update params
            pass
        
        # if left1_5 is stopping this frame...
        if left1_5.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > left1_5.tStartRefresh + 3.0-frameTolerance:
                # keep track of stop time/frame for later
                left1_5.tStop = t  # not accounting for scr refresh
                left1_5.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'left1_5.stopped')
                # update status
                left1_5.status = FINISHED
                left1_5.setAutoDraw(False)
        
        # *right1_5* updates
        
        # if right1_5 is starting this frame...
        if right1_5.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            right1_5.frameNStart = frameN  # exact frame index
            right1_5.tStart = t  # local t and not account for scr refresh
            right1_5.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(right1_5, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'right1_5.started')
            # update status
            right1_5.status = STARTED
            right1_5.setAutoDraw(True)
        
        # if right1_5 is active this frame...
        if right1_5.status == STARTED:
            # update params
            pass
        
        # if right1_5 is stopping this frame...
        if right1_5.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > right1_5.tStartRefresh + 3.0-frameTolerance:
                # keep track of stop time/frame for later
                right1_5.tStop = t  # not accounting for scr refresh
                right1_5.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'right1_5.stopped')
                # update status
                right1_5.status = FINISHED
                right1_5.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in leftComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "left" ---
    for thisComponent in leftComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('left.stopped', globalClock.getTime())
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if routineForceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-3.000000)
    
    # --- Prepare to start Routine "R_R" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('R_R.started', globalClock.getTime())
    # keep track of which components have finished
    R_RComponents = [No1left_2, No1right_2, right]
    for thisComponent in R_RComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "R_R" ---
    routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 8.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *No1left_2* updates
        
        # if No1left_2 is starting this frame...
        if No1left_2.status == NOT_STARTED and tThisFlip >= 5.0-frameTolerance:
            # keep track of start time/frame for later
            No1left_2.frameNStart = frameN  # exact frame index
            No1left_2.tStart = t  # local t and not account for scr refresh
            No1left_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(No1left_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'No1left_2.started')
            # update status
            No1left_2.status = STARTED
            No1left_2.setAutoDraw(True)
        
        # if No1left_2 is active this frame...
        if No1left_2.status == STARTED:
            # update params
            pass
        
        # if No1left_2 is stopping this frame...
        if No1left_2.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > No1left_2.tStartRefresh + 3.0-frameTolerance:
                # keep track of stop time/frame for later
                No1left_2.tStop = t  # not accounting for scr refresh
                No1left_2.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'No1left_2.stopped')
                # update status
                No1left_2.status = FINISHED
                No1left_2.setAutoDraw(False)
        
        # *No1right_2* updates
        
        # if No1right_2 is starting this frame...
        if No1right_2.status == NOT_STARTED and tThisFlip >= 5.0-frameTolerance:
            # keep track of start time/frame for later
            No1right_2.frameNStart = frameN  # exact frame index
            No1right_2.tStart = t  # local t and not account for scr refresh
            No1right_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(No1right_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'No1right_2.started')
            # update status
            No1right_2.status = STARTED
            No1right_2.setAutoDraw(True)
        
        # if No1right_2 is active this frame...
        if No1right_2.status == STARTED:
            # update params
            pass
        
        # if No1right_2 is stopping this frame...
        if No1right_2.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > No1right_2.tStartRefresh + 3.0-frameTolerance:
                # keep track of stop time/frame for later
                No1right_2.tStop = t  # not accounting for scr refresh
                No1right_2.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'No1right_2.stopped')
                # update status
                No1right_2.status = FINISHED
                No1right_2.setAutoDraw(False)
        
        # *right* updates
        
        # if right is starting this frame...
        if right.status == NOT_STARTED and tThisFlip >= 5.0-frameTolerance:
            # keep track of start time/frame for later
            right.frameNStart = frameN  # exact frame index
            right.tStart = t  # local t and not account for scr refresh
            right.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(right, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'right.started')
            # update status
            right.status = STARTED
            right.setAutoDraw(True)
        
        # if right is active this frame...
        if right.status == STARTED:
            # update params
            pass
        
        # if right is stopping this frame...
        if right.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > right.tStartRefresh + 3.0-frameTolerance:
                # keep track of stop time/frame for later
                right.tStop = t  # not accounting for scr refresh
                right.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'right.stopped')
                # update status
                right.status = FINISHED
                right.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in R_RComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "R_R" ---
    for thisComponent in R_RComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('R_R.stopped', globalClock.getTime())
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if routineForceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-8.000000)
    
    # --- Prepare to start Routine "Right" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('Right.started', globalClock.getTime())
    # keep track of which components have finished
    RightComponents = [left1_6, right1_6]
    for thisComponent in RightComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    outlet.push_sample([correct])  # Push event marker.
    print("correct")
    # --- Run Routine "Right" ---
    routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 3.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *left1_6* updates
        
        # if left1_6 is starting this frame...
        if left1_6.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            left1_6.frameNStart = frameN  # exact frame index
            left1_6.tStart = t  # local t and not account for scr refresh
            left1_6.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(left1_6, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'left1_6.started')
            # update status
            left1_6.status = STARTED
            left1_6.setAutoDraw(True)
        
        # if left1_6 is active this frame...
        if left1_6.status == STARTED:
            # update params
            pass
        
        # if left1_6 is stopping this frame...
        if left1_6.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > left1_6.tStartRefresh + 3.0-frameTolerance:
                # keep track of stop time/frame for later
                left1_6.tStop = t  # not accounting for scr refresh
                left1_6.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'left1_6.stopped')
                # update status
                left1_6.status = FINISHED
                left1_6.setAutoDraw(False)
        
        # *right1_6* updates
        
        # if right1_6 is starting this frame...
        if right1_6.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            right1_6.frameNStart = frameN  # exact frame index
            right1_6.tStart = t  # local t and not account for scr refresh
            right1_6.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(right1_6, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'right1_6.started')
            # update status
            right1_6.status = STARTED
            right1_6.setAutoDraw(True)
        
        # if right1_6 is active this frame...
        if right1_6.status == STARTED:
            # update params
            pass
        
        # if right1_6 is stopping this frame...
        if right1_6.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > right1_6.tStartRefresh + 3.0-frameTolerance:
                # keep track of stop time/frame for later
                right1_6.tStop = t  # not accounting for scr refresh
                right1_6.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'right1_6.stopped')
                # update status
                right1_6.status = FINISHED
                right1_6.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in RightComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Right" ---
    for thisComponent in RightComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('Right.stopped', globalClock.getTime())
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if routineForceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-3.000000)
    
    # --- Prepare to start Routine "L_R" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('L_R.started', globalClock.getTime())
    # keep track of which components have finished
    L_RComponents = [No1left_3, No1right_3, left_2]
    for thisComponent in L_RComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "L_R" ---
    routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 8.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *No1left_3* updates
        
        # if No1left_3 is starting this frame...
        if No1left_3.status == NOT_STARTED and tThisFlip >= 5.0-frameTolerance:
            # keep track of start time/frame for later
            No1left_3.frameNStart = frameN  # exact frame index
            No1left_3.tStart = t  # local t and not account for scr refresh
            No1left_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(No1left_3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'No1left_3.started')
            # update status
            No1left_3.status = STARTED
            No1left_3.setAutoDraw(True)
        
        # if No1left_3 is active this frame...
        if No1left_3.status == STARTED:
            # update params
            pass
        
        # if No1left_3 is stopping this frame...
        if No1left_3.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > No1left_3.tStartRefresh + 3.0-frameTolerance:
                # keep track of stop time/frame for later
                No1left_3.tStop = t  # not accounting for scr refresh
                No1left_3.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'No1left_3.stopped')
                # update status
                No1left_3.status = FINISHED
                No1left_3.setAutoDraw(False)
        
        # *No1right_3* updates
        
        # if No1right_3 is starting this frame...
        if No1right_3.status == NOT_STARTED and tThisFlip >= 5.0-frameTolerance:
            # keep track of start time/frame for later
            No1right_3.frameNStart = frameN  # exact frame index
            No1right_3.tStart = t  # local t and not account for scr refresh
            No1right_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(No1right_3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'No1right_3.started')
            # update status
            No1right_3.status = STARTED
            No1right_3.setAutoDraw(True)
        
        # if No1right_3 is active this frame...
        if No1right_3.status == STARTED:
            # update params
            pass
        
        # if No1right_3 is stopping this frame...
        if No1right_3.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > No1right_3.tStartRefresh + 3.0-frameTolerance:
                # keep track of stop time/frame for later
                No1right_3.tStop = t  # not accounting for scr refresh
                No1right_3.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'No1right_3.stopped')
                # update status
                No1right_3.status = FINISHED
                No1right_3.setAutoDraw(False)
        
        # *left_2* updates
        
        # if left_2 is starting this frame...
        if left_2.status == NOT_STARTED and tThisFlip >= 5.0-frameTolerance:
            # keep track of start time/frame for later
            left_2.frameNStart = frameN  # exact frame index
            left_2.tStart = t  # local t and not account for scr refresh
            left_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(left_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'left_2.started')
            # update status
            left_2.status = STARTED
            left_2.setAutoDraw(True)
        
        # if left_2 is active this frame...
        if left_2.status == STARTED:
            # update params
            pass
        
        # if left_2 is stopping this frame...
        if left_2.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > left_2.tStartRefresh + 3.0-frameTolerance:
                # keep track of stop time/frame for later
                left_2.tStop = t  # not accounting for scr refresh
                left_2.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'left_2.stopped')
                # update status
                left_2.status = FINISHED
                left_2.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in L_RComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "L_R" ---
    for thisComponent in L_RComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('L_R.stopped', globalClock.getTime())
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if routineForceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-8.000000)
    
    # --- Prepare to start Routine "Right" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('Right.started', globalClock.getTime())
    # keep track of which components have finished
    RightComponents = [left1_6, right1_6]
    for thisComponent in RightComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    outlet.push_sample([error])  # Push event marker.
    print("error")
    # --- Run Routine "Right" ---
    routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 3.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *left1_6* updates
        
        # if left1_6 is starting this frame...
        if left1_6.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            left1_6.frameNStart = frameN  # exact frame index
            left1_6.tStart = t  # local t and not account for scr refresh
            left1_6.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(left1_6, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'left1_6.started')
            # update status
            left1_6.status = STARTED
            left1_6.setAutoDraw(True)
        
        # if left1_6 is active this frame...
        if left1_6.status == STARTED:
            # update params
            pass
        
        # if left1_6 is stopping this frame...
        if left1_6.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > left1_6.tStartRefresh + 3.0-frameTolerance:
                # keep track of stop time/frame for later
                left1_6.tStop = t  # not accounting for scr refresh
                left1_6.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'left1_6.stopped')
                # update status
                left1_6.status = FINISHED
                left1_6.setAutoDraw(False)
        
        # *right1_6* updates
        
        # if right1_6 is starting this frame...
        if right1_6.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            right1_6.frameNStart = frameN  # exact frame index
            right1_6.tStart = t  # local t and not account for scr refresh
            right1_6.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(right1_6, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'right1_6.started')
            # update status
            right1_6.status = STARTED
            right1_6.setAutoDraw(True)
        
        # if right1_6 is active this frame...
        if right1_6.status == STARTED:
            # update params
            pass
        
        # if right1_6 is stopping this frame...
        if right1_6.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > right1_6.tStartRefresh + 3.0-frameTolerance:
                # keep track of stop time/frame for later
                right1_6.tStop = t  # not accounting for scr refresh
                right1_6.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'right1_6.stopped')
                # update status
                right1_6.status = FINISHED
                right1_6.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in RightComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Right" ---
    for thisComponent in RightComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('Right.stopped', globalClock.getTime())
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if routineForceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-3.000000)
    
    # --- Prepare to start Routine "R_R" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('R_R.started', globalClock.getTime())
    # keep track of which components have finished
    R_RComponents = [No1left_2, No1right_2, right]
    for thisComponent in R_RComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "R_R" ---
    routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 8.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *No1left_2* updates
        
        # if No1left_2 is starting this frame...
        if No1left_2.status == NOT_STARTED and tThisFlip >= 5.0-frameTolerance:
            # keep track of start time/frame for later
            No1left_2.frameNStart = frameN  # exact frame index
            No1left_2.tStart = t  # local t and not account for scr refresh
            No1left_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(No1left_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'No1left_2.started')
            # update status
            No1left_2.status = STARTED
            No1left_2.setAutoDraw(True)
        
        # if No1left_2 is active this frame...
        if No1left_2.status == STARTED:
            # update params
            pass
        
        # if No1left_2 is stopping this frame...
        if No1left_2.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > No1left_2.tStartRefresh + 3.0-frameTolerance:
                # keep track of stop time/frame for later
                No1left_2.tStop = t  # not accounting for scr refresh
                No1left_2.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'No1left_2.stopped')
                # update status
                No1left_2.status = FINISHED
                No1left_2.setAutoDraw(False)
        
        # *No1right_2* updates
        
        # if No1right_2 is starting this frame...
        if No1right_2.status == NOT_STARTED and tThisFlip >= 5.0-frameTolerance:
            # keep track of start time/frame for later
            No1right_2.frameNStart = frameN  # exact frame index
            No1right_2.tStart = t  # local t and not account for scr refresh
            No1right_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(No1right_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'No1right_2.started')
            # update status
            No1right_2.status = STARTED
            No1right_2.setAutoDraw(True)
        
        # if No1right_2 is active this frame...
        if No1right_2.status == STARTED:
            # update params
            pass
        
        # if No1right_2 is stopping this frame...
        if No1right_2.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > No1right_2.tStartRefresh + 3.0-frameTolerance:
                # keep track of stop time/frame for later
                No1right_2.tStop = t  # not accounting for scr refresh
                No1right_2.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'No1right_2.stopped')
                # update status
                No1right_2.status = FINISHED
                No1right_2.setAutoDraw(False)
        
        # *right* updates
        
        # if right is starting this frame...
        if right.status == NOT_STARTED and tThisFlip >= 5.0-frameTolerance:
            # keep track of start time/frame for later
            right.frameNStart = frameN  # exact frame index
            right.tStart = t  # local t and not account for scr refresh
            right.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(right, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'right.started')
            # update status
            right.status = STARTED
            right.setAutoDraw(True)
        
        # if right is active this frame...
        if right.status == STARTED:
            # update params
            pass
        
        # if right is stopping this frame...
        if right.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > right.tStartRefresh + 3.0-frameTolerance:
                # keep track of stop time/frame for later
                right.tStop = t  # not accounting for scr refresh
                right.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'right.stopped')
                # update status
                right.status = FINISHED
                right.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in R_RComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "R_R" ---
    for thisComponent in R_RComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('R_R.stopped', globalClock.getTime())
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if routineForceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-8.000000)
    
    # --- Prepare to start Routine "Right" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('Right.started', globalClock.getTime())
    # keep track of which components have finished
    RightComponents = [left1_6, right1_6]
    for thisComponent in RightComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    outlet.push_sample([correct])  # Push event marker.
    print("correct")
    # --- Run Routine "Right" ---
    routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 3.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *left1_6* updates
        
        # if left1_6 is starting this frame...
        if left1_6.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            left1_6.frameNStart = frameN  # exact frame index
            left1_6.tStart = t  # local t and not account for scr refresh
            left1_6.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(left1_6, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'left1_6.started')
            # update status
            left1_6.status = STARTED
            left1_6.setAutoDraw(True)
        
        # if left1_6 is active this frame...
        if left1_6.status == STARTED:
            # update params
            pass
        
        # if left1_6 is stopping this frame...
        if left1_6.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > left1_6.tStartRefresh + 3.0-frameTolerance:
                # keep track of stop time/frame for later
                left1_6.tStop = t  # not accounting for scr refresh
                left1_6.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'left1_6.stopped')
                # update status
                left1_6.status = FINISHED
                left1_6.setAutoDraw(False)
        
        # *right1_6* updates
        
        # if right1_6 is starting this frame...
        if right1_6.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            right1_6.frameNStart = frameN  # exact frame index
            right1_6.tStart = t  # local t and not account for scr refresh
            right1_6.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(right1_6, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'right1_6.started')
            # update status
            right1_6.status = STARTED
            right1_6.setAutoDraw(True)
        
        # if right1_6 is active this frame...
        if right1_6.status == STARTED:
            # update params
            pass
        
        # if right1_6 is stopping this frame...
        if right1_6.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > right1_6.tStartRefresh + 3.0-frameTolerance:
                # keep track of stop time/frame for later
                right1_6.tStop = t  # not accounting for scr refresh
                right1_6.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'right1_6.stopped')
                # update status
                right1_6.status = FINISHED
                right1_6.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in RightComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Right" ---
    for thisComponent in RightComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('Right.stopped', globalClock.getTime())
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if routineForceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-3.000000)
    
    # --- Prepare to start Routine "L_L" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('L_L.started', globalClock.getTime())
    # keep track of which components have finished
    L_LComponents = [No1left, No1right, left_1]
    for thisComponent in L_LComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "L_L" ---
    routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 8.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *No1left* updates
        
        # if No1left is starting this frame...
        if No1left.status == NOT_STARTED and tThisFlip >= 5.0-frameTolerance:
            # keep track of start time/frame for later
            No1left.frameNStart = frameN  # exact frame index
            No1left.tStart = t  # local t and not account for scr refresh
            No1left.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(No1left, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'No1left.started')
            # update status
            No1left.status = STARTED
            No1left.setAutoDraw(True)
        
        # if No1left is active this frame...
        if No1left.status == STARTED:
            # update params
            pass
        
        # if No1left is stopping this frame...
        if No1left.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > No1left.tStartRefresh + 3.0-frameTolerance:
                # keep track of stop time/frame for later
                No1left.tStop = t  # not accounting for scr refresh
                No1left.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'No1left.stopped')
                # update status
                No1left.status = FINISHED
                No1left.setAutoDraw(False)
        
        # *No1right* updates
        
        # if No1right is starting this frame...
        if No1right.status == NOT_STARTED and tThisFlip >= 5.0-frameTolerance:
            # keep track of start time/frame for later
            No1right.frameNStart = frameN  # exact frame index
            No1right.tStart = t  # local t and not account for scr refresh
            No1right.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(No1right, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'No1right.started')
            # update status
            No1right.status = STARTED
            No1right.setAutoDraw(True)
        
        # if No1right is active this frame...
        if No1right.status == STARTED:
            # update params
            pass
        
        # if No1right is stopping this frame...
        if No1right.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > No1right.tStartRefresh + 3.0-frameTolerance:
                # keep track of stop time/frame for later
                No1right.tStop = t  # not accounting for scr refresh
                No1right.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'No1right.stopped')
                # update status
                No1right.status = FINISHED
                No1right.setAutoDraw(False)
        
        # *left_1* updates
        
        # if left_1 is starting this frame...
        if left_1.status == NOT_STARTED and tThisFlip >= 5.0-frameTolerance:
            # keep track of start time/frame for later
            left_1.frameNStart = frameN  # exact frame index
            left_1.tStart = t  # local t and not account for scr refresh
            left_1.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(left_1, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'left_1.started')
            # update status
            left_1.status = STARTED
            left_1.setAutoDraw(True)
        
        # if left_1 is active this frame...
        if left_1.status == STARTED:
            # update params
            pass
        
        # if left_1 is stopping this frame...
        if left_1.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > left_1.tStartRefresh + 3.0-frameTolerance:
                # keep track of stop time/frame for later
                left_1.tStop = t  # not accounting for scr refresh
                left_1.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'left_1.stopped')
                # update status
                left_1.status = FINISHED
                left_1.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in L_LComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "L_L" ---
    for thisComponent in L_LComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('L_L.stopped', globalClock.getTime())
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if routineForceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-8.000000)
    
    # --- Prepare to start Routine "left" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('left.started', globalClock.getTime())
    # keep track of which components have finished
    leftComponents = [left1_5, right1_5]
    for thisComponent in leftComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    outlet.push_sample([correct])  # Push event marker.
    print("correct")
    # --- Run Routine "left" ---
    routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 3.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *left1_5* updates
        
        # if left1_5 is starting this frame...
        if left1_5.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            left1_5.frameNStart = frameN  # exact frame index
            left1_5.tStart = t  # local t and not account for scr refresh
            left1_5.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(left1_5, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'left1_5.started')
            # update status
            left1_5.status = STARTED
            left1_5.setAutoDraw(True)
        
        # if left1_5 is active this frame...
        if left1_5.status == STARTED:
            # update params
            pass
        
        # if left1_5 is stopping this frame...
        if left1_5.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > left1_5.tStartRefresh + 3.0-frameTolerance:
                # keep track of stop time/frame for later
                left1_5.tStop = t  # not accounting for scr refresh
                left1_5.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'left1_5.stopped')
                # update status
                left1_5.status = FINISHED
                left1_5.setAutoDraw(False)
        
        # *right1_5* updates
        
        # if right1_5 is starting this frame...
        if right1_5.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            right1_5.frameNStart = frameN  # exact frame index
            right1_5.tStart = t  # local t and not account for scr refresh
            right1_5.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(right1_5, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'right1_5.started')
            # update status
            right1_5.status = STARTED
            right1_5.setAutoDraw(True)
        
        # if right1_5 is active this frame...
        if right1_5.status == STARTED:
            # update params
            pass
        
        # if right1_5 is stopping this frame...
        if right1_5.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > right1_5.tStartRefresh + 3.0-frameTolerance:
                # keep track of stop time/frame for later
                right1_5.tStop = t  # not accounting for scr refresh
                right1_5.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'right1_5.stopped')
                # update status
                right1_5.status = FINISHED
                right1_5.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in leftComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "left" ---
    for thisComponent in leftComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('left.stopped', globalClock.getTime())
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if routineForceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-3.000000)
    
    # --- Prepare to start Routine "R_L" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('R_L.started', globalClock.getTime())
    # keep track of which components have finished
    R_LComponents = [No1left_4, No1right_4, right_2]
    for thisComponent in R_LComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "R_L" ---
    routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 8.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *No1left_4* updates
        
        # if No1left_4 is starting this frame...
        if No1left_4.status == NOT_STARTED and tThisFlip >= 5.0-frameTolerance:
            # keep track of start time/frame for later
            No1left_4.frameNStart = frameN  # exact frame index
            No1left_4.tStart = t  # local t and not account for scr refresh
            No1left_4.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(No1left_4, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'No1left_4.started')
            # update status
            No1left_4.status = STARTED
            No1left_4.setAutoDraw(True)
        
        # if No1left_4 is active this frame...
        if No1left_4.status == STARTED:
            # update params
            pass
        
        # if No1left_4 is stopping this frame...
        if No1left_4.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > No1left_4.tStartRefresh + 3.0-frameTolerance:
                # keep track of stop time/frame for later
                No1left_4.tStop = t  # not accounting for scr refresh
                No1left_4.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'No1left_4.stopped')
                # update status
                No1left_4.status = FINISHED
                No1left_4.setAutoDraw(False)
        
        # *No1right_4* updates
        
        # if No1right_4 is starting this frame...
        if No1right_4.status == NOT_STARTED and tThisFlip >= 5.0-frameTolerance:
            # keep track of start time/frame for later
            No1right_4.frameNStart = frameN  # exact frame index
            No1right_4.tStart = t  # local t and not account for scr refresh
            No1right_4.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(No1right_4, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'No1right_4.started')
            # update status
            No1right_4.status = STARTED
            No1right_4.setAutoDraw(True)
        
        # if No1right_4 is active this frame...
        if No1right_4.status == STARTED:
            # update params
            pass
        
        # if No1right_4 is stopping this frame...
        if No1right_4.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > No1right_4.tStartRefresh + 3.0-frameTolerance:
                # keep track of stop time/frame for later
                No1right_4.tStop = t  # not accounting for scr refresh
                No1right_4.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'No1right_4.stopped')
                # update status
                No1right_4.status = FINISHED
                No1right_4.setAutoDraw(False)
        
        # *right_2* updates
        
        # if right_2 is starting this frame...
        if right_2.status == NOT_STARTED and tThisFlip >= 5.0-frameTolerance:
            # keep track of start time/frame for later
            right_2.frameNStart = frameN  # exact frame index
            right_2.tStart = t  # local t and not account for scr refresh
            right_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(right_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'right_2.started')
            # update status
            right_2.status = STARTED
            right_2.setAutoDraw(True)
        
        # if right_2 is active this frame...
        if right_2.status == STARTED:
            # update params
            pass
        
        # if right_2 is stopping this frame...
        if right_2.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > right_2.tStartRefresh + 3.0-frameTolerance:
                # keep track of stop time/frame for later
                right_2.tStop = t  # not accounting for scr refresh
                right_2.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'right_2.stopped')
                # update status
                right_2.status = FINISHED
                right_2.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in R_LComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "R_L" ---
    for thisComponent in R_LComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('R_L.stopped', globalClock.getTime())
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if routineForceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-8.000000)
    
    # --- Prepare to start Routine "left" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('left.started', globalClock.getTime())
    # keep track of which components have finished
    leftComponents = [left1_5, right1_5]
    for thisComponent in leftComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    outlet.push_sample([error])  # Push event marker.
    print("error")
    # --- Run Routine "left" ---
    routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 3.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *left1_5* updates
        
        # if left1_5 is starting this frame...
        if left1_5.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            left1_5.frameNStart = frameN  # exact frame index
            left1_5.tStart = t  # local t and not account for scr refresh
            left1_5.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(left1_5, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'left1_5.started')
            # update status
            left1_5.status = STARTED
            left1_5.setAutoDraw(True)
        
        # if left1_5 is active this frame...
        if left1_5.status == STARTED:
            # update params
            pass
        
        # if left1_5 is stopping this frame...
        if left1_5.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > left1_5.tStartRefresh + 3.0-frameTolerance:
                # keep track of stop time/frame for later
                left1_5.tStop = t  # not accounting for scr refresh
                left1_5.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'left1_5.stopped')
                # update status
                left1_5.status = FINISHED
                left1_5.setAutoDraw(False)
        
        # *right1_5* updates
        
        # if right1_5 is starting this frame...
        if right1_5.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            right1_5.frameNStart = frameN  # exact frame index
            right1_5.tStart = t  # local t and not account for scr refresh
            right1_5.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(right1_5, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'right1_5.started')
            # update status
            right1_5.status = STARTED
            right1_5.setAutoDraw(True)
        
        # if right1_5 is active this frame...
        if right1_5.status == STARTED:
            # update params
            pass
        
        # if right1_5 is stopping this frame...
        if right1_5.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > right1_5.tStartRefresh + 3.0-frameTolerance:
                # keep track of stop time/frame for later
                right1_5.tStop = t  # not accounting for scr refresh
                right1_5.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'right1_5.stopped')
                # update status
                right1_5.status = FINISHED
                right1_5.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in leftComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "left" ---
    for thisComponent in leftComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('left.stopped', globalClock.getTime())
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if routineForceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-3.000000)
    
    # --- Prepare to start Routine "L_L" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('L_L.started', globalClock.getTime())
    # keep track of which components have finished
    L_LComponents = [No1left, No1right, left_1]
    for thisComponent in L_LComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "L_L" ---
    routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 8.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *No1left* updates
        
        # if No1left is starting this frame...
        if No1left.status == NOT_STARTED and tThisFlip >= 5.0-frameTolerance:
            # keep track of start time/frame for later
            No1left.frameNStart = frameN  # exact frame index
            No1left.tStart = t  # local t and not account for scr refresh
            No1left.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(No1left, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'No1left.started')
            # update status
            No1left.status = STARTED
            No1left.setAutoDraw(True)
        
        # if No1left is active this frame...
        if No1left.status == STARTED:
            # update params
            pass
        
        # if No1left is stopping this frame...
        if No1left.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > No1left.tStartRefresh + 3.0-frameTolerance:
                # keep track of stop time/frame for later
                No1left.tStop = t  # not accounting for scr refresh
                No1left.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'No1left.stopped')
                # update status
                No1left.status = FINISHED
                No1left.setAutoDraw(False)
        
        # *No1right* updates
        
        # if No1right is starting this frame...
        if No1right.status == NOT_STARTED and tThisFlip >= 5.0-frameTolerance:
            # keep track of start time/frame for later
            No1right.frameNStart = frameN  # exact frame index
            No1right.tStart = t  # local t and not account for scr refresh
            No1right.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(No1right, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'No1right.started')
            # update status
            No1right.status = STARTED
            No1right.setAutoDraw(True)
        
        # if No1right is active this frame...
        if No1right.status == STARTED:
            # update params
            pass
        
        # if No1right is stopping this frame...
        if No1right.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > No1right.tStartRefresh + 3.0-frameTolerance:
                # keep track of stop time/frame for later
                No1right.tStop = t  # not accounting for scr refresh
                No1right.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'No1right.stopped')
                # update status
                No1right.status = FINISHED
                No1right.setAutoDraw(False)
        
        # *left_1* updates
        
        # if left_1 is starting this frame...
        if left_1.status == NOT_STARTED and tThisFlip >= 5.0-frameTolerance:
            # keep track of start time/frame for later
            left_1.frameNStart = frameN  # exact frame index
            left_1.tStart = t  # local t and not account for scr refresh
            left_1.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(left_1, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'left_1.started')
            # update status
            left_1.status = STARTED
            left_1.setAutoDraw(True)
        
        # if left_1 is active this frame...
        if left_1.status == STARTED:
            # update params
            pass
        
        # if left_1 is stopping this frame...
        if left_1.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > left_1.tStartRefresh + 3.0-frameTolerance:
                # keep track of stop time/frame for later
                left_1.tStop = t  # not accounting for scr refresh
                left_1.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'left_1.stopped')
                # update status
                left_1.status = FINISHED
                left_1.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in L_LComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "L_L" ---
    for thisComponent in L_LComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('L_L.stopped', globalClock.getTime())
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if routineForceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-8.000000)
    
    # --- Prepare to start Routine "left" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('left.started', globalClock.getTime())
    # keep track of which components have finished
    leftComponents = [left1_5, right1_5]
    for thisComponent in leftComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    outlet.push_sample([correct])  # Push event marker.
    print("correct")
    # --- Run Routine "left" ---
    routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 3.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *left1_5* updates
        
        # if left1_5 is starting this frame...
        if left1_5.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            left1_5.frameNStart = frameN  # exact frame index
            left1_5.tStart = t  # local t and not account for scr refresh
            left1_5.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(left1_5, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'left1_5.started')
            # update status
            left1_5.status = STARTED
            left1_5.setAutoDraw(True)
        
        # if left1_5 is active this frame...
        if left1_5.status == STARTED:
            # update params
            pass
        
        # if left1_5 is stopping this frame...
        if left1_5.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > left1_5.tStartRefresh + 3.0-frameTolerance:
                # keep track of stop time/frame for later
                left1_5.tStop = t  # not accounting for scr refresh
                left1_5.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'left1_5.stopped')
                # update status
                left1_5.status = FINISHED
                left1_5.setAutoDraw(False)
        
        # *right1_5* updates
        
        # if right1_5 is starting this frame...
        if right1_5.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            right1_5.frameNStart = frameN  # exact frame index
            right1_5.tStart = t  # local t and not account for scr refresh
            right1_5.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(right1_5, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'right1_5.started')
            # update status
            right1_5.status = STARTED
            right1_5.setAutoDraw(True)
        
        # if right1_5 is active this frame...
        if right1_5.status == STARTED:
            # update params
            pass
        
        # if right1_5 is stopping this frame...
        if right1_5.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > right1_5.tStartRefresh + 3.0-frameTolerance:
                # keep track of stop time/frame for later
                right1_5.tStop = t  # not accounting for scr refresh
                right1_5.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'right1_5.stopped')
                # update status
                right1_5.status = FINISHED
                right1_5.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in leftComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "left" ---
    for thisComponent in leftComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('left.stopped', globalClock.getTime())
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if routineForceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-3.000000)
    
    # --- Prepare to start Routine "R_R" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('R_R.started', globalClock.getTime())
    # keep track of which components have finished
    R_RComponents = [No1left_2, No1right_2, right]
    for thisComponent in R_RComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "R_R" ---
    routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 8.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *No1left_2* updates
        
        # if No1left_2 is starting this frame...
        if No1left_2.status == NOT_STARTED and tThisFlip >= 5.0-frameTolerance:
            # keep track of start time/frame for later
            No1left_2.frameNStart = frameN  # exact frame index
            No1left_2.tStart = t  # local t and not account for scr refresh
            No1left_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(No1left_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'No1left_2.started')
            # update status
            No1left_2.status = STARTED
            No1left_2.setAutoDraw(True)
        
        # if No1left_2 is active this frame...
        if No1left_2.status == STARTED:
            # update params
            pass
        
        # if No1left_2 is stopping this frame...
        if No1left_2.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > No1left_2.tStartRefresh + 3.0-frameTolerance:
                # keep track of stop time/frame for later
                No1left_2.tStop = t  # not accounting for scr refresh
                No1left_2.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'No1left_2.stopped')
                # update status
                No1left_2.status = FINISHED
                No1left_2.setAutoDraw(False)
        
        # *No1right_2* updates
        
        # if No1right_2 is starting this frame...
        if No1right_2.status == NOT_STARTED and tThisFlip >= 5.0-frameTolerance:
            # keep track of start time/frame for later
            No1right_2.frameNStart = frameN  # exact frame index
            No1right_2.tStart = t  # local t and not account for scr refresh
            No1right_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(No1right_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'No1right_2.started')
            # update status
            No1right_2.status = STARTED
            No1right_2.setAutoDraw(True)
        
        # if No1right_2 is active this frame...
        if No1right_2.status == STARTED:
            # update params
            pass
        
        # if No1right_2 is stopping this frame...
        if No1right_2.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > No1right_2.tStartRefresh + 3.0-frameTolerance:
                # keep track of stop time/frame for later
                No1right_2.tStop = t  # not accounting for scr refresh
                No1right_2.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'No1right_2.stopped')
                # update status
                No1right_2.status = FINISHED
                No1right_2.setAutoDraw(False)
        
        # *right* updates
        
        # if right is starting this frame...
        if right.status == NOT_STARTED and tThisFlip >= 5.0-frameTolerance:
            # keep track of start time/frame for later
            right.frameNStart = frameN  # exact frame index
            right.tStart = t  # local t and not account for scr refresh
            right.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(right, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'right.started')
            # update status
            right.status = STARTED
            right.setAutoDraw(True)
        
        # if right is active this frame...
        if right.status == STARTED:
            # update params
            pass
        
        # if right is stopping this frame...
        if right.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > right.tStartRefresh + 3.0-frameTolerance:
                # keep track of stop time/frame for later
                right.tStop = t  # not accounting for scr refresh
                right.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'right.stopped')
                # update status
                right.status = FINISHED
                right.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in R_RComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "R_R" ---
    for thisComponent in R_RComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('R_R.stopped', globalClock.getTime())
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if routineForceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-8.000000)
    
    # --- Prepare to start Routine "Right" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('Right.started', globalClock.getTime())
    # keep track of which components have finished
    RightComponents = [left1_6, right1_6]
    for thisComponent in RightComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    outlet.push_sample([correct])  # Push event marker.
    print("correct")
    # --- Run Routine "Right" ---
    routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 3.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *left1_6* updates
        
        # if left1_6 is starting this frame...
        if left1_6.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            left1_6.frameNStart = frameN  # exact frame index
            left1_6.tStart = t  # local t and not account for scr refresh
            left1_6.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(left1_6, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'left1_6.started')
            # update status
            left1_6.status = STARTED
            left1_6.setAutoDraw(True)
        
        # if left1_6 is active this frame...
        if left1_6.status == STARTED:
            # update params
            pass
        
        # if left1_6 is stopping this frame...
        if left1_6.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > left1_6.tStartRefresh + 3.0-frameTolerance:
                # keep track of stop time/frame for later
                left1_6.tStop = t  # not accounting for scr refresh
                left1_6.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'left1_6.stopped')
                # update status
                left1_6.status = FINISHED
                left1_6.setAutoDraw(False)
        
        # *right1_6* updates
        
        # if right1_6 is starting this frame...
        if right1_6.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            right1_6.frameNStart = frameN  # exact frame index
            right1_6.tStart = t  # local t and not account for scr refresh
            right1_6.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(right1_6, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'right1_6.started')
            # update status
            right1_6.status = STARTED
            right1_6.setAutoDraw(True)
        
        # if right1_6 is active this frame...
        if right1_6.status == STARTED:
            # update params
            pass
        
        # if right1_6 is stopping this frame...
        if right1_6.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > right1_6.tStartRefresh + 3.0-frameTolerance:
                # keep track of stop time/frame for later
                right1_6.tStop = t  # not accounting for scr refresh
                right1_6.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'right1_6.stopped')
                # update status
                right1_6.status = FINISHED
                right1_6.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in RightComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Right" ---
    for thisComponent in RightComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('Right.stopped', globalClock.getTime())
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if routineForceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-3.000000)
    
    # --- Prepare to start Routine "L_L" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('L_L.started', globalClock.getTime())
    # keep track of which components have finished
    L_LComponents = [No1left, No1right, left_1]
    for thisComponent in L_LComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "L_L" ---
    routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 8.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *No1left* updates
        
        # if No1left is starting this frame...
        if No1left.status == NOT_STARTED and tThisFlip >= 5.0-frameTolerance:
            # keep track of start time/frame for later
            No1left.frameNStart = frameN  # exact frame index
            No1left.tStart = t  # local t and not account for scr refresh
            No1left.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(No1left, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'No1left.started')
            # update status
            No1left.status = STARTED
            No1left.setAutoDraw(True)
        
        # if No1left is active this frame...
        if No1left.status == STARTED:
            # update params
            pass
        
        # if No1left is stopping this frame...
        if No1left.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > No1left.tStartRefresh + 3.0-frameTolerance:
                # keep track of stop time/frame for later
                No1left.tStop = t  # not accounting for scr refresh
                No1left.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'No1left.stopped')
                # update status
                No1left.status = FINISHED
                No1left.setAutoDraw(False)
        
        # *No1right* updates
        
        # if No1right is starting this frame...
        if No1right.status == NOT_STARTED and tThisFlip >= 5.0-frameTolerance:
            # keep track of start time/frame for later
            No1right.frameNStart = frameN  # exact frame index
            No1right.tStart = t  # local t and not account for scr refresh
            No1right.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(No1right, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'No1right.started')
            # update status
            No1right.status = STARTED
            No1right.setAutoDraw(True)
        
        # if No1right is active this frame...
        if No1right.status == STARTED:
            # update params
            pass
        
        # if No1right is stopping this frame...
        if No1right.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > No1right.tStartRefresh + 3.0-frameTolerance:
                # keep track of stop time/frame for later
                No1right.tStop = t  # not accounting for scr refresh
                No1right.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'No1right.stopped')
                # update status
                No1right.status = FINISHED
                No1right.setAutoDraw(False)
        
        # *left_1* updates
        
        # if left_1 is starting this frame...
        if left_1.status == NOT_STARTED and tThisFlip >= 5.0-frameTolerance:
            # keep track of start time/frame for later
            left_1.frameNStart = frameN  # exact frame index
            left_1.tStart = t  # local t and not account for scr refresh
            left_1.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(left_1, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'left_1.started')
            # update status
            left_1.status = STARTED
            left_1.setAutoDraw(True)
        
        # if left_1 is active this frame...
        if left_1.status == STARTED:
            # update params
            pass
        
        # if left_1 is stopping this frame...
        if left_1.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > left_1.tStartRefresh + 3.0-frameTolerance:
                # keep track of stop time/frame for later
                left_1.tStop = t  # not accounting for scr refresh
                left_1.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'left_1.stopped')
                # update status
                left_1.status = FINISHED
                left_1.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in L_LComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "L_L" ---
    for thisComponent in L_LComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('L_L.stopped', globalClock.getTime())
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if routineForceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-8.000000)
    
    # --- Prepare to start Routine "left" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('left.started', globalClock.getTime())
    # keep track of which components have finished
    leftComponents = [left1_5, right1_5]
    for thisComponent in leftComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    outlet.push_sample([correct])  # Push event marker.
    print("correct")
    # --- Run Routine "left" ---
    routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 3.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *left1_5* updates
        
        # if left1_5 is starting this frame...
        if left1_5.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            left1_5.frameNStart = frameN  # exact frame index
            left1_5.tStart = t  # local t and not account for scr refresh
            left1_5.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(left1_5, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'left1_5.started')
            # update status
            left1_5.status = STARTED
            left1_5.setAutoDraw(True)
        
        # if left1_5 is active this frame...
        if left1_5.status == STARTED:
            # update params
            pass
        
        # if left1_5 is stopping this frame...
        if left1_5.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > left1_5.tStartRefresh + 3.0-frameTolerance:
                # keep track of stop time/frame for later
                left1_5.tStop = t  # not accounting for scr refresh
                left1_5.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'left1_5.stopped')
                # update status
                left1_5.status = FINISHED
                left1_5.setAutoDraw(False)
        
        # *right1_5* updates
        
        # if right1_5 is starting this frame...
        if right1_5.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            right1_5.frameNStart = frameN  # exact frame index
            right1_5.tStart = t  # local t and not account for scr refresh
            right1_5.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(right1_5, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'right1_5.started')
            # update status
            right1_5.status = STARTED
            right1_5.setAutoDraw(True)
        
        # if right1_5 is active this frame...
        if right1_5.status == STARTED:
            # update params
            pass
        
        # if right1_5 is stopping this frame...
        if right1_5.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > right1_5.tStartRefresh + 3.0-frameTolerance:
                # keep track of stop time/frame for later
                right1_5.tStop = t  # not accounting for scr refresh
                right1_5.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'right1_5.stopped')
                # update status
                right1_5.status = FINISHED
                right1_5.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in leftComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "left" ---
    for thisComponent in leftComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('left.stopped', globalClock.getTime())
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if routineForceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-3.000000)
    
    # --- Prepare to start Routine "R_R" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('R_R.started', globalClock.getTime())
    # keep track of which components have finished
    R_RComponents = [No1left_2, No1right_2, right]
    for thisComponent in R_RComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "R_R" ---
    routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 8.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *No1left_2* updates
        
        # if No1left_2 is starting this frame...
        if No1left_2.status == NOT_STARTED and tThisFlip >= 5.0-frameTolerance:
            # keep track of start time/frame for later
            No1left_2.frameNStart = frameN  # exact frame index
            No1left_2.tStart = t  # local t and not account for scr refresh
            No1left_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(No1left_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'No1left_2.started')
            # update status
            No1left_2.status = STARTED
            No1left_2.setAutoDraw(True)
        
        # if No1left_2 is active this frame...
        if No1left_2.status == STARTED:
            # update params
            pass
        
        # if No1left_2 is stopping this frame...
        if No1left_2.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > No1left_2.tStartRefresh + 3.0-frameTolerance:
                # keep track of stop time/frame for later
                No1left_2.tStop = t  # not accounting for scr refresh
                No1left_2.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'No1left_2.stopped')
                # update status
                No1left_2.status = FINISHED
                No1left_2.setAutoDraw(False)
        
        # *No1right_2* updates
        
        # if No1right_2 is starting this frame...
        if No1right_2.status == NOT_STARTED and tThisFlip >= 5.0-frameTolerance:
            # keep track of start time/frame for later
            No1right_2.frameNStart = frameN  # exact frame index
            No1right_2.tStart = t  # local t and not account for scr refresh
            No1right_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(No1right_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'No1right_2.started')
            # update status
            No1right_2.status = STARTED
            No1right_2.setAutoDraw(True)
        
        # if No1right_2 is active this frame...
        if No1right_2.status == STARTED:
            # update params
            pass
        
        # if No1right_2 is stopping this frame...
        if No1right_2.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > No1right_2.tStartRefresh + 3.0-frameTolerance:
                # keep track of stop time/frame for later
                No1right_2.tStop = t  # not accounting for scr refresh
                No1right_2.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'No1right_2.stopped')
                # update status
                No1right_2.status = FINISHED
                No1right_2.setAutoDraw(False)
        
        # *right* updates
        
        # if right is starting this frame...
        if right.status == NOT_STARTED and tThisFlip >= 5.0-frameTolerance:
            # keep track of start time/frame for later
            right.frameNStart = frameN  # exact frame index
            right.tStart = t  # local t and not account for scr refresh
            right.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(right, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'right.started')
            # update status
            right.status = STARTED
            right.setAutoDraw(True)
        
        # if right is active this frame...
        if right.status == STARTED:
            # update params
            pass
        
        # if right is stopping this frame...
        if right.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > right.tStartRefresh + 3.0-frameTolerance:
                # keep track of stop time/frame for later
                right.tStop = t  # not accounting for scr refresh
                right.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'right.stopped')
                # update status
                right.status = FINISHED
                right.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in R_RComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "R_R" ---
    for thisComponent in R_RComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('R_R.stopped', globalClock.getTime())
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if routineForceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-8.000000)
    
    # --- Prepare to start Routine "Right" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('Right.started', globalClock.getTime())
    # keep track of which components have finished
    RightComponents = [left1_6, right1_6]
    for thisComponent in RightComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    outlet.push_sample([correct])  # Push event marker.
    print("correct")
    # --- Run Routine "Right" ---
    routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 3.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *left1_6* updates
        
        # if left1_6 is starting this frame...
        if left1_6.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            left1_6.frameNStart = frameN  # exact frame index
            left1_6.tStart = t  # local t and not account for scr refresh
            left1_6.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(left1_6, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'left1_6.started')
            # update status
            left1_6.status = STARTED
            left1_6.setAutoDraw(True)
        
        # if left1_6 is active this frame...
        if left1_6.status == STARTED:
            # update params
            pass
        
        # if left1_6 is stopping this frame...
        if left1_6.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > left1_6.tStartRefresh + 3.0-frameTolerance:
                # keep track of stop time/frame for later
                left1_6.tStop = t  # not accounting for scr refresh
                left1_6.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'left1_6.stopped')
                # update status
                left1_6.status = FINISHED
                left1_6.setAutoDraw(False)
        
        # *right1_6* updates
        
        # if right1_6 is starting this frame...
        if right1_6.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            right1_6.frameNStart = frameN  # exact frame index
            right1_6.tStart = t  # local t and not account for scr refresh
            right1_6.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(right1_6, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'right1_6.started')
            # update status
            right1_6.status = STARTED
            right1_6.setAutoDraw(True)
        
        # if right1_6 is active this frame...
        if right1_6.status == STARTED:
            # update params
            pass
        
        # if right1_6 is stopping this frame...
        if right1_6.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > right1_6.tStartRefresh + 3.0-frameTolerance:
                # keep track of stop time/frame for later
                right1_6.tStop = t  # not accounting for scr refresh
                right1_6.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'right1_6.stopped')
                # update status
                right1_6.status = FINISHED
                right1_6.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in RightComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Right" ---
    for thisComponent in RightComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('Right.stopped', globalClock.getTime())
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if routineForceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-3.000000)
    
    # --- Prepare to start Routine "end" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('end.started', globalClock.getTime())
    # keep track of which components have finished
    endComponents = [finish]
    for thisComponent in endComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "end" ---
    routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 5.5:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *finish* updates
        
        # if finish is starting this frame...
        if finish.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
            # keep track of start time/frame for later
            finish.frameNStart = frameN  # exact frame index
            finish.tStart = t  # local t and not account for scr refresh
            finish.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(finish, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'finish.started')
            # update status
            finish.status = STARTED
            finish.setAutoDraw(True)
        
        # if finish is active this frame...
        if finish.status == STARTED:
            # update params
            pass
        
        # if finish is stopping this frame...
        if finish.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > finish.tStartRefresh + 5.0-frameTolerance:
                # keep track of stop time/frame for later
                finish.tStop = t  # not accounting for scr refresh
                finish.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'finish.stopped')
                # update status
                finish.status = FINISHED
                finish.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in endComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "end" ---
    for thisComponent in endComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('end.stopped', globalClock.getTime())
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if routineForceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-5.500000)
    
    # mark experiment as finished
    endExperiment(thisExp, win=win, inputs=inputs)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, inputs=None, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    inputs : dict
        Dictionary of input devices by name.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # mark experiment handler as finished
    thisExp.status = FINISHED
    # shut down eyetracker, if there is one
    if inputs is not None:
        if 'eyetracker' in inputs and inputs['eyetracker'] is not None:
            inputs['eyetracker'].setConnectionState(False)
    logging.flush()


def quit(thisExp, win=None, inputs=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    inputs : dict
        Dictionary of input devices by name.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    if inputs is not None:
        if 'eyetracker' in inputs and inputs['eyetracker'] is not None:
            inputs['eyetracker'].setConnectionState(False)
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    inputs = setupInputs(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win, 
        inputs=inputs
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win, inputs=inputs)
