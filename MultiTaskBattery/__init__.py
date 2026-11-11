"""MultiTaskBattery: a toolbox for running multi-domain cognitive and motor
tasks during fMRI sessions, built on PsychoPy."""

import os

# PsychoPy imports pygame when it is installed (MTB needs it for gamepads): hide its banner
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')

__version__ = "1.4.4"
