"""Read a game controller as if its buttons were the response keys.

Set ``gamepad_buttons`` in constants.py, one button per entry of ``response_keys``,
in the same order::

    response_keys   = ['1', '2', '3', '4']
    gamepad_buttons = ['X', 'Y', 'A', 'B']

A press of X then counts as the key '1', Y as '2', and so on, wherever MTB reads a
response. The keyboard keeps working. Without ``gamepad_buttons`` nothing changes.

A button is a name (A, B, X, Y, START, LEFTSHOULDER, DPAD_UP, ...) for a pad SDL
knows (Xbox, PlayStation, 8BitDo), or a number, the raw button number, for a pad it
does not know (a Current Designs 932 in joystick mode).

The pad is polled, not buffered: a press counts when MTB looks for responses, and
presses made while it is not looking are dropped, as event.clearEvents() drops keys.
"""
import os

from psychopy import event, core

_keys = {}      # button -> response key
_held = set()   # buttons down at the last poll
_pad = None     # the opened pygame controller or joystick
_named = False  # True: buttons are SDL names, False: raw numbers


def setup(const):
    """Opens the pad if constants.py sets gamepad_buttons. Called once by the Experiment."""
    global _pad, _named
    close()
    buttons = getattr(const, 'gamepad_buttons', None)
    if not buttons:
        return
    if len(buttons) != len(const.response_keys):
        raise ValueError(f"gamepad_buttons has {len(buttons)} buttons but response_keys has "
                         f"{len(const.response_keys)} keys: give one button per response key, in the same order.")

    # the PsychoPy window has the focus, not pygame: without this SDL ignores the pad
    os.environ['SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS'] = '1'
    import pygame
    from pygame._sdl2 import controller
    pygame.display.init()   # SDL's event loop, which updates the pad; opens no window
    pygame.joystick.init()
    controller.init()
    _named = all(isinstance(b, str) for b in buttons)
    names = [b.upper() for b in buttons] if _named else []
    unknown = [b for b in names if not hasattr(pygame, f'CONTROLLER_BUTTON_{b}')]
    if unknown:
        raise ValueError(f"Unknown gamepad button(s) {unknown}: use A, B, X, Y, START, BACK, "
                         f"LEFTSHOULDER, RIGHTSHOULDER, DPAD_UP, DPAD_DOWN, DPAD_LEFT or DPAD_RIGHT.")
    if pygame.joystick.get_count() == 0:
        raise RuntimeError("gamepad_buttons is set in constants.py but no gamepad is plugged in. "
                           "Plug it in, or remove gamepad_buttons to use the keyboard only.")

    if _named:
        if not controller.is_controller(0):
            raise RuntimeError(f"The gamepad '{pygame.joystick.Joystick(0).get_name()}' has no SDL button names. "
                               f"Give its raw button numbers in gamepad_buttons instead, e.g. [0, 1, 2, 3].")
        _pad = controller.Controller(0)
        ids = [getattr(pygame, f'CONTROLLER_BUTTON_{b}') for b in names]
    elif all(isinstance(b, int) for b in buttons):
        _pad = pygame.joystick.Joystick(0)
        ids = list(buttons)
    else:
        raise ValueError("gamepad_buttons must be all names (e.g. ['X', 'Y', 'A', 'B']) "
                         "or all numbers (e.g. [0, 1, 2, 3]), not a mix.")
    _keys.update(zip(ids, const.response_keys))
    print(f"Gamepad: {pygame.joystick.Joystick(0).get_name()}, "
          f"buttons {list(buttons)} = keys {list(const.response_keys)}")


def close():
    """Releases the pad."""
    global _pad
    if _pad is not None:
        _pad.quit()
    _pad = None
    _keys.clear()
    _held.clear()


def _poll():
    """Returns the response keys of the buttons pressed since the last poll."""
    import pygame
    pygame.event.pump()
    down = {b for b in _keys if _pad.get_button(b)}
    pressed = down - _held
    _held.clear()
    _held.update(down)
    return [_keys[b] for b in pressed]


def get_keys(keyList=None, timeStamped=False):
    """event.getKeys, plus the pad's presses as their response keys."""
    keys = event.getKeys(keyList=keyList, timeStamped=timeStamped)
    if _pad is None:
        return keys
    pad = [k for k in _poll() if keyList is None or k in keyList]
    if not pad:
        return keys
    if timeStamped:
        t = timeStamped.getTime() if hasattr(timeStamped, 'getTime') else core.getTime()
        return sorted(list(keys) + [(k, t) for k in pad], key=lambda kt: kt[1])
    return list(keys) + pad


def wait_keys(maxWait=float('inf'), keyList=None, timeStamped=False):
    """event.waitKeys, plus the pad's presses. Returns None when maxWait runs out."""
    if _pad is None:
        return event.waitKeys(maxWait=maxWait, keyList=keyList, timeStamped=timeStamped)
    end = core.getTime() + maxWait
    while True:
        keys = get_keys(keyList=keyList, timeStamped=timeStamped)
        if keys:
            return keys
        if core.getTime() >= end:
            return None
        core.wait(0.001, hogCPUperiod=0)


def is_held(key):
    """True while the pad button of this response key is held down."""
    if _pad is None:
        return False
    import pygame
    pygame.event.pump()
    return any(_pad.get_button(b) for b, k in _keys.items() if k == key)
