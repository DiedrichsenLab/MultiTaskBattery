# created 2023: Bassel Arafat, Jorn Diedrichsen, Ince Husain
from psychopy import visual, monitors

class Screen:
    def __init__(self, const):
        """    A class to create a screen for the experiment

               Args: 
                    const (module):
                        local constants.py module (see example_experiment/constants.py) as example
                Returns:
                    self (object):
                        an instance of the Screen class
        """
        self.fullscr  = const['fullscr']
        self.units    = 'deg'
        self.color    = '#808080'
        self.size     = const['size'] #[800, 800] #[1440, 900]
        self.distance = 57.0
        self.width    = 30.0
        self.allowGUI = True
        self.screen_number = const['number']
        self.monitor  = monitors.Monitor(
                "stimulus",
                distance = self.distance,
                width = self.width,
        )
        self.monitor.setSizePix(self.size) # screen size (not window!) look in display prefs
        self.monitor.saveMon()

        self.window   = visual.Window(size = self.size,
                             screen = self.screen_number,
                             monitor = self.monitor,
                             fullscr = self.fullscr,
                             units = self.units,
                             color = self.color,
                             allowGUI = self.allowGUI, allowStencil=True)

    def fixation_cross(self, color='white', flip=True):
        """    A method to draw a fixation cross on the screen
        
               Args: 
                    color (str):
                        color of the fixation cross
                    flip (bool):
                        whether to flip the screen after drawing the fixation cross
                Returns:
                    None
        """
        # Draw the fixation cross
        fixation = visual.ShapeStim(self.window,
            vertices=((0, -0.06), (0, 0.06), (0,0), (-0.04,0), (0.04, 0)),
            lineWidth=20,
            closeShape=False,
            lineColor=color,
            units='norm'
        )
        fixation.draw()
        
        # Flip the screen only if flip is True
        if flip:
            self.window.flip()

    def check_mark(self, color='green', flip=True):
        """    A method to draw a check mark on the screen
        
               Args: 
                    color (str):
                        color of the check mark
                    flip (bool):
                        whether to flip the screen after drawing the check mark
                Returns:
                    None
        """
        # Draw the check mark
        check = visual.ShapeStim(self.window,
            vertices=((-0.04,0), (0, -0.06), (0.04, 0.06)),
            lineWidth=20,
            closeShape=False,
            lineColor=color,
            units='norm'
        )
        check.draw()

        # Flip the screen only if flip is True
        if flip:
            self.window.flip()

    def error_cross(self, color='red', flip=True):
        """    A method to draw an error cross on the screen
        
               Args: 
                    color (str):
                        color of the error cross
                    flip (bool):
                        whether to flip the screen after drawing the error cross
                Returns:
                    None
        """
        # Draw the fixation cross
        cross_leg1 = visual.ShapeStim(self.window,
            vertices=((-0.04, -0.06), (0.04, 0.06)),
            lineWidth=20,
            closeShape=False,
            lineColor=color,
            units='norm'
        )
        cross_leg1.draw()

        cross_leg2 = visual.ShapeStim(self.window,
            vertices=((-0.04, 0.06), (0.04, -0.06)),
            lineWidth=20,
            closeShape=False,
            lineColor=color,
            units='norm'
        )
        cross_leg2.draw()

        # Flip the screen only if flip is True
        if flip:
            self.window.flip()