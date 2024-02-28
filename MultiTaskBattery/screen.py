# created 2023: Bassel Arafat, Jorn Diedrichsen, Ince Husain
from psychopy import visual, monitors

class Screen:
    def __init__(self, const):
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