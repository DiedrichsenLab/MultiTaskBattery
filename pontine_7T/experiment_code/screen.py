from psychopy import visual, core, logging, event, data, monitors

class Screen: 

    def __init__(self, fullscr = True, screen_number = 0):
        self.fullscr  = fullscr
        self.units    = 'deg'
        self.color    = '#808080'
        self.size     = [1024, 768] #[800, 800] #[1440, 900]
        self.distance = 57.0
        self.width    = 30.0
        self.allowGUI = True
        self.screen_number = screen_number
        self.window   = self._create_window()
        self.monitor  = self._create_monitor()
        
    def _create_window(self): 
        return visual.Window(size = self.size, 
                             screen = self.screen_number,
                             monitor = self._create_monitor(),
                             fullscr = self.fullscr,
                             units = self.units,
                             color = self.color, 
                             allowGUI = self.allowGUI) 

    def _create_monitor(self):
        # set up monitor
        monitor = monitors.Monitor(
       "stimulus",
        distance = self.distance,
        width = self.width,
        )
        monitor.setSizePix(self.size) # screen size (not window!) look in display prefs 
        monitor.saveMon()
        return monitor

    def fixation_cross(self):
        #fixation cross
        fixation = visual.ShapeStim(self.window, 
            vertices=((0, -0.05), (0, 0.05), (0,0), (-0.03,0), (0.03, 0)),
            lineWidth=5,
            closeShape=False,
            lineColor='white',
            units='norm'
        )
        fixation.draw()
        self.window.flip()