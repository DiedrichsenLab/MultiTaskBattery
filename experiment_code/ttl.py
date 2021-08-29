from psychopy import core, event

class TTL:
    
    def __init__(self):
        self.clock      = None #
        self.count      = 0    # the number of ttl pulses 
        self.time       = 0    # the time of the incoming ttl pulse
        self.ttl_button = '5'  # the button used for simulating a ttl pulse

    def reset(self):
        self.clock = core.Clock()
        self.count = 0
        self.time = 0

    def check(self):
        # check time of incoming TRs and count number of TRs
        assert self.clock, "must set clock attribute"
        
        # event.getKeys returns a list of tuples. 
        # how many tuples? Depends on how many keys have been pressed
        # in the tuple, the first element is always the pressed key and the second element is the time of press
        keys = event.getKeys([self.ttl_button], timeStamped=self.clock)

        # checks if the pressed key is the key used as the ttl pulse 
        if keys and keys[0][0] == self.ttl_button and (keys[0][1] - self.time) > 0.3:
            self.count += 1 # each time a ttl button is pressed, ttl count increases
            self.time = keys[0][1] # the time when the ttl button has been pressed
            print(f"TR count: {self.count} -    TR time: {self.time}", end = "\r")
ttl = TTL()