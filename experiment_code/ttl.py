from psychopy import core, event

class TTL:
    
    def __init__(self):
        self.clock = None
        self.count = 0
        self.time = 0
        self.ttl_button = '5'

    def reset(self):
        self.clock = core.Clock()
        self.count = 0
        self.time = 0

    def check(self):
        # check time of incoming TRs and count number of TRs
        assert self.clock, "must set clock attribute"
        keys = event.getKeys([self.ttl_button], timeStamped=self.clock)
        if keys and keys[0][0] == self.ttl_button and (keys[0][1] - self.time) > 0.3:
            self.count += 1
            self.time = keys[0][1]
            print(f"new TR: {self.count}")

ttl = TTL()