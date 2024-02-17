# Created 2023: Bassel Arafat, Jorn Diedrichsen, Ince Husain
from psychopy import core, event

class TTLClock:

    def __init__(self):
        self.clock      = core.Clock() #
        self.ttl_count      = 0    # the number of ttl pulses
        self.ttl_time       = 0    # time stamp of the last incoming ttl pulse
        self.ttl_button = 't'  # the button used for simulating a ttl pulse

    def reset(self, start_time=0):
        """ resets the clock and ttl-counter"""
        self.clock.reset(start_time)
        self.ttl_count = 0
        self.ttl_time = 0

    def wait_for_first_ttl(self, wait = True):
        """ This function waits for the first TTL and then resets the clock appropriately
        Args:
            wait (bool):
                if True, the function will wait for the first TTL
        """
        if wait:
            print('Waiting for first TTL...')
            while self.ttl_count == 0:
                self.update()
        self.clock.reset()
        self.ttl_time = 0

    def wait_until(self,time):
        """waits until the given  time since the beginning of the run
        Args:
            time (float):
                time in seconds
        Returns:
            real_time (float):
                time that the real start occurs
            start_ttl (int):
                Number of TRs recorded since the beginning of the run
            start_ttl_time (float):
                Time since the last TR  """
        while self.clock.getTime() < time:
            self.update()
        return self.clock.getTime(), self.ttl_count, self.clock.getTime()-self.ttl_time

    def get_time(self):
        return self.clock.getTime()

    def update(self):
        # get all the ttl pulses in the buffer
        keys = event.getKeys([self.ttl_button], timeStamped=self.clock)

        # checks if the pressed key is the key used as the ttl pulse
        for k in keys:
            self.ttl_count += 1 # each time a ttl button is pressed, ttl count increases
            self.ttl_time = k[1] # the time when the ttl button has been pressed
            print(f"TR count: {self.ttl_count} -    TR time: {self.ttl_time}", end = "\r")