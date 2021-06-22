import pylink

# 1. open a connection to the tracker
# replace the IP address with None will open a simulated connection
eye_link = pylink.EyeLink(None)
# eye_link = pylink.EyeLink('100.1.1.1') # getting the IP address for the eyetracker