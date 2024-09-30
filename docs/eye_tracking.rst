Eye tracking
============

Eye tracking is not required for a MTB experiment. If you do not want to use eyetracking, simply set the variable ``eye_tracker`` in your ``constants.py`` file to ```False``.
The MTB framework currently only work with EyeLink eyetrackers.

EyeLink setup
-------------

You will need to install EyeLink Developers Kit for your OS to be able to do eyetracking.
First sign up on SR Research Forum and then follow the instructions here:

https://www.sr-support.com/thread-13.html

Once you have installed EyeLink Developers Kit, you need to install the correct version of pylink. To do that, follow the instructions here:

https://www.sr-support.com/thread-48.html

** Do not pip install pylink. It will install another package with the same name!

Connecting to EyeLink
---------------------

The code uses pylink to connect to the eyetracker. For that, you need to use an ethernet cable. You can use a USB to Ethernet adapter if your laptop does not have an Ethernet port. Once the laptop is connected to the EyeLink Host PC, modify the Eyelink local network as follows:

** go to ethernet settings and find the EyeLink network.

** "Change adapter settings":
    ** Click on "Internet Protocol Version 4 (TCP/IPv4)" and click Properties

   ** Enter the following information:

   * IP address: '100.1.1.2'

   * Subnet: 255.255.255.0

   * Gateway: leave blank

