Eye tracking
============

.. warning::
   Eye tracking support is experimental and has not been tested with the current version of MTB. Please test it on your setup before using it in a study.

Eye tracking is not required for a MTB experiment. If you do not want to use eyetracking, simply set the variable ``eye_tracker`` in your ``constants.py`` file to ``False``.
The MTB framework currently only work with EyeLink eyetrackers.

EyeLink setup
-------------

You will need to install the EyeLink Developers Kit for your OS and the matching version of pylink. Both are available from SR Research (registration required); follow their installation instructions.

.. warning::
   Do not ``pip install pylink`` — it installs a different, unrelated package with the same name.

Connecting to EyeLink
---------------------

The code uses pylink to connect to the eyetracker. For that, you need to use an ethernet cable. You can use a USB to Ethernet adapter if your laptop does not have an Ethernet port. Once the laptop is connected to the EyeLink Host PC, modify the Eyelink local network as follows:

* Go to ethernet settings and find the EyeLink network.
* Open **"Change adapter settings"**:

  * Click on "Internet Protocol Version 4 (TCP/IPv4)" and click **Properties**.
  * Enter the following information:

    * IP address: ``100.1.1.2``
    * Subnet: ``255.255.255.0``
    * Gateway: leave blank

