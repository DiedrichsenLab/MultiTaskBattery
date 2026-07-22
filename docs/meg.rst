MEG / EEG compatibility
=======================

.. warning::
   This feature is **experimental and has not yet been tested on MEG/EEG
   hardware.** The code paths described below run and log data, but the timing
   and photodiode alignment have not been validated on a real recording system.
   Verify everything with a photodiode on your own rig before collecting data.

Unlike fMRI — where events are synced to the scanner TR at second-level
resolution — MEG and EEG require knowing the *true* stimulus-onset time to the
millisecond, because brain activity is averaged (epoched) relative to that
onset. The complication is that the screen does not physically update at the
instant the code calls ``flip()``: the OS compositor and GPU add a delay of
1–3 frames between the ``flip()`` call returning and the photons reaching the
participant. The standard, ground-truth solution is a **photodiode**: a small
square in a screen corner is flashed on stimulus onset, and a light sensor taped
over it records the true onset directly into the MEG/EEG file.

MTB implements this photodiode marker plus screen-flip-time logging.

Enabling the photodiode marker
------------------------------

Set the following in your experiment's ``constants.py``::

   photodiode = True

Optionally, move or resize the photodiode square (``norm`` units, bottom-right
corner by default) by adding these keys to the ``screen`` dictionary::

   screen = {
       ...,
       'photodiode_pos':  (0.9, -0.9),   # (x, y), norm units
       'photodiode_size': 0.2,           # width = height, norm units
   }

When ``photodiode = False`` (the default) the MTB does not show the square as done for fMRI experiments.

What it does
------------

When enabled, on the **first flip of each trial** (the stimulus onset) MTB:

* draws the photodiode square white for that flip, so a photodiode over the
  square records the true onset in the MEG/EEG recording, and
* logs the flip time to a new ``flip_time`` column in the task ``.tsv``. This
  time is on the same clock as the TR counts and reaction times, so it can be
  cross-referenced with the photodiode pulse and used to sort trials by
  condition.

Because every event carries its own photodiode mark, clock drift between the
stimulus computer and the MEG system is handled automatically.

Marking the correct flip
------------------------

By default the **first flip of a trial** is treated as the stimulus onset. This
is correct for most tasks, but some tasks flip a fixation or blank screen before
the stimulus. You must confirm — with the photodiode — that the flash lands on
the actual stimulus for each task you use.

To control marking explicitly inside a task's ``run_trial``, pass ``marker`` to
``self.flip()``:

* ``self.flip(marker=True)``  — force this flip to be the onset marker.
* ``self.flip(marker=False)`` — force a plain flip (use on a pre-stimulus
  fixation flip so the marker lands on the real stimulus instead).
* ``self.flip()`` (default)   — auto-mark the first flip of the trial.

Before you record
-----------------

* A physical photodiode must be placed over the square and wired into the
  MEG/EEG system — the square alone does nothing without it.
* Verify per task that the square flashes on the true stimulus onset.
* Measure the flip-to-light delay with the photodiode and check whether it is
  fixed or variable on your rig.
