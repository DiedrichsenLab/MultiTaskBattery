.. MTB_experiment index

Multi Task Battery experiments
==============================

The MTB repository provides a Python-based framework (built on PsychoPy) to run fMRI experiments that contain many different tasks. Such task batteries are useful for individual localization of function, and to understand the relationship between different tasks. The main advance from traditional multi-task approaches (e.g., the HCP task-based dataset) is that we run many different task within a single imaging run, rather than acquire different task in separate imaging runs. While participants have to switch between task during scanning, this approach optimizes the statistical power to relate the activity patterns across many different tasks within the same individual. We have used this approach with great success in several studies, starting with the Multi-domain Task Battery (MDTB) described in King et al. (2019).

The repository implements a number of validated tasks, from which the user can easily assemble a new multi-task battery that fits their specific research needs. The code is object-oriented and can easily be expanded to implement new tasks.

The main benefit of using this framework is that you can include standardized tasks and stimuli in your battery that are also used in other studies. With our collaborators, we are aiming to build a library of datasets using this approach. Having multiple anchor-tasks across experiments (rather than just rest), allows a rich set of analyzes to answer fundamental questions of human mental function. If you are interested in contributing to this library, please contact as us at jdiedric@uwo.ca.

Licence and Acknowledgements
----------------------------
The software written and maintained by members of the Diedrichsen lab and collaborators (Bassel Arafat, Ince Husain, Caroline Nettekoven, Ladan Shahshahani, Suzanne Witt, Maedbh King, Jorn Diedrichsen). We are are still working a formal paper on the framework - until then the software should be cited as a github repository. For description of the underlying approach please cite King et al. (2019). The software distributed under the MIT License: The software is provided as is, without any warranty.

Documentation
-------------
.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation.rst
   overview.rst
   eye_tracking.rst
   MRI_instructions.rst
   task_descriptions.rst
   building_experiment.rst
   creating_tasks.rst
   reference.rst

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
