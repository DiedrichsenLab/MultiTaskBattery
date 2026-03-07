Optimal Battery
========================

To construct a multi-task battery suitable for parcellating a specific brain structure (or ROI within a brain structure), we developed an algorithm that selects an optimal battery based on a large library of task activity patterns.

Load library
--------------
The task library is hosted on Zenodo (DOI: `10.5281/zenodo.18793343 <https://doi.org/10.5281/zenodo.18793343>`_). The ``fetch_task_library`` function downloads and caches it locally.

The download fetches two files:

- **Data file** (.dscalar.nii): Group-averaged fMRI activity patterns in CIFTI format (cortex in fs32k space, subcortical structures in MNI152NLin6Asym space)
- **Info file** (.tsv): Metadata for each row in the data file, including:
  - ``task_code``: unique task code
  - ``cond_code``: condition code (some tasks have multiple conditions)
  - ``full_code``: task_code + cond_code (used to identify conditions)
  - ``source``: datasets used to compute the average activation pattern
  - ``total_subjects``: number of subjects the activation pattern is based on

.. code-block:: python

    import MultiTaskBattery.battery as bat

    data, info = bat.fetch_task_library(version='V1')

You can also filter by brain structure:

.. code-block:: python

    data, info = bat.fetch_task_library(version='V1', structures=['CEREBELLUM_LEFT', 'CEREBELLUM_RIGHT'])

The available structures in the V1 library are:

- ``CORTEX_LEFT``, ``CORTEX_RIGHT``
- ``ACCUMBENS_LEFT``, ``ACCUMBENS_RIGHT``
- ``AMYGDALA_LEFT``, ``AMYGDALA_RIGHT``
- ``BRAIN_STEM``
- ``CAUDATE_LEFT``, ``CAUDATE_RIGHT``
- ``CEREBELLUM_LEFT``, ``CEREBELLUM_RIGHT``
- ``DIENCEPHALON_VENTRAL_LEFT``, ``DIENCEPHALON_VENTRAL_RIGHT``
- ``HIPPOCAMPUS_LEFT``, ``HIPPOCAMPUS_RIGHT``
- ``PALLIDUM_LEFT``, ``PALLIDUM_RIGHT``
- ``PUTAMEN_LEFT``, ``PUTAMEN_RIGHT``
- ``THALAMUS_LEFT``, ``THALAMUS_RIGHT``

Evaluate a battery
-------------------
Given a set of task conditions, ``evaluate_battery`` computes the **Negative Inverse Trace (NIT)** of the task-by-task covariance matrix. Higher NIT means the tasks produce more separable brain activation patterns (higher potential for brain mapping).

.. code-block:: python

    # Define a battery using full_code values from the info file
    my_battery = ['tom_task', 'movie_romance', 'fingr_sequence',
                  'nbckverb_2bcknrep', 'vissearch_small', 'rest_task',
                  'vrbgen_generate', 'actobserv_knot']

    nit = bat.evaluate_battery(data, info, my_battery)
    print(f"NIT score: {nit:.4f}")

Find optimal batteries
-----------------------
``get_top_batteries`` performs a random search over many task combinations to find batteries with the highest NIT scores.

.. code-block:: python

    # Search 100,000 random 8-task batteries, keep top 10
    top = bat.get_top_batteries(data, info,
                                n_samples=100000,
                                battery_size=8,
                                n_top_batteries=10)
    print(top)

You can also force certain tasks to always be included:

.. code-block:: python

    # Always include rest, search for best remaining 7 tasks
    top = bat.get_top_batteries(data, info,
                                n_samples=100000,
                                battery_size=8,
                                forced_tasks=['rest_task'])

Load from local files
----------------------
If you already have the library files downloaded, use ``load_library`` directly:

.. code-block:: python

    data, info = bat.load_library('path/to/data.dscalar.nii', 'path/to/info.tsv')
