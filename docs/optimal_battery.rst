Optimal battery
========================

To construct a multi-task battery suitable for parcellating a specific brain structure (or ROI), we developed an algorithim that selects an on optimal battery based a large library of task activity patterns.

Load library
--------------
The first step is to download the task library currently available on Zenodo (doi:)

The download step fetches both a CIFTI (.dscalar) file that includes the data and an info file that describes the data.

The data file includes:
- data for the cortex in fs32k space
- data for the subcortex in MNI152NLin6Asym space in different subcortical atlases (e.g., cerebellum-left, cerebellum-right)

The info file includes information for each row in the data file:
- task_code: unique task code assigned for row
- cond_code: unique condition code assigned for row (some tasks have multiple conditions)
- full_code: task_code + cond_code
- dataset_sources: datasets used to get the average activity pattern for the row
- cond_id: unique id for each task condition
- total_subjects: total number of subjects in the dataset_sources


.. code-block:: python

    import MultiTaskBattery.battery as bat

    version = 'V1' 
    data,info = bat.fetch_task_library(atlas= atlas,version='V1') 

