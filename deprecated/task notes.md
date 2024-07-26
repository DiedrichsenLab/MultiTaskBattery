task notes

--- fixes:--- 

nback:
- cross needs to be a lot thicker! - 
- feedback is barely visible

oddball:
- has no feedback?

tom:
- fix orphaning in text

auditory narrative:
- the beginning of this narrative was degraded for me and then kept repeating the word 'had no, had no, had no, had no'. This must either be something wrong with my computer or the code. Can you try it out on yours? I don't have access to another device.

intact passage:
- played nothing for me and only started the passage after 5 seconds of silence - i'll try again but this might be a bug

tongue:
- circle needs to be thicker

flexion_extension:
- i think instructions should read 'flex' and 'extend'

--- things to discuss ---

action_observation:
- Abyss: video shows no actual knot tying and is likely an outtake. 
- Many videos cut off before the knot is actually tied. We should only be using the good ones for the task.
  
  - Good videos:
    - Adage
    - Brigand
    - Brocade
    - Casement
    - Cornice
    - Flora
    - Frontage
    - Gadfly
    - Garret
    - Mutton
    - Placard
    - Purser
  
  - Medium videos (only to be used if more than the good videos are needed):
    - (Baron)
    - (Belfry)
    - (Simper)


verb generation:
- generate should be indicated by something other than a word
- instructions need to be changed


--- fixes for all tasks: ---
fix in randomisation: auditory narrative should never be followed or preceded by degraded passage or intact passage

For running several runs it looks like there’s a bug:
  - the same stimuli are used for all runs (especially noticeable in intact_passage, auditory_narrative,   degraded_passage, romance_movie, sentence_reading, etc.). The issue comes from the dialogue box: The default input for the run counts up but not the default input for the run file. That should be set to run_02.tsv and so on (edited) 

- I think for all tasks where one word is presented at a time, the font should be made a lot bigger. This has to be tried out in the scanner, but my feeling is that the font is way too small