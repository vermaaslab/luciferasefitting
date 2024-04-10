# The Basics

Step 1: Run `python3 parsedata.py` to generate binary files from the ods files. These are faster to read in than the .ods files when fitting.
Step 2: Run `python3 fitsequential.py`, which will sequentially go through fitting to first *just* the no-shell data, and then based on the parameters from that fit determine capped and uncapped fits.
The alternative is to run `python3 fitindependently.py`, which does the same thing, but does not feed in the fit parameters from the no-shell data as starting points.
This yields somewhat different results.
Step 3: Run `python3 plotopt.py`, which will take the overall fit determined in step 2, and plot it against a single trace of the observed luminance, as well as a fit that is specific to that trace to see how much "better" a tailored fit can be rather than a model that shares parameters across multiple timeseries.
