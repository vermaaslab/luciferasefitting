# The Basics
This repository contains the python scripts needed to reproduce data for an upcoming journal article.
The scripts themselves are broken into 3 parts, and rely on some common libraries that can be easily installed with `pip install`:
- [numpy](https://numpy.org/)
- [scipy](https://scipy.org/)
- [lmfit](https://lmfit.github.io/lmfit-py/)
- [matplotlib](https://matplotlib.org/)

# Quickrun
- Step 1: Run `python3 parsedata.py` to generate binary files from the ods files. These are faster to read in than the .ods files when fitting.
- Step 2: Run `python3 fitsequential.py`, which will sequentially go through fitting to first *just* the no-shell data, and then based on the parameters from that fit determine capped and uncapped fits. Note that this can take hours, so grab a cup of coffee, and come back to this much later.
The alternative is to run `python3 fitindependently.py`, which does the same thing. It does not feed in the fit parameters from the no-shell data as starting points, and is equally slow, mostly since `lmfit` is single-threaded.
This yields somewhat different results.
- Step 3: Run `python3 plotopt.py`, which will take the overall fit determined in step 2, and plot it against a single trace of the observed luminance, as well as a fit that is specific to that trace to see how much "better" a tailored fit can be rather than a model that shares parameters across multiple timeseries.
