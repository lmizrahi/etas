# ETAS: Epidemic-Type Aftershock Sequence

[![DOI](https://zenodo.org/badge/341629005.svg)](https://zenodo.org/badge/latestdoi/341629005)

### This code was written for the following articles:

Leila Mizrahi, Shyam Nandan, Stefan Wiemer 2021;<br/>The Effect of Declustering on the Size Distribution of Mainshocks.<br/>
_Seismological Research Letters_; doi: https://doi.org/10.1785/0220200231<br/>
<br/>

### The option for (space-time-)varying completeness magnitude in the parameter inversion is described in:

Leila Mizrahi, Shyam Nandan, Stefan Wiemer 2021;<br/> Embracing Data Incompleteness for Better Earthquake Forecasting. (Section 3.1)<br/>
_Journal of Geophysical Research: Solid Earth_; doi: https://doi.org/10.1029/2021JB022379<br/>
<br/>
<br/>

To cite the code, plase use its [DOI](https://zenodo.org/badge/latestdoi/341629005), and if appropriate, please cite the article(s).<br/>
For more documentation on the code, see the (electronic supplement of the) articles.<br/>
For Probabilistic, Epidemic-Type Aftershock Incomplenteness, see [PETAI](https://github.com/lmizrahi/petai).<br/>
In case of questions or comments, contact me: leila.mizrahi@sed.ethz.ch.
<br/>
<br/>

### Contents:

-   <code>runnable_code/</code> scripts to be run for parameter inversion or catalog simulation
    -   <code>ch_forecast.py</code> estimates ETAS parameters and creates 100 simulations using the Swiss catalog
    -   <code>estimate_mc.py</code> estimates constant completeness magnitude for a set of magnitudes
    -   <code>invert_etas.py</code> calibrates ETAS parameters based on an input catalog (option for varying mc, and option to fix certain parameters available)
    -   <code>simulate_catalog.py</code> simulates a synthetic catalog
    -   <code>simulate_catalog_continuation.py</code> simulates a continuation of a catalog, after the parameters have been inverted. if you run this _many times_, you get a forecast. **this only works if you run <code>invert_etas.py</code> beforehand.**
    -   <code>visualize_fit.py</code> makes plots which visualize the model fit to the data. **this only works if you run <code>invert_etas.py</code> beforehand, and set <code>store_pij = True</code>.**
-   <code>config/</code> configuration files for running the scripts in <code>runnable_code/</code>
    -   names should be self-explanatory.
-   <code>input_data/</code> input data to run example inversions and simulations
    -   <code>california_shape.npy</code> shape of polygon around California
    -   <code>ch_catalog.csv</code> Swiss catalog 1972 - 2021, used by <code>ch_forecast.py</code>    
    -   <code>ch_rect.npy</code> shape of rectangle around Switzerland
    -   <code>example_catalog.csv</code> to be inverted by <code>invert_etas.py</code>
    -   <code>example_catalog_mc_var.csv</code> to be inverted by <code>invert_etas.py</code> when varying mc mode is used
    -   <code>magnitudes.npy</code> example magnitudes for mc estimation
-   <code>output_data/</code> does not contain anything.
    -   your output goes here
-   <code>etas/ </code>
    -   here is where all the important functions algorithms are defined
