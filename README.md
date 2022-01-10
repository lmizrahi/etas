# ETAS: Epidemic-Type Aftershock Sequence

### This code was written by Leila Mizrahi with help from Shyam Nandan for the following article:

Leila Mizrahi, Shyam Nandan, Stefan Wiemer 2021;<br/>The Effect of Declustering on the Size Distribution of Mainshocks.<br/>
_Seismological Research Letters_; doi: https://doi.org/10.1785/0220200231<br/>
<br/>


### The option for (space-time-)varying completeness magnitude in the parameter inversion is described in:

Leila Mizrahi, Shyam Nandan, Stefan Wiemer 2021;<br/> Embracing Data Incompleteness for Better Earthquake Forecasting. (Section 3.1)<br/>
_Journal of Geophysical Research: Solid Earth_; doi: https://doi.org/10.1029/2021JB022379<br/>
<br/>
<br/>

To cite the code, please cite the article(s).<br/>
For more documentation on the code, see the (electronic supplement of the) articles.<br/>
For Probabilistic, Epidemic-Type Aftershock Incomplenteness, see [PETAI](https://github.com/lmizrahi/petai).<br/>
In case of questions or comments, contact me: leila.mizrahi@sed.ethz.ch.
<br/>
<br/>
### Contents:
* inversion.py: Expectation Maximization algorithm to estimate ETAS parameters.
* simulation.py: Catalog simulation given ETAS parameters.
* mc_b_est.py: Coupled estimation of beta and Mc. including example estimation.
* invert_etas.py: run example parameter inversion.
* invert_etas_mc_var.py: run example parameter inversion with varying magnitude of completeness.
* simulate_catalog.py: simulate one example catalog.
* magnitudes.npy: magnitude sample for mc/beta estimation.
* california_shape.npy: polygon coordinates of California.
* synthetic_catalog.csv: synthetic catalog to be inverted by invert_etas.py
* synthetic_catalog_mc_var.csv: synthetic catalog to be inverted by invert_etas_mc_var.py


Previous dependencies on certain package versions should now be gone, but just in case, here is my pip freeze:<br/>

* geopandas==0.9.0
* numpy==1.19.1
* pandas==1.1.1
* pymap3d==2.4.3
* pynverse==0.1.4.4
* pyproj==3.0.1
* scikit-learn==0.23.2
* scipy==1.5.2
* Shapely==1.7.1
