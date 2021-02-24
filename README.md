# etas

This code was written by Leila Mizrahi with help from Shyam Nandan for the following article:

Leila Mizrahi, Shyam Nandan, Stefan Wiemer; The Effect of Declustering on the Size Distribution of Mainshocks.
Seismological Research Letters 2021; doi: https://doi.org/10.1785/0220200231

To cite the code, please cite the article.
For more documentation on the code, see the (electronic supplement of the) article.
In case of questions or comments, contact me: leila.mizrahi@sed.ethz.ch


Contents:
    inversion.py: Expectation Maximization algorithm to estimate ETAS parameters
    simulation.py: Catalog simulation given ETAS parameters
    invert_etas.py: run example parameter inversion
    simulate_catalog.py: simulate one example catalog.


Parts of the code, especially regarding regional polygons, only work with specific versions of the packages.
Here is my pip freeze:

Fiona==1.8.13.post1
geopandas==0.6.1
numpy==1.19.1
pandas==1.1.1
pymap3d==2.4.3
pynverse==0.1.4.4
pyproj==1.9.6
scikit-learn==0.23.2
scipy==1.5.2
Shapely==1.7.1