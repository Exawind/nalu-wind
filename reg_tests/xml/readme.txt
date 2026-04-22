
We have added MueLu milestone.xml files that demonstrate the best performance
(strong scaling) and lowest run times on Peregrine and Cori for KynemaUGF cases:

ABL: milestone_ABL.xml (copy into milestone.xml read as default input file)

The smoother is ILUT (threshold ILU) and agglomeration employs dropping tol=0.005
A local sub-domain additive Schwarz algorithm with 0-overlap is applied.
Note that a distance Laplacian (agglomeration: drop scheme) is not employed
Also note max levels set to 6 and implicit transpose options.

McAlister: milestone_McAlister.xml 

The smoother is ILUT (threshold ILU) and agglomeration employs dropping tol=0.005
A local sub-domain additive Schwarz algorithm with 0-overlap is applied.
Note that a distance Laplacian (agglomeration: drop scheme) IS employed for high aspect ratio.
Also note max levels set to 6 and implicit transpose options.


