Topological Support
-------------------

The currently supported elements are as follows: hex, tet, pyramid,
wedge, quad, and tri. In general, hybrid meshes are fully supported for
the edge-based scheme. For CVFEM, hybrid meshes are also supported,
however, wedge and pyramid elements are not permitted at exposed open or
symmetry boundaries. The remedy to the CVFEM constraint is to simply
implement the exposed face gradient operators.
