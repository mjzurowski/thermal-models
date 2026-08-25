# Thermal models
General class that allows you to construct thermal models using TES'.

This is done using various classes/objects:
- Component: a material that has associated with it an initial temperature and heat capacity
- Connection: a connection between two materials defined with a conductance (G) measured at some temperature
- TES: a special case of a component that include Joule heating and noise modelling
- ThermalModel: a collection of components and connections (which must include at least a TES, bath, and absorber)

A particular system is set up by defining a list of components, setting up connections between them, then feeding them all to a ThermalModel.