Code from Joost van de Weijer and Arjan Gijsenij, downloaded from
http://colorconstancy.com/

Two modifications:
- added prefix jvdw_ to avoid name space clashes
- Slight extension to work with superpixels:
  * ggw_per_superpixel.m
    computes generalized gray world per superpixel
  * apply_ggw_on_barnard.m
	demo application to look at the output for superpixelized version.
	Can also be used to verify the results in the TIP 2007 paper: edge-based
	color constancy.
