# SBF_interaction

## Overview
SBF_interaction is an additive interaction model based on a new formulation.

## Main functions
save_kde: Computation and storage of 3D and 4D kernel density estimates (KDEs) \
fit: Fits the additive interaction model. \
predict: Predicts responses based on the fitted model. 

## Example

import SBF_interaction

path = './home/' \
ngrid = 101  # A small ngrid value is recommended for efficient storage 

h = h_RT(X, X.shape[1], ngrid) \
model = BSBF_inter(X, Y, h, ngrid=ngrid) \
model.save_kde()  # Not necessary when X.shape[1] == 2 \
model.fit() 

## License
This project is licensed under the MIT License.
