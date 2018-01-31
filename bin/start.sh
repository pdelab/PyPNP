#! /bin/bash

clear && docker run -ti -v $(pwd):/home/fenics/PyPNP -w /home/fenics/PyPNP quay.io/fenicsproject/stable:current
