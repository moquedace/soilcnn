# The tests R CMD check runs -- CRAN's too: a fast subset of what the package
# promises, on synthetic numbers and on the example landscape, and none of it
# trains a network or needs torch's backend. The full suite -- networks
# trained, maps predicted, the package built and installed, every example run
# -- is tests/run_all.R, which the tarball leaves out and which runs here
# before every commit.
library(testthat)
library(soilcnn)

test_check("soilcnn")
