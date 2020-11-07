# Pitch Scapes

[![PyPI version](https://badge.fury.io/py/pitchscapes.svg)](https://badge.fury.io/py/pitchscapes)
![build](https://github.com/robert-lieck/pitchscapes/workflows/build/badge.svg)
![tests](https://github.com/robert-lieck/pitchscapes/workflows/tests/badge.svg)
[![codecov](https://codecov.io/gh/robert-lieck/pitchscapes/branch/master/graph/badge.svg?token=UVBQF5J5HG)](undefined)

[![DOI](https://zenodo.org/badge/282043116.svg)](https://zenodo.org/badge/latestdoi/282043116)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

<img src="./doc/figure_1.png" alt="Figure 1" width=35%>                                 <img src="./doc/figure_2.png" alt="Figure 2" width=25%>

Python library to compute pitch scapes for music analysis. The implemented methods are described in (please cite when using this library for publications, BibTeX below): Lieck R, Rohrmeier M (2020) [**Modelling Hierarchical Key Structure with Pitch Scapes**](http://robert-lieck.com/literature/pdfs/8K8MJHK9/Lieck_and_Rohrmeier_-_2020_-_Modelling_Hierarchical_Key_Structure_With_Pitch_Sc.pdf). In: *Proceedings of the 21st International Society for Music Information Retrieval Conference*. Montréal, Canada
```
@inproceedings{lieckModellingHierarchicalKey2020,
  title = {Modelling Hierarchical Key Structure with Pitch Scapes},
  booktitle = {Proceedings of the 21st International Society for Music Information Retrieval Conference},
  author = {Lieck, Robert and Rohrmeier, Martin},
  year = {2020},
  location = {Montréal, Canada},
  eventtitle = {21st International Society for Music Information Retrieval Conference}
}
```

## Getting Started

Please have a look at the [Tutorial](./Tutorial.ipynb), which guides you through all the basic functionality of reading data, plotting pitch scapes and training the unsupervised clustering model.

**Note:** I tried to make the code accessible and provide some convenience functions for getting started smoothly. But there is still lots of room for better documentation and testing. Please contact me if you have any questions, are missing anything or encounter bugs!
