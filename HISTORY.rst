History
=======

0.5
---

* First published version.

0.6
---

* More robust graph partition.
* Allow forcing feature genes.
* Rename "project" to "convey" to prepare for addition of atlas projection functionality.

0.7.0
-----

* Switch to new project template.
* Fix some edge cases in the pipeline.
* Switch to using ``psutil`` for detecting system resources.
* Fix binary wheel issues.
* Give up on using travis-ci.

0.8.0
-----

* Add inner fold factor computation for metacells quality control.
* Add deviant fold factor computation for metacells quality control.
* Add projection of query data onto an atlas.
* Self-adjusting pile sizes.
* Add convenience function for computing data for MCView.
* Better control over filtering using absolute fold factors.
* Fix edge case in computing noisy lonely genes.
* Additional outliers certificates.
* Stricter deviants detection policy

0.9.0
-----

* Improved and published projection algorithm.
* Restrict number of rare gene candidates.
* Tighter control over metacells size and internal quality.
* Improved divide-and-conquer strategy.
* Terminology changes:
  * Rename "forbidden" genes to "lateral" genes.
  * Add "noisy" genes ("lateral", but also ignored when computing deviant outlier cells).
  * Rename "feature" genes to "selected" genes.
  * Rename "significant" genes to "marker" genes.
