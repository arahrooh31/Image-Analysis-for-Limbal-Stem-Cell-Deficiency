The Excel sheet contains information about which images have nerves for each patient. This is the "gold standard" which I manually chose. It probably isn't perfect but it's better than nothing. This must be in the same file that the Jupyter notebook file is in because the notebook uses the Excel sheet for evaluation.

The Original Images file analyzes the unprocessed original images. The methods in it perform similar functions to the TrainingTest file, but they are modified to work with the original images. Comments have been left in the file.

The TrainingTest file analyzes the processed images (blur and Canny edge detection). The meothds in it perform similar functions to the Original Images file, but they are modified to work with the processed images. Comments have been left in the file. 

Dylan Steinecke
dylan.steinecke@gmail.com

Al Rahrooh
arahrooh@g.ucla.edu


Limbal stem cell deficiency is an eye disease characterized by loss of limbal stem cells and partially diagnosed by analyzed images of the nerves in the limbus. Significant human effort is required to select and grade the nerve-containing images. Here, we present work on the first part of an automated process to select, register, and grade the images. For identifying nerve-containing images, a framework for exploring, analyzing, and evaluating the performance of classification algorithms and features for multiple patients and their image sequences is discussed, along with preliminary results (F1 score = 0.463 and 0.606). For registering confocal eye images, an automated pipeline was successful in transforming multiple images in sequence for a single output image. Though, incomplete, necessary progress has been made towards developing the pipeline using classical non-ML image analysis algorithms and image registration techniques. These are the first steps towards improving upon existing diagnostic approaches for LCSD diagnoses.
