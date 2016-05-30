# Automated Detection and Removal of Advertisements in Audio Clips
This is a project to detect the advt. segment in a given audio clip by extracting relevant features, build a learning model and remove the advt. segments in the test data(audio segments) using Machine Learning techniques.
* Extracted critical features from radio recordings to classify the input samples into
songs and advertisements using the Support Vector Machines.
* Trained the SVM classifier with 20000 samples to classify test samples into two
classes – Ad or Song. 
* Wrote a correction algorithm to improve accuracy.
___
## Dependencies:
* Ubuntu === 14.04 (This script may work on any Linux distro)
* Python === 2.7 
* Pandas === 0.13.1
* Scipy === 0.13.3
* NumPy === 1.8.2
* scikit-learn === 0.17.1
* pyAudioAnalysis === [Go to the repository] (https://github.com/tyiannak/pyAudioAnalysis/wiki)
* Octave === 3.8.1
___
## Steps to run this script:
1. Install all the dependencies if not installed already.
2. Add any test audio file (.wav format) in the _output/_ directory.
3. In terminal go into the home of this repo: `cd Ad-Detect`.
4. Run the feature-extraction script: `python feature-extract.py`.
5. Run the classification script: `python master-classifier.py`.
6. When prompt asks the user to input test file name of feature file generated at _test-data/_ directory. eg. `test6.2.dat`
7. When prompt comes back after execution, run: `cd output`
8. Change the file names in line no. 14, 25, 41 and 42 of wavgen.mat file. (It's formatted appropriately for test input.)
9. Execute the octave code: `octave wavgen.mat`
10. Filename specified on line 41 (residing in _output/_ directory) contains the desired song output.
11. Filename specified on line 42 (residing in _output/_ directory) contains the discarded ad segments.

___
## Contributors:
* Shreesha S (shreesha.suresh@gmail.com)
* Rakshith S Singh (r7.silverfox@gmail.com)
* S Rajesh Kumar (rajesh.kumar1995@gmail.com)
