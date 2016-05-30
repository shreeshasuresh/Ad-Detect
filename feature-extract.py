######################################################################
#                                                                    #
#                   Title: feature-extract.py                        #  
#   Project: Automated Detection and Removal of Ads in Audio Clips   #
#     Contributors: Shreesha S, S Rajesh Kumar, Rakshith S Singh     #
#  Email: {shreesha.suresh,rajesh.kumar1995,r7.silverfox}@gmail.com  #
#                                                                    #
######################################################################

from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioFeatureExtraction

feature_file = open("test-data/test6.2.dat", "a")

# Read the Audio(.wav) file.
[Fs, x] = audioBasicIO.readAudioFile("output/test6.2.wav");

try:
  # Extract the Features using the pyAudioAnalysis Library.
  F = audioFeatureExtraction.stFeatureExtraction(x, Fs, 0.50*Fs, 0.25*Fs);
except:
  print "Sampling Frequency not matching. Aborted."

print "Feature file Generating..."
F = F.transpose()
# Write the column numbers(headers) of the features extracted to the file first.
feature_file.write(str(range(1, 35)).replace(" ", "")[1:-1])
feature_file.write("\n")

for i in F:
  # Print the features extracted to a file opened earlier.
  feature_file.write(str(i.tolist())[1:-1].replace("  ", ",").replace(" ", ""))
  feature_file.write("\n")

