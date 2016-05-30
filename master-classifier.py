######################################################################
#                                                                    #
#                  Title: master-classifier.py                       #      
#   Project: Automated Detection and Removal of Ads in Audio Clips   #
#     Contributors: Shreesha S, S Rajesh Kumar, Rakshith S Singh     #
#  Email: {shreesha.suresh,rajesh.kumar1995,r7.silverfox}@gmail.com  #
#                                                                    #
######################################################################

import pandas as pd
import scipy
import numpy as np

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

print "Reading Data....."

# Read the .csv files into Pandas DataFrames
# The Ad Training dataset:
ad0data = pd.read_csv("ad-train/ad0.dat")
ad1data = pd.read_csv("ad-train/ad1.dat")
ad2data = pd.read_csv("ad-train/ad2.dat")
ad3data = pd.read_csv("ad-train/ad3.dat")
ad4data = pd.read_csv("ad-train/ad4.dat")
ad5data = pd.read_csv("ad-train/ad5.dat")
ad7data = pd.read_csv("ad-train/ad7.dat")
ad8data = pd.read_csv("ad-train/ad8.dat")
ad9data = pd.read_csv("ad-train/ad9.dat")
ad10data = pd.read_csv("ad-train/ad10.dat")
ad11data = pd.read_csv("ad-train/ad11.dat")
ad12data = pd.read_csv("ad-train/ad12.dat")
ad13data = pd.read_csv("ad-train/ad13.dat")
ad14data = pd.read_csv("ad-train/test1opads.dat")

# The Song Training dataset:
song0data = pd.read_csv("song-train/song0.dat")
song1data = pd.read_csv("song-train/song1.dat")
song2data = pd.read_csv("song-train/song2.dat")
song3data = pd.read_csv("song-train/song3.dat")
song4data = pd.read_csv("song-train/song4.dat")
song5data = pd.read_csv("song-train/song5.dat")
song6data = pd.read_csv("song-train/song6.dat")
song7data = pd.read_csv("song-train/song7.dat")
song8data = pd.read_csv("song-train/song8.dat")
song9data = pd.read_csv("song-train/song9.dat")
song10data = pd.read_csv("song-train/song10.dat")
song11data = pd.read_csv("song-train/song11.dat")
song12data = pd.read_csv("song-train/song12.dat")
song13data = pd.read_csv("song-train/song13.dat")

# Aggregate all the Ad training dataset and compute its total size.
ad_train = ad0data.append([ad1data,ad2data,ad3data,ad4data,ad5data,ad7data,ad8data,
              ad9data,ad10data,ad11data,ad12data,ad13data,ad14data])
len_ad_train = len(ad_train)
print "Length of ad-training dataset = " + str(len_ad_train)

# Append all the song datasets to create a huge train dataset containing both-song and ad samples.
train = ad_train.append([song0data,song1data,song2data,song3data,song4data,song5data,
          song6data,song7data,song8data,song9data,song10data,song11data,song12data,song13data])
print "Length of song-training dataset = " + str(len(train)-len_ad_train)


# TrainingClass = Songs are classified as 1 and ads as 0
train_class = pd.DataFrame([0]*len_ad_train + [1]*(len(train)-len_ad_train))

# Change train class to 1D Array (Transpose). 
train_class = train_class.as_matrix().ravel()

print "Training Data...."
# Use SVM with linear kernel class.
classifier = SVC(kernel="linear")

# Train the data.
classifier.fit(train, train_class)

"""
  # Other classifiers: 
  1. clf = RandomForestClassifier(n_estimators=10)
  2. clf = LogisticRegression(C=1e5)
  
  clf = clf.fit(train, train_class)
  class_ans = clf.predict(test1)

"""

# Predict the test data.....
# Entering a while loop to take infinite number of input files and predict the classes.
while(1):
  try:
    print "Enter test file name. Or type exit to return"
    test_name = raw_input()
    if test_name == "exit":
      break
    else:
      # Read the test feature file and store in Pandas DataFrame
      print "Reading test data.."
      test_name = "test-data/"+test_name
      test = pd.read_csv(test_name)
      
      # Predict the class label for each window.
      class_ans = list(classifier.predict(test))
      
      # Store the labels in a new file.
      output_name = "output/svm_"+test_name.split("/")[1][:-4]+"_class.dat"
      output = open(output_name, "w")
      """
      -> Correction Algorithm:
           The objective of Correction algorithm is to improve accuracy by reversing 
           the prediction (ie. 0 to 1 OR 1 to 0) if, in a window size of 2 seconds,
           the majority labels are of the other binary value.
         Example:
           Let a random 2 seconds(8 samples with a window size of 250ms each) sample
           of initial prediction be:
              1, 0, 1, 1, 1, 0, 1, 1
           Now, it is obvious for humans to decide that there cannot be an ad between
           two song samples measuring a lenth of 250ms. Hence we re-label it to 1.
              Final : 1, 1, 1, 1, 1, 1, 1, 1
           Thus, we achieve higher accuracy by avoiding misclassification. 
           
           The same is applicable when number of 0s are more than or equal to number of 1s.
      """
      print "Running correction algorithm."
      len_class = len(class_ans)
      # counter is a 2 second frame indicator
      counter = 0
      
      while counter < len_class:
        # zcount is the variable that stores the number of zeroes in current frame
        zcount = class_ans[counter:counter+8].count(0)
        next = counter+8
        # 8 here signifies the window size for considering 2 seconds. Each value ...
        # ... here measures 250ms on original radio stream.
        
        if zcount >= class_ans[counter:counter+8].count(1):
          # Here, no. of zeros are more than (or equal to) the no. of ones. Hence, ...
          # ... convert all the ones to zeroes in the current frame length of 2 seconds.
          for k in range(counter, counter+8):
            try:
              class_ans[k] = 0
            except:
              pass
        else:
          # Here, no. of zeros are less than the no. of ones. Hence, convert ...
          # ... all the zeroes to ones in the current frame length of 2 seconds.
          for k in range(counter, counter+8):
            try:
              class_ans[k] = 1
            except:
              pass
        # Consider the next 8 labels.
        counter = next

      # Beautifying the output at writing to the output file. 
      output.write(str(class_ans).replace(" ", "")[1:-1])
      output.close()
      
      # Print the results(Count of ad samples and song samples) to the console.
      print "SVM Kernel = linear | "+test_name.split("/")[1]+" ==="
      print "Ad = ",
      print list(class_ans).count(0)
      print "Song = ",
      print list(class_ans).count(1)
  except:
    print "Error: Try again with correct input."
    pass
