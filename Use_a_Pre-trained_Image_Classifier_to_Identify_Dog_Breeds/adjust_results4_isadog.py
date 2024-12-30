#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */AIPND-revision/intropyproject-classify-pet-images/adjust_results4_isadog.py
#                                                                             
# PROGRAMMER: Ingyin San
# DATE CREATED: 10-28-2024                                
# REVISED DATE: 11-1-2024
# PURPOSE: Create a function adjust_results4_isadog that adjusts the results 
#          dictionary to indicate whether or not the pet image label is of-a-dog, 
#          and to indicate whether or not the classifier image label is of-a-dog.
#          All dog labels from both the pet images and the classifier function
#          will be found in the dognames.txt file. We recommend reading all the
#          dog names in dognames.txt into a dictionary where the 'key' is the 
#          dog name (from dognames.txt) and the 'value' is one. If a label is 
#          found to exist within this dictionary of dog names then the label 
#          is of-a-dog, otherwise the label isn't of a dog. Alternatively one 
#          could also read all the dog names into a list and then if the label
#          is found to exist within this list - the label is of-a-dog, otherwise
#          the label isn't of a dog. 
#         This function inputs:
#            -The results dictionary as results_dic within adjust_results4_isadog 
#             function and results for the function call within main.
#            -The text file with dog names as dogfile within adjust_results4_isadog
#             function and in_arg.dogfile for the function call within main. 
#           This function uses the extend function to add items to the list 
#           that's the 'value' of the results dictionary. You will be adding the
#           whether or not the pet image label is of-a-dog as the item at index
#           3 of the list and whether or not the classifier label is of-a-dog as
#           the item at index 4 of the list. Note we recommend setting the values
#           at indices 3 & 4 to 1 when the label is of-a-dog and to 0 when the 
#           label isn't a dog.
#
##
# TODO 4: Define adjust_results4_isadog function below, specifically replace the None
#       below by the function definition of the adjust_results4_isadog function. 
#       Notice that this function doesn't return anything because the 
#       results_dic dictionary that is passed into the function is a mutable 
#       data type so no return is needed.
# 
def adjust_results4_isadog(results_dic, dogfile):
    """
    Adjusts the results dictionary to determine if classifier correctly 
    classified images 'as a dog' or 'not a dog' especially when not a match. 
    Demonstrates if model architecture correctly classifies dog images even if
    it gets dog breed wrong (not a match).
    Parameters:
      results_dic - Dictionary with 'key' as image filename and 'value' as a 
                    List. Where the list will contain the following items: 
                  index 0 = pet image label (string)
                  index 1 = classifier label (string)
                  index 2 = 1/0 (int)  where 1 = match between pet image
                    and classifer labels and 0 = no match between labels
                ------ where index 3 & index 4 are added by this function -----
                 NEW - index 3 = 1/0 (int)  where 1 = pet image 'is-a' dog and 
                            0 = pet Image 'is-NOT-a' dog. 
                 NEW - index 4 = 1/0 (int)  where 1 = Classifier classifies image 
                            'as-a' dog and 0 = Classifier classifies image  
                            'as-NOT-a' dog.
     dogfile - A text file that contains names of all dogs from the classifier
               function and dog names from the pet image files. This file has 
               one dog name per line dog names are all in lowercase with 
               spaces separating the distinct words of the dog name. Dog names
               from the classifier function can be a string of dog names separated
               by commas when a particular breed of dog has multiple dog names 
               associated with that breed (ex. maltese dog, maltese terrier, 
               maltese) (string - indicates text file's filename)
    Returns:
           None - results_dic is mutable data type so no return needed.
    """       
    # dictionary to match results_dic labels with real answers and classifier's answers    
    dognames_dic = dict()
    
    # read lines from the dog names file, closing after the process
    with open(dogfile, 'r') as name_file:

      # read the first line
      line = name_file.readline()

      # continue reading until the end of the file
      while line != "":

        # strip newline character at the end of the line
        name = line.rstrip("\n")

        # print a warning message if the dog name is already in the dictionary
        if name in dognames_dic:
          print(f"Warning: The dog name, {name}, already exists in the dictionary.")

        # otherwise, adds the dog name to the dictionary with a value of 1
        else:
          dognames_dic[name] = 1

        # move to the next line
        line = name_file.readline()

    # iterate through the values in the results_dic
    for labels in results_dic.values():

      # pet image label is in the dognames_dic   
      if (labels[0] in dognames_dic):

        # if classifier label is in the dognames_dic,
        # set both pet image label and classifier label to "is a dog" (1)
        if (labels[1] in dognames_dic):
          labels.extend([1, 1])

        # otherwise, set pet image label to "is a dog" (1) and classifier label to "not a dog" (0)
        else:
          labels.extend([1, 0])

      # pet image label not in the dognames_dic
      else:

        # if classifier label is in the dognames_dic,
        # set pet image label to "not a dog" (0) and classifier label to "is a dog" (1)
        if (labels[1] in dognames_dic):
          labels.extend([0, 1])

        # otherwise, set both pet image label and classifier label to "not a dog" (0)
        else:
          labels.extend([0, 0])
