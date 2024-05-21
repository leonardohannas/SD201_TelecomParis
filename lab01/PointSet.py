from typing import List, Tuple

from enum import Enum
import numpy as np

class FeaturesTypes(Enum):
    """Enumerate possible features types"""
    BOOLEAN=0
    CLASSES=1
    REAL=2

class PointSet:
    """A class representing set of training points.

    Attributes
    ----------
        types : List[FeaturesTypes]
            Each element of this list is the type of one of the
            features of each point
        features : np.array[float]
            2D array containing the features of the points. Each line
            corresponds to a point, each column to a feature.
        labels : np.array[bool]
            1D array containing the labels of the points.
    """
    
    def __init__(self, features: List[List[float]], labels: List[bool], types: List[FeaturesTypes]):
        """
        Parameters
        ----------
        features : List[List[float]]
            The features of the points. Each sublist contained in the
            list represents a point, each of its elements is a feature
            of the point. All the sublists should have the same size as
            the `types` parameter, and the list itself should have the
            same size as the `labels` parameter.
        labels : List[bool]
            The labels of the points.
        types : List[FeaturesTypes]
            The types of the features of the points.
        """
        self.types = types
        self.features = np.array(features)
        self.labels = np.array(labels)
        
        self.value_best_spliting_feature = 0
        self.index_feature_best_spliting = None
    

    def get_gini(self) -> float:
        """        
        Computes the Gini score of the set of points
        Returns
        -------
        float
            The Gini score of the set of points
        """
 
        numberLabelsTrue = 0
        numberLabelsFalse = 0

        for label in self.labels:
            if label == True:
                numberLabelsTrue += 1
            else:
                numberLabelsFalse += 1

        probabilityOfTrues = numberLabelsTrue / len(self.labels)
        probabilityOfFalse = numberLabelsFalse / len(self.labels)

        gini = 1 - (probabilityOfTrues ** 2 + probabilityOfFalse ** 2)
            
        return gini


    def compute_gini(self, numberLabelsTrue, numberLabelsFalse):
        """        
        Computes the Gini score of the set of points
        Returns
        -------
        float
            The Gini score of the set of points
        """
        
        if numberLabelsTrue == 0 and numberLabelsFalse == 0:
            return 1
        
        numberTotalLabels = numberLabelsTrue + numberLabelsFalse
        gini = 1 - (numberLabelsTrue / numberTotalLabels) ** 2 - (numberLabelsFalse / numberTotalLabels) ** 2
       
        return gini
        

    def get_best_gain(self) -> Tuple[int, float]:
        """
        Compute the feature along which splitting provides the best gain
        Returns
        -------
        int
            The ID of the feature along which splitting the set provides the
        best Gini gain.
        float
            The best Gini gain achievable by splitting this set along one of
        its features.
        """
        
        # Stores the index of the feature used to split, its respective Gini
        # Gain and the unique value of the feature that was used to split
        giniGainArray = [] 
        
        # List of lists containing every unique value of each feature
        for column in range(len(self.features[0])):
            unique_values = np.unique(self.features[:,column])  # Unique values of each feature
            
            for unique_value in unique_values:
                
                numberLabelsFalseFeaturePosJ = 0
                numberLabelsTrueFeaturePosJ = 0
            
                numberLabelsFalseFeaturePosDifJ = 0
                numberLabelsTrueFeaturePosDifJ = 0
                
                if self.types[column] == FeaturesTypes.CLASSES or self.types[column] == FeaturesTypes.BOOLEAN:

                    for row in range(len(self.features)):
                        
                        if self.features[row][column] == unique_value: # feature c1 = X
                            if self.labels[row] == True:  # label = True
                                numberLabelsTrueFeaturePosJ += 1
                            else:  # label = False
                                numberLabelsFalseFeaturePosJ += 1
                                
                        else:  # feature c1 != X
                            if self.labels[row] == True:  # label = True
                                numberLabelsTrueFeaturePosDifJ += 1
                            else:  # label = False
                                numberLabelsFalseFeaturePosDifJ += 1 
            
                    # Gini of the partitions
                    giniFeaturePosJ = self.compute_gini(numberLabelsTrueFeaturePosJ, numberLabelsFalseFeaturePosJ)
                    giniFeaturePosDifJ = self.compute_gini(numberLabelsTrueFeaturePosDifJ, numberLabelsFalseFeaturePosDifJ)
                    
                    gini_split = ((numberLabelsFalseFeaturePosDifJ + numberLabelsTrueFeaturePosDifJ) * giniFeaturePosDifJ + (numberLabelsFalseFeaturePosJ + numberLabelsTrueFeaturePosJ) * giniFeaturePosJ) / (numberLabelsFalseFeaturePosDifJ + numberLabelsTrueFeaturePosDifJ + numberLabelsFalseFeaturePosJ  + numberLabelsTrueFeaturePosJ)
                    gini_gain = self.get_gini() - gini_split
                    
                    giniGainArray.append([column, gini_gain, unique_value])
                
                # Same logic if the feature is of type REAL
                elif self.types[column] == FeaturesTypes.REAL:
                    for row in range(len(self.features)):
                        
                        if self.features[row][column] < unique_value: # c1 = X
                            if self.labels[row] == True: # label = True
                                numberLabelsTrueFeaturePosJ += 1
                            else: # label = False
                                numberLabelsFalseFeaturePosJ += 1
                                
                        else: # self.features[l][k] != 0, ie, c1 != X
                            if self.labels[row] == True: # label = True
                                numberLabelsTrueFeaturePosDifJ += 1
                            else: # label = False
                                numberLabelsFalseFeaturePosDifJ += 1 
            
                    giniFeaturePosJ = self.compute_gini(numberLabelsTrueFeaturePosJ, numberLabelsFalseFeaturePosJ)
                    giniFeaturePosDifJ = self.compute_gini(numberLabelsTrueFeaturePosDifJ, numberLabelsFalseFeaturePosDifJ)
                    
                    gini_split = ((numberLabelsFalseFeaturePosDifJ + numberLabelsTrueFeaturePosDifJ) * giniFeaturePosDifJ + (numberLabelsFalseFeaturePosJ + numberLabelsTrueFeaturePosJ) * giniFeaturePosJ) / (numberLabelsFalseFeaturePosDifJ + numberLabelsTrueFeaturePosDifJ + numberLabelsFalseFeaturePosJ  + numberLabelsTrueFeaturePosJ)
                    gini_gain = self.get_gini() - gini_split
                    
                    giniGainArray.append([column, gini_gain, unique_value])
                    
              
        index = -1
        value_best_spliting_feature = 0                
        max_gini_gain = - 1

        for i in range(len(giniGainArray)):
            if giniGainArray[i][1] > max_gini_gain:
                max_gini_gain = giniGainArray[i][1]
                value_best_spliting_feature = giniGainArray[i][2]
                index = giniGainArray[i][0]
   
      
        self.value_best_spliting_feature = value_best_spliting_feature   
        
        self.index_feature_best_spliting = index
        return (index, max_gini_gain)
                

    def split_on_feature(self, feature_index: int):
        """
        Splits the PointSet based on a feature.

        Args:
            feature_index (int): The index of the feature to split on.

        Returns:
            tuple: Two PointSets, one where the feature equals the best splitting feature and one where it doesn't.
        """
        
        features_equal_to_best_splitting_feature = []
        features_dif_to_best_splitting_feature = []
        
        labels_equal_to_best_splitting_feature = []
        labels_dif_to_best_splitting_feature = []
        
        print("For the question 'eval 6', it is taking about 2 minutes to run this function")
        
        if self.types[feature_index] == FeaturesTypes.BOOLEAN or self.types[feature_index] == FeaturesTypes.CLASSES:
        
            for i in range(len(self.features)):
                if self.features[i,feature_index] == self.value_best_spliting_feature:
                    features_equal_to_best_splitting_feature.append(self.features[i].copy())
                    labels_equal_to_best_splitting_feature.append(self.labels[i].copy())
                else:
                    features_dif_to_best_splitting_feature.append(self.features[i].copy())
                    labels_dif_to_best_splitting_feature.append(self.labels[i].copy())
                
        
        elif self.types[feature_index] == FeaturesTypes.REAL:
            
            for i in range(len(self.features)):
                if self.features[i,feature_index] < self.get_best_threshold():
                    features_equal_to_best_splitting_feature.append(self.features[i].copy())
                    labels_equal_to_best_splitting_feature.append(self.labels[i].copy())
                else:
                    features_dif_to_best_splitting_feature.append(self.features[i].copy())
                    labels_dif_to_best_splitting_feature.append(self.labels[i].copy())
                    
        
        points_equal_to_best_splitting_feature = PointSet(features_equal_to_best_splitting_feature, labels_equal_to_best_splitting_feature, self.types)
        points_dif_to_best_splitting_feature = PointSet(features_dif_to_best_splitting_feature, labels_dif_to_best_splitting_feature, self.types)
        
        return points_equal_to_best_splitting_feature, points_dif_to_best_splitting_feature 
        

    def get_majority_label(self): 
        """
        Determines the majority label in the PointSet.

        Returns:
            bool: True if the majority of labels are True, False otherwise.
        """
        
        counter_of_falses = counter_of_trues = 0
        for i in range(len(self.labels)):
            if self.labels[i] == True:
                counter_of_trues += 1
            else:
                counter_of_falses += 1 
        
        if counter_of_trues >= counter_of_falses:
            return True
        return False
    
    
    def get_best_threshold(self) -> float:
        """
        Determines the best threshold for splitting the PointSet.

        Returns:
            float: The best threshold for splitting. If the best splitting feature is of type 
            CLASSES,returns the index of the feature. If it's of type BOOLEAN, returns None.
        """
        
        if self.types[self.index_feature_best_spliting] == FeaturesTypes.CLASSES:
            return self.index_feature_best_spliting
        
        elif self.types[self.index_feature_best_spliting] == FeaturesTypes.BOOLEAN:
            return None
        
        elif self.types[self.index_feature_best_spliting] == FeaturesTypes.REAL:
            greater_than_best_spliting_value = []
            smaller_than_best_spliting_value = []
            
            for feature_value in self.features[:,self.index_feature_best_spliting]:
                
                if feature_value < self.value_best_spliting_feature:
                    smaller_than_best_spliting_value.append(feature_value)
                     
                else:
                    greater_than_best_spliting_value.append(feature_value)
                
            return (max(smaller_than_best_spliting_value, default=0) + min(greater_than_best_spliting_value, default=0)) / 2



    def has_all_same_spliting_feature(self) -> bool:
        """
        Checks if all features in the PointSet have the same value for the best splitting feature.

        Returns:
            bool: True if all features have the same value for the best splitting feature, False otherwise.
        """
            
        for i in range(len(self.features)):
            if self.features[i][self.index_feature_best_spliting] != self.value_best_spliting_feature:
                return False
        return True
    
    