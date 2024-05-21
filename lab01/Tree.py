from typing import List

from PointSet import PointSet, FeaturesTypes


class Tree:
    """A decision Tree

    Attributes
    ----------
        points : PointSet
            The training points of the tree
    """
    

    def __init__(self,
                  features: List[List[float]],
                  labels: List[bool],
                  types: List[FeaturesTypes],
                  h: int = 1):
        """
        Parameters
        ----------
            labels : List[bool]
                The labels of the training points.
            features : List[List[float]]
                The features of the training points. Each sublist
                represents a single point. The sublists should have
                the same length as the `types` parameter, while the
                list itself should have the same length as the
                `labels` parameter.
            types : List[FeaturesTypes]
                The types of the features.
        """
        
        self.points = PointSet(features, labels, types)
        self.h = h
                
        # Compute the index so that I will start splitting
        self.best_feature_index, gini_gain = self.points.get_best_gain()
                
        if gini_gain != 0 and not self.points.has_all_same_spliting_feature():            
            
            self.left_point_set, self.right_point_set = self.points.split_on_feature(self.best_feature_index)
            
            if self.h > 1:
                
                if self.left_point_set.get_best_gain()[1] != 0 and not self.left_point_set.has_all_same_spliting_feature():
                    self.left_point_set = Tree(self.left_point_set.features, self.left_point_set.labels, self.left_point_set.types, self.h - 1)
                    
                if self.right_point_set.get_best_gain()[1] != 0 and not self.right_point_set.has_all_same_spliting_feature():
                    self.right_point_set = Tree(self.right_point_set.features, self.right_point_set.labels, self.right_point_set.types, self.h - 1)
                       
        


    def decide(self, features: List[float]) -> bool:
        """
        Give the guessed label of the tree to an unlabeled point

        Parameters
        ----------
            features : List[float]
                The features of the unlabeled point.

        Returns
        -------
            bool
                The label of the unlabeled point,
                guessed by the Tree
        """
        
        if features[self.best_feature_index] == self.points.value_best_spliting_feature:
            if type(self.left_point_set) == PointSet:
                return self.left_point_set.get_majority_label()
            elif type(self.left_point_set) == Tree:
                return self.left_point_set.decide(features)
        else:
            if type(self.right_point_set) == PointSet:
                return self.right_point_set.get_majority_label()
            elif type(self.right_point_set) == Tree:
                return self.right_point_set.decide(features)

