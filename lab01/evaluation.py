from typing import List

def precision_recall(expected_results: List[bool], actual_results: List[bool]) -> (float, float):
    """
    Compute the precision and recall of a series of predictions

    Parameters
    ----------
        expected_results : List[bool]
            The true results, that is the results that the predictor
            should have find.
        actual_results : List[bool]
            The predicted results, that have to be evaluated.

    Returns
    -------
        float
            The precision of the predicted results.
        float
            The recall of the predicted results.
    """
    
    true_positives = false_positives = false_negatives = 0
    
    for i in range(len(actual_results)):
        if actual_results[i] == True and expected_results[i] == True:
            true_positives += 1
        elif actual_results[i] == True and expected_results[i] == False:
            false_positives += 1
        elif actual_results[i] == False and expected_results[i] == True:
            false_negatives += 1
    
    if true_positives == 0:
        precision = 0
        recall = 0
    else:
        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)
    
    return precision, recall



def F1_score(expected_results: List[bool], actual_results: List[bool]) -> float:
    """Compute the F1-score of a series of predictions

    Parameters
    ----------
        expected_results : List[bool]
            The true results, that is the results that the predictor
            should have find.
        actual_results : List[bool]
            The predicted results, that have to be evaluated.

    Returns
    -------
        float
            The F1-score of the predicted results.
    """
    
    precision, recall = precision_recall(expected_results, actual_results)
    
    if precision == 0 or recall == 0:
        f1_score = 0
        
    else:
        f1_score = (2 * precision * recall) / (precision + recall)
    
    return f1_score
