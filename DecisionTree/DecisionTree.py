"""
Authors :
    1) Vandit Maheshwari
    2) Aviral Upadhyay
    3) Rishika Verma
"""

import math
import pandas as pd
from pandas import DataFrame
from collections import Counter

def entropy(probs):
    #calculate entropy
    return sum([-prob*math.log2(prob) for prob in probs])

def entropy_of_list(a_list):
    count = Counter(x for x in a_list)
    num_instances = len(a_list)
    probs = [x/num_instances for x in count.values()]
    return entropy(probs)

def information_gain(data,split_attribute_name,target_attribute,trace=0):
    data_split = data.groupby(split_attribute_name)
    nobs = len(data.index) * 1.0
    df_agg_ent = data_split.agg({target_attribute : [entropy_of_list, lambda x: len(x)/nobs] })[target_attribute]
    df_agg_ent.columns = ['Entropy', 'PropObservations']
    if trace: # helps understand what function is doing:
        print(df_agg_ent)
    
    # Calculate Information Gain:
    new_entropy = sum( df_agg_ent['Entropy'] * df_agg_ent['PropObservations'] )
    old_entropy = entropy_of_list(data[target_attribute])
    return old_entropy-new_entropy

def id3(df, target_attribute_name, attribute_names, default_class=None):
    
    ## Tally target attribute:
    from collections import Counter
    cnt = Counter(x for x in df[target_attribute_name])
    
    ## First check: Is this split of the dataset homogeneous?
    if len(cnt) == 1:
        return cnt.keys()
    
    ## Second check: Is this split of the dataset empty?
    # if yes, return a default value
    elif df.empty or (not attribute_names):
        return default_class 
    
    ## Otherwise: This dataset is ready to be divvied up!
    else:
        # Get Default Value for next recursive call of this function:
        index_of_max = list(cnt.values()).index(max(cnt.values()))
        default_class = cnt.most_common()[index_of_max]
        
        # Choose Best Attribute to split on:
        gainz = [information_gain(df, attr, target_attribute_name) for attr in attribute_names]
        index_of_max = gainz.index(max(gainz)) 
        best_attr = attribute_names[index_of_max]
        
        # Create an empty tree, to be populated in a moment
        tree = {best_attr:{}}
        remaining_attribute_names = [i for i in attribute_names if i != best_attr]
        
        # Split dataset
        # On each split, recursively call this algorithm.
        # populate the empty tree with subtrees, which
        # are the result of the recursive call
        for attr_val, data_subset in df.groupby(best_attr):
            subtree = id3(data_subset,
                        target_attribute_name,
                        remaining_attribute_names,
                        default_class)
            tree[best_attr][attr_val] = subtree
        return tree

def classify(instance, tree, default='Yes'):
    attribute = list(tree.keys())[0]
    if instance[attribute] in tree[attribute].keys():
        result = tree[attribute][instance[attribute]]
        if isinstance(result,dict): # this is a tree
            return classify(instance,result)
        else:
            return result # this is a label
    else:
        return default