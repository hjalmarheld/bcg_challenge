"""
This is where you put a model adopting the sklearn api

Final object must be named model
"""


from lightgbm import LGBMClassifier

model = LGBMClassifier(scale_pos_weight=5)