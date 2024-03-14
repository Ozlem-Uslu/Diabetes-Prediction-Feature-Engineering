#1. Datenvorverarbeitung
#1.1. Bibliotheken importieren

#Grundlegende Bibliotheken
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Metriken ETC..
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

#Modelle für maschinelles Lernen
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

#Anpassung, um Warnungen zu entfernen und eine bessere Beobachtung zu ermöglichen
import warnings
warnings.simplefilter(action="ignore")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_rows', 20)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
