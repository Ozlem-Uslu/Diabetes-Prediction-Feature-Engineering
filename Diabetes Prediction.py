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

#1.2.Einlesen eines Datensatzes
dff = pd.read_csv("diabetes.csv")

#Wir haben den Datensatz geladen! :)

#Ich habe die Daten der Variablen dff zugewiesen. Jetzt kopiere ich es und weise es der Variablen df zu. Wenn ich auf ein Problem stoße, muss ich den Datensatz nicht erneut lesen, 🪖 Ich mache den Kopiervorgang einfach noch einmal. (Diese Technik wird normalerweise bei großen Datensätzen angewendet ...)

df = dff.copy()
df.head()

#1.3. Explorative Datenanalyse(EDA)
#Ich werde die Funktion, die ich zuvor geschrieben habe, verwenden, um einen kurzen Blick auf den Datensatz zu werfen. Diese Funktion funktioniert, Dinge wie Form, Head, doppelte Werte und fehlende Werte zu erkennen und Quantilwerte zu beobachten.💁🏻
def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df)

#1.3.1.Definition einer Funktion, um die numerischen und kategorialen Variablen des Datensatzes zu erfassen
def grab_col_names(dataframe, cat_th=10, car_th=20):

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)

cat_cols
num_cols

#1.3.2. Analyse kategorischer Variablen
def cat_summary(dataframe, cat_cols, plot=True):
    for col_name in cat_cols:
        print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                            "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))

        if plot:
            sns.countplot(x=dataframe[col_name], data=dataframe)
            plt.show()

cat_summary(df, cat_cols)

#1.3.3. Analyse numerischer Variablen¶
def num_summary(dataframe, numerical_col, plot=True):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()

for col in num_cols:
    num_summary(df, col, plot=True)

#1.3.4. Analyse numerischer Variablen nach Zielvariable
df["Outcome"].value_counts()

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

for col in num_cols:
    target_summary_with_num(df, "Outcome", col)

#1.3.5. Korrelationsanalyse

#Korrelation gibt die Richtung und Stärke der linearen Beziehung zwischen zwei Zufallsvariablen in der Wahrscheinlichkeitstheorie und Statistik an.
df.corr()

# Korrelationsmatrix
f, ax = plt.subplots(figsize=[10, 8])
sns.heatmap(df.corr(), annot=True, fmt=".2f", ax=ax, cmap="magma")
ax.set_title("Correlation Matrix", fontsize=18)
plt.show()

#2. Feature-Engineering
#2.1. Ausreißeranalyse (Outliers Analysis)

#Zunächst definieren wir eine Funktion, um Ausreißer leichter zu finden.
def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit
# Schreiben wir nun mit Hilfe der oben erstellten Funktion eine weitere Funktion, um zu prüfen, ob in den Spalten Werte fehlen.
def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False
for col in df.columns:
    print(col, check_outlier(df, col))

#2.2. Die Analyse fehlender Werte(The Missing Values Analysis)¶
#Es ist bekannt, dass andere Variablenwerte als „Schwangerschaften(Pregnancies)“ und „Ergebnis(Outcome)“ beim Menschen nicht "0" sein können. Daher müssen Handlungsentscheidungen hinsichtlich dieser Werte getroffen werden. Werte, die "0" sind, können NaN zugewiesen werden.

# Wählen wir die Spalten aus, in denen wir diesen Vorgang durchführen werden.
zero_columns = [col for col in df.columns if (df[col].min() == 0 and col not in ["Pregnancies", "Outcome"])]
zero_columns

# Gehen wir zu jeder Variablen mit 0 in den Beobachtungseinheiten und ersetzen wir die Beobachtungswerte, die 0 enthalten, durch NaN.
for col in zero_columns:
    df[col] = np.where(df[col] == 0, np.nan, df[col])

df.isnull().sum()

# Füllen wir die fehlenden Werte mit den Medianen der Variablen aus.
for col in zero_columns:
    df[col] = df[col].fillna(df[col].median())

# Nach dem Füllvorgang werden fehlende Werte überprüft.
df.isnull().sum()

#2.3. Erstellen neuer Funktionsinteraktionen
df.head()

# Erstellung kategoriale Glucose-Variablen
df.loc[(df['Glucose'] < 70), 'New_Glucose_Cat'] = 'hypoglycemia'
df.loc[(df['Glucose'] >= 70) & (df['Glucose'] < 100), 'New_Glucose_Cat'] = 'normal'
df.loc[(df['Glucose'] >= 100) & (df['Glucose'] < 126), 'New_Glucose_Cat'] = 'secret candy'
df.loc[(df['Glucose'] >= 126), 'New_Glucose_Cat'] = 'diabetes'

df.groupby(["New_Glucose_Cat"]).agg({"Outcome": ["mean","count"]})

#Erstellung kategorialer Age-Variablen
df.loc[(df['Age'] < 25), 'New_Age_Cat'] = 'young'
df.loc[(df['Age'] >= 25) & (df['Age'] < 56), 'New_Age_Cat'] = 'mature'
df.loc[(df['Age'] >= 56), 'New_Age_Cat'] = 'senior'

df.groupby(["New_Age_Cat"]).agg({"Outcome": ["mean","count"]})

# Erstellung kategorialer diastolischen Blutdruck-Variablen
df.loc[(df['BloodPressure'] < 80), 'New_BloodPressure_Cat'] = 'normal'
df.loc[(df['BloodPressure'] >= 80) & (df['Glucose'] < 90), 'New_BloodPressure_Cat'] = 'high'
df.loc[(df['BloodPressure'] >= 90), 'New_BloodPressure_Cat'] = 'hypertension'

df.groupby(["New_BloodPressure_Cat"]).agg({"Outcome": ["mean","count"]})

#Erstellung kategorialer diastolischen Body-Mass-Index-Variablen
df.loc[(df['BMI'] < 18.5), 'New_BMI_Cat'] = 'unterweight'
df.loc[(df['BMI'] >= 18.5) & (df['BMI'] < 25), 'New_BMI_Cat'] = 'normalweight'
df.loc[(df['BMI'] >= 25) & (df['BMI'] < 30), 'New_BMI_Cat'] = 'overweight'
df.loc[(df['BMI'] >= 30), 'New_BMI_Cat'] = 'obese'

df.groupby(["New_BMI_Cat"]).agg({"Outcome": ["mean","count"]})

# Estellung kategorialer diastolischen Insulin-Variablen
df.loc[(df['Insulin'] < 120), 'New_Insulin_Cat'] ="normal"
df.loc[(df['Insulin'] >= 120), 'New_Insulin_Cat'] ="abnormal"

df.groupby(["New_Insulin_Cat"]).agg({"Outcome": ["mean","count"]})

# Erstellung kategorialer Swangerschaft-Variablen
df.loc[(df['Pregnancies'] == 0), 'New_Preg_Cat'] ="unpregnant"
df.loc[(df['Pregnancies'] > 0) & (df['Pregnancies'] <= 5), 'New_Preg_Cat'] ="normal"
df.loc[(df['Pregnancies'] > 5) & (df['Pregnancies'] <= 10), 'New_Preg_Cat'] ="high"
df.loc[(df['Pregnancies'] > 10), 'New_Preg_Cat'] ="very high"

df.groupby(["New_Preg_Cat"]).agg({"Outcome": ["mean","count"]})

#Erstellung kategorialer Age-Insulin-Variablen
df.loc[((df['New_Insulin_Cat'] == "abnormal") & (df["New_Age_Cat"] == "mature")), 'Age_Insilun_Cat'] ="mature_abnormal"
df.loc[((df['New_Insulin_Cat'] == "abnormal") & (df["New_Age_Cat"] == "senior")), 'Age_Insilun_Cat'] ="senior_abnormal"
df.loc[((df['New_Insulin_Cat'] == "abnormal") & (df["New_Age_Cat"] == "young")), 'Age_Insilun_Cat'] ="young_abnormal"
df.loc[((df['New_Insulin_Cat'] == "normal") & (df["New_Age_Cat"] == "mature")), 'Age_Insilun_Cat'] ="mature_normal"
df.loc[((df['New_Insulin_Cat'] == "normal") & (df["New_Age_Cat"] == "senior")), 'Age_Insilun_Cat'] ="senior_normal"
df.loc[((df['New_Insulin_Cat'] == "normal") & (df["New_Age_Cat"] == "young")), 'Age_Insilun_Cat'] ="young_normal"

df.groupby(["Age_Insilun_Cat"]).agg({"Outcome": ["mean","count"]})

df.head()

#Dem Datensatz wurden 7 neue Variablen(New_Glucose_Cat, New_Age_Cat, New_BloodPressure_Cat, New_BMI_Cat, New_Insulin_Cat, New_Preg_Cat, Age_Insilun_Cat) hinzugefügt.

#3.Modellierung
#3.1.Verarbeitung der Kodierung & One-Hot-Kodierung

# Label Encoding & Binary Encoding
binary_cols = [col for col in df.columns if df[col].dtype not in [int, float]
               and df[col].nunique() == 2]
df[binary_cols].head()

# Definition einer Funktion zum Label-Encoder
def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

# Anwendung der Laber-Encoding-Funktion auf Binär-Spalte
for col in binary_cols:
    df = label_encoder(df, col)

df.head()

# Definition einer Funktion zum One-Hot-Encoder
def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first, dtype=int)
    return dataframe

# Drucken von Spalten, in denen wir die One-Hot-Encoding-Funktion anwenden können
ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]
df = one_hot_encoder(df, ohe_cols)

df.head()

#Alle Variablen wurden in einen numerischen Typ konvertiert.

df.info()

#Wie ersichtlich ist, erhöhte sich die Anzahl der Variablen im Datensatz nach dem One-Hot-Encoding-Prozess von 16 auf 28.

#3.2. Standardisierung für numerische Variablen

# Um numerische Variablen zu standardisieren, erfassen wir zunächst die numerischen Spalten.
def grab_col_names(dataframe, cat_th=10, car_th=20):
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = grab_col_names(df)

#Standardisierung der numerischen Variablen
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

# Prüfung der numerischen Variablen,die standardisiert sind.
df[num_cols].head()

#Daten prüfen
df.head()

#3.3. Auswahl des besten Modells¶
#Aufteilen der Daten

#Wir werden einen Teil unserer Daten (in diesem Fall 30 %) verwenden, um die Genauigkeit unserer verschiedenen Modelle zu testen.

y = df["Outcome"]     #Target
x = df.drop(["Outcome"], axis=1) #non-target variablen
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=17)

# Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

gaussian = GaussianNB()
gaussian.fit(x_train, y_train)
y_pred = gaussian.predict(x_test)
acc_gaussian = round(accuracy_score(y_pred, y_test) * 100, 2)
print(acc_gaussian)

# Logistic Regression
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(x_train, y_train)
y_pred = logreg.predict(x_test)
acc_logreg = round(accuracy_score(y_pred, y_test) * 100, 2)
print(acc_logreg)

# Support Vector Machines
from sklearn.svm import SVC

svc = SVC()
svc.fit(x_train, y_train)
y_pred = svc.predict(x_test)
acc_svc = round(accuracy_score(y_pred, y_test) * 100, 2)
print(acc_svc)

# Linear SVC
from sklearn.svm import LinearSVC

linear_svc = LinearSVC()
linear_svc.fit(x_train, y_train)
y_pred = linear_svc.predict(x_test)
acc_linear_svc = round(accuracy_score(y_pred, y_test) * 100, 2)
print(acc_linear_svc)

# Perceptron
from sklearn.linear_model import Perceptron

perceptron = Perceptron()
perceptron.fit(x_train, y_train)
y_pred = perceptron.predict(x_test)
acc_perceptron = round(accuracy_score(y_pred, y_test) * 100, 2)
print(acc_perceptron)

#Decision Tree
from sklearn.tree import DecisionTreeClassifier

decisiontree = DecisionTreeClassifier()
decisiontree.fit(x_train, y_train)
y_pred = decisiontree.predict(x_test)
acc_decisiontree = round(accuracy_score(y_pred, y_test) * 100, 2)
print(acc_decisiontree)

# Random Forest
from sklearn.ensemble import RandomForestClassifier

randomforest = RandomForestClassifier()
randomforest.fit(x_train, y_train)
y_pred = randomforest.predict(x_test)
acc_randomforest = round(accuracy_score(y_pred, y_test) * 100, 2)
print(acc_randomforest)

# KNN or k-Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)
acc_knn = round(accuracy_score(y_pred, y_test) * 100, 2)
print(acc_knn)

# Stochastic Gradient Descent
from sklearn.linear_model import SGDClassifier

sgd = SGDClassifier()
sgd.fit(x_train, y_train)
y_pred = sgd.predict(x_test)
acc_sgd = round(accuracy_score(y_pred, y_test) * 100, 2)
print(acc_sgd)

# Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier

gbk = GradientBoostingClassifier()
gbk.fit(x_train, y_train)
y_pred = gbk.predict(x_test)
acc_gbk = round(accuracy_score(y_pred, y_test) * 100, 2)
print(acc_gbk)

#Vergleichen wir die Genauigkeiten der einzelnen Modelle!

models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression',
              'Random Forest', 'Naive Bayes', 'Perceptron', 'Linear SVC',
              'Decision Tree', 'Stochastic Gradient Descent', 'Gradient Boosting Classifier'],
    'Score': [acc_svc, acc_knn, acc_logreg,
              acc_randomforest, acc_gaussian, acc_perceptron,acc_linear_svc, acc_decisiontree,
              acc_sgd, acc_gbk]})
models.sort_values(by='Score', ascending=False)