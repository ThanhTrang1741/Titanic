# **TITANIC ANALYSIS**

# **1. Introdution**

On April 15, 1912, during the maiden voyage of Titanic, the widely considered “unsinkable” RMS Titanic sank after colliding with an iceberg. Unfortunately, there were not enough lifeboats for everyone on board, resulting in the death of 1502 out of 2224 passengers and crew.

While there was some element of luck involved in surviving, it seems some groups of people were more likely to survive than others.

**Random Forest** used to the answers the question: “what sorts of people were more likely to survive?” using passenger data (ie name, age, gender, socio-economic class, etc).

# **2. Importing Libraries**

    import numpy as np # linear algebra
    import pandas as pd # data processing
    from matplotlib import pyplot as plt # visualisation basics
    import seaborn as sns # handy statistical visualisation
    import statsmodels.api as sm # Logistic Regression
    from sklearn.ensemble import RandomForestClassifier # Machine learning

# **3. Importing Training Dataset**

    train_google_sheet_id = '1UeFzpZYA3zSEN4W_BYnuezPtVA7_bRSlk5FlEy5Sdzo'
    url='https://docs.google.com/spreadsheets/d/' + train_google_sheet_id + '/export?format=xlsx'
    train_data = pd.read_excel(url,sheet_name='train')

**Dataset details are below:**

1.   PassengerId - The passenger number
2.   Survived - Survival (0 = No, 1 = Yes)
3.   Pclass - Class passenger belows (1 = 1st: Upper, 2 = 2nd: Middle, 3 = 3rd:Lower)
4.   Name - Names of passenger
5.   Sex - Gender of passenger
6.   Age - Age of passenger
7.   SibSp - # of siblings / spouses aboard the Titanic
8.   Parch - # of parents / children aboard the Titanic
9.   Ticket - Ticket number
10.  Fare - Passenger fare
11.  Cabin - Cabin number, NaN specifies no cabin allocated
12.  Embarked - Port of Embarkation	(C = Cherbourg, Q = Queenstown, S = Southampton)

    train_data.head()

|   | PassengerId | Survived | Pclass |                                              Name |    Sex |  Age | SibSp | Parch |           Ticket |    Fare | Cabin | Embarked |
|--:|------------:|---------:|-------:|--------------------------------------------------:|-------:|-----:|------:|------:|-----------------:|--------:|------:|---------:|
| 0 |           1 |        0 |      3 |                           Braund, Mr. Owen Harris |   male | 22.0 |     1 |     0 |        A/5 21171 |  7.2500 |   NaN |        S |
| 1 |           2 |        1 |      1 | Cumings, Mrs. John Bradley (Florence Briggs Th... | female | 38.0 |     1 |     0 |         PC 17599 | 71.2833 |   C85 |        C |
| 2 |           3 |        1 |      3 |                            Heikkinen, Miss. Laina | female | 26.0 |     0 |     0 | STON/O2. 3101282 |  7.9250 |   NaN |        S |
| 3 |           4 |        1 |      1 |      Futrelle, Mrs. Jacques Heath (Lily May Peel) | female | 35.0 |     1 |     0 |           113803 | 53.1000 |  C123 |        S |
| 4 |           5 |        0 |      3 |                          Allen, Mr. William Henry |   male | 35.0 |     0 |     0 |           373450 |  8.0500 |   NaN |        S |


# **4. Exploratory Data Analysis**

    train_data.info()

<class 'pandas.core.frame.DataFrame'>
RangeIndex: 891 entries, 0 to 890
Data columns (total 12 columns):
 #   Column       Non-Null Count  Dtype  
---  ------       --------------  -----  
 0   PassengerId  891 non-null    int64  
 1   Survived     891 non-null    int64  
 2   Pclass       891 non-null    int64  
 3   Name         891 non-null    object 
 4   Sex          891 non-null    object 
 5   Age          714 non-null    float64
 6   SibSp        891 non-null    int64  
 7   Parch        891 non-null    int64  
 8   Ticket       891 non-null    object 
 9   Fare         891 non-null    float64
 10  Cabin        204 non-null    object 
 11  Embarked     889 non-null    object 
dtypes: float64(2), int64(5), object(5)
memory usage: 83.7+ KB


**Analysis of categorical features**

    import matplotlib.pyplot as plt
    import seaborn as sns

    # List of columns to visualize
    col = ['SibSp', 'Parch', 'Pclass', 'Sex', 'Embarked']

    # Create a figure with a specified size
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Flatten axes array for easier iteration
    axes = axes.flatten()

    # Loop through each column and create a countplot
    for i, feature in enumerate(col):
    sns.countplot(data=train_data, x=feature, hue='Survived', ax=axes[i],
                  palette=['black', 'lightgray'])  # Black for 'not survived', light gray for 'survived'
    axes[i].set_title(f'Survival Count by {feature}')
    axes[i].set_xlabel(feature)
    axes[i].set_ylabel('Count')

    # Hide the last subplot (if there are any extra subplots)
    if len(col) < len(axes):
    for j in range(len(col), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

![image](https://github.com/user-attachments/assets/9711600d-3909-4e2f-ad67-b5e24994841f)

* The SibSp and Parch features indicate that passengers who traveled alone had significantly lower survival rates compared to those traveling with family members.
* A higher Pclass corresponds to a higher percentage of survivors.
* The survival rate for females is much higher than for males.
* Passengers boarding from different ports had varying chances of survival --> further investigation

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    sns.countplot(data=train_data,
              x='Embarked',
              hue='Pclass',
              ax=ax1,
              palette='Blues')

    sns.countplot(data=train_data,
              x='Embarked',
              hue='Sex',
              ax=ax2,
              palette='Blues')

![image](https://github.com/user-attachments/assets/f71b2793-3def-428f-8741-9b4f477f2463)

There are different combinations of Sex and Pclass for every port of embarcation. E.g. passengers who embarked from the port 'C' were mostly from the 1st Pclass and males here were just a little more than females. Now it looks reasonable why the passengers embarked from this port had higher survivability compared to others.

Explore the patterns of missing data, particularly in **the Age, Cabin, and Embarked** columns

    fig, (ax) = plt.subplots(1, figsize=(20,4))

    sns.histplot(data=train_data,
             x='Fare',
             hue='Survived',
             ax=ax)
    ax.set_title('Distribution of fare')

![image](https://github.com/user-attachments/assets/361ffab6-6314-4f9d-8fe2-bc78cff3e4f1)

On the histograms above we can see that in general increase in fare leads to increase in number of survivors for every Pclass.

    train_data.isnull().sum()

|           0 |     |
|------------:|----:|
| PassengerId |   0 |
|   Survived  |   0 |
|    Pclass   |   0 |
|     Name    |   0 |
|     Sex     |   0 |
|     Age     | 177 |
|    SibSp    |   0 |
|    Parch    |   0 |
|    Ticket   |   0 |
|     Fare    |   0 |
|    Cabin    | 687 |
|   Embarked  |   2 |


Number of null values for:

* Age - 177
* Cabin - 687
* Embarked - 2

Heat map to examine the correlation between numerical features

    train_data_num_col = train_data.select_dtypes(exclude=['object']).columns
    train_data_num = train_data[train_data_num_col]
    plt.figure(figsize=(10, 5))
    sns.heatmap(train_data_num.corr(), annot=True, cmap='Blues');

![image](https://github.com/user-attachments/assets/5ed3178e-c9d8-4001-a50f-7bf6b39c3146)

We observe that Age and Pclass have a moderate positive correlation. We'll use the average age values corresponding to each Pclass to impute missing ages

**Age**

    plt.figure(figsize=(12, 7))
    sns.boxplot(x='Pclass', y='Age', hue='Pclass', data=train_data, palette=sns.light_palette("blue", as_cmap=False, n_colors=3), legend=False);
    plt.show()

![image](https://github.com/user-attachments/assets/61afd94d-72f1-426d-8fe4-d2f839a6ded7) 

as wealthier passengers in higher classes tend to be older, which makes sense.

    # prompt: find median age by pclass
    median_age_by_pclass = train_data.groupby('Pclass')['Age'].median()
    print(median_age_by_pclass)

Pclass
1    37.0
2    29.0
3    24.0
Name: Age, dtype: float64

From the boxplot it is very clear that older people are in Pclass 1 and so on. So to impute age, we can use below code. Also the median of Pclass1 = 37, Pclass2 = 29 and PClass3 = 24.

First we can write an if-else struture to define impute_train_age(), which will return age, whose value will be median of that particular Pclass.

    def impute_train_age(Age, Pclass):
       if pd.isnull(Age):
           if Pclass == 1:
               return 37
           elif Pclass == 2:
               return 29
           else:
               return 24
       else:
           return Age

So next we can replace the null value of Age column using function impute_train_age().
    train_data['Age'] = train_data.apply(lambda row: impute_train_age(row['Age'], row['Pclass']), axis=1)


**Cabin & Embarked**

Out of 891 rows, 687 rows have null values in the Cabin column. Additionally, cabins are represented in formats like C85, C123, etc. Therefore, we can drop this column due to the high number of missing values and lack of meaningful information

    train_data.drop('Cabin',axis=1,inplace=True)

 Only two rows of Embarked is missing, so we can drop those two rows.

     train_data.dropna(inplace=True)
     train_data.info()

<class 'pandas.core.frame.DataFrame'>
Index: 889 entries, 0 to 890
Data columns (total 11 columns):
 #   Column       Non-Null Count  Dtype  
---  ------       --------------  -----  
 0   PassengerId  889 non-null    int64  
 1   Survived     889 non-null    int64  
 2   Pclass       889 non-null    int64  
 3   Name         889 non-null    object 
 4   Sex          889 non-null    object 
 5   Age          889 non-null    float64
 6   SibSp        889 non-null    int64  
 7   Parch        889 non-null    int64  
 8   Ticket       889 non-null    object 
 9   Fare         889 non-null    float64
 10  Embarked     889 non-null    object 
dtypes: float64(2), int64(5), object(4)
memory usage: 83.3+ KB


From train_data.info(), it is clear that we have 4 categorical variables in our dataset. They are ***Name, Sex, Ticket and Embarked***.

Features ***Name and Ticket*** will have no significant meaning for determining target, so we can **drop** those two. Where as ***Sex and Embarked*** features will have **to be encoded** before builiding model.

    train_data.drop(['Name','Ticket'],axis=1,inplace=True)

    # prompt: one-hot encoding
    train_data = pd.get_dummies(train_data, columns = ['Sex'], drop_first=True)
    train_data = pd.get_dummies(train_data, columns = ['Embarked'], drop_first=True)

    train_data[['Sex_male', 'Embarked_Q', 'Embarked_S']] = train_data[['Sex_male', 'Embarked_Q', 'Embarked_S']].astype(int)
    train_data.head()

|          | PassengerId | Survived | Pclass |  Age | SibSp | Parch |    Fare | Sex_male | Embarked_Q | Embarked_S |
|---------:|------------:|---------:|-------:|-----:|------:|------:|--------:|---------:|-----------:|-----------:|
|     0    |           1 |        0 |      3 | 22.0 |     1 |     0 |  7.2500 |        1 |          0 |          1 |
|     1    |           2 |        1 |      1 | 38.0 |     1 |     0 | 71.2833 |        0 |          0 |          0 |
|     2    |           3 |        1 |      3 | 26.0 |     0 |     0 |  7.9250 |        0 |          0 |          1 |
|     3    |           4 |        1 |      1 | 35.0 |     1 |     0 | 53.1000 |        0 |          0 |          1 |
|     4    |           5 |        0 |      3 | 35.0 |     0 |     0 |  8.0500 |        1 |          0 |          1 |

# **6. Building model**:
    
**Train/Test Split**

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(train_data.drop(['PassengerId','Survived'],axis=1),
                                                    train_data['Survived'], test_size=0.30,
                                                    random_state=101)


    # prompt: built random forest

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    # Evaluate the model
    from sklearn.metrics import accuracy_score, classification_report

    y_pred = rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    print(classification_report(y_test, y_pred))

Accuracy: 0.8127340823970037
|              | precision | recall | f1-score | support |
|-------------:|----------:|-------:|----------|---------|
|       0      |      0.83 | 0.87   | 0.85     | 163     |
|       1      |      0.78 | 0.72   | 0.75     | 104     |
|   accuracy   |      0.81 | 267    |          |         |
|   macro avg  |      0.81 | 0.80   | 0.80     | 267     |
| weighted avg |      0.81 | 0.81   | 0.81     | 267     |


    # Get feature importances from the trained Random Forest model
    feature_importances = pd.Series(rf_model.feature_importances_, index=X_train.columns)

    # Sort feature importances in descending order
    feature_importances = feature_importances.sort_values(ascending=False)

    # Print the feature importances
    print("Feature Importances:\n", feature_importances)

Feature Importances:
 Fare          0.278208
Age           0.269994
Sex_male      0.246287
Pclass        0.083446
SibSp         0.047131
Parch         0.042330
Embarked_S    0.021844
Embarked_Q    0.010760
dtype: float64

    # prompt: built KNN calsification

    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report

    # Assuming X_train, X_test, y_train, y_test are defined as in the original code

    # Build KNN Classifier
    knn_model = KNeighborsClassifier(n_neighbors=5)  # You can adjust the number of neighbors
    knn_model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = knn_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    print(classification_report(y_test, y_pred))
    
Accuracy: 0.6928838951310862
|              | precision | recall | f1-score | support |
|:------------:|:---------:|-------:|----------|---------|
|       0      |      0.75 | 0.75   | 0.75     | 163     |
|       1      |      0.61 | 0.61   | 0.61     | 104     |
|   accuracy   |      0.69 | 267    | 0.80     | 267     |
|   macro avg  |      0.68 | 0.68   | 0.68     | 267     |
| weighted avg |      0.69 | 0.69   | 0.69     | 267     |

    # prompt: built decesion tree

    from sklearn.tree import DecisionTreeClassifier
    # Assuming X_train, X_test, y_train, y_test are defined as in the original code

    # Build Decision Tree Classifier
    dt_model = DecisionTreeClassifier(random_state=42) # You can adjust hyperparameters
    dt_model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = dt_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    print(classification_report(y_test, y_pred))
Accuracy: 0.7715355805243446
|              | precision | recall | f1-score | support |
|:------------:|:---------:|-------:|----------|---------|
|       0      |      0.81 | 0.82   | 0.81     | 163     |
|       1      |      0.71 | 0.69   | 0.70     | 104     |
|   accuracy   |      0.77 | 267    | 0.76     | 267     |
|   macro avg  |      0.76 | 0.76   | 0.76     | 267     |
| weighted avg |      0.77 | 0.77   | 0.77     | 267     |

# prompt: buitl gradient boosting

from sklearn.ensemble import GradientBoostingClassifier

# Assuming X_train, X_test, y_train, y_test are defined as in the original code

# Build Gradient Boosting Classifier
gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42) # You can adjust hyperparameters
gb_model.fit(X_train, y_train)

# Evaluate the model
y_pred = gb_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print(classification_report(y_test, y_pred))

Accuracy: 0.8352059925093633
|              | precision | recall | f1-score | support |
|:------------:|:---------:|-------:|----------|---------|
|       0      |      0.84 | 0.90   | 0.87     | 163     |
|       1      |      0.83 | 0.73   | 0.78     | 104     |
|   accuracy   |      0.84 | 267    | 0.76     | 267     |
|   macro avg  |      0.83 | 0.82   | 0.82     | 267     |
| weighted avg |      0.83 | 0.84   | 0.83     | 267     |

    # prompt: built XGboost classification

    import xgboost as xgb
    from sklearn.metrics import accuracy_score, classification_report

    # Assuming X_train, X_test, y_train, y_test are defined as in the original code

    # Build XGBoost Classifier
    xgb_model = xgb.XGBClassifier(objective="binary:logistic", random_state=42) # You can adjust hyperparameters
    xgb_model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = xgb_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    print(classification_report(y_test, y_pred))

Accuracy: 0.8277153558052435
|              | precision | recall | f1-score | support |
|:------------:|:---------:|-------:|----------|---------|
|       0      |      0.84 | 0.89   | 0.86     | 163     |
|       1      |      0.81 | 0.73   | 0.77     | 104     |
|   accuracy   |      0.83 | 267    | 0.82     | 267     |
|   macro avg  |      0.82 | 0.81   | 0.82     | 267     |
| weighted avg |      0.83 | 0.83   | 0.83     | 267     |


    # prompt: built naive bayes

    from sklearn.naive_bayes import GaussianNB

    # Assuming X_train, X_test, y_train, y_test are defined as in the original code

    # Build Naive Bayes Classifier
    nb_model = GaussianNB()
    nb_model.fit(X_train, y_train)

    # Evaluate the model
   y_pred = nb_model.predict(X_test)
   accuracy = accuracy_score(y_test, y_pred)
   print("Accuracy:", accuracy)
   print(classification_report(y_test, y_pred))

Accuracy: 0.8052434456928839

|              | precision | recall | f1-score | support |
|:------------:|:---------:|-------:|----------|---------|
|       0      |      0.81 | 0.88   | 0.85     | 163     |
|       1      |      0.79 | 0.68   | 0.73     | 104     |
|   accuracy   |      0.81 | 267    | 0.82     | 267     |
|   macro avg  |      0.80 | 0.78   | 0.79     | 267     |
| weighted avg |      0.80 | 0.81   | 0.80     | 267     |


* **"Not Survived" Metrics:** Gradient Boosting performed the best with the highest recall (0.93) and F1-score (0.87), indicating better detection of the not survived class. 
* **"Survived" Metrics**: Gradient Boosting also has the highest precision (0.88) for the survived class In summary, there is room for improvement in the recall for the "Survived" class across all three models, particularly in the Random Forest model.


    # prompt: confusion matrix4

    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Assuming y_test and y_pred are defined from the model prediction
    # Example: y_pred = rf_model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Predicted 0", "Predicted 1"],
            yticklabels=["Actual 0", "Actual 1"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.show()

![Uploading image.png…]()


    # prompt: logistic regression

    # Assuming X_train, X_test, y_train, y_test are defined as in the original code

    # Build Logistic Regression model
    logreg_model = sm.Logit(y_train, sm.add_constant(X_train))
    result = logreg_model.fit()

    # Evaluate the model
    y_pred_prob = result.predict(sm.add_constant(X_test))
    y_pred = (y_pred_prob > 0.5).astype(int) # Convert probabilities to class labels

    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    print(classification_report(y_test, y_pred))

    # Print model summary (optional, but helpful for analysis)
    print(result.summary())

Optimization terminated successfully.
         Current function value: 0.429874
         Iterations 6
Accuracy: 0.7910447761194029
              precision    recall  f1-score   support

           0       0.78      0.88      0.83       154
           1       0.80      0.68      0.73       114

    accuracy                           0.79       268
   macro avg       0.79      0.78      0.78       268
weighted avg       0.79      0.79      0.79       268

                           Logit Regression Results                           
==============================================================================
Dep. Variable:               Survived   No. Observations:                  623
Model:                          Logit   Df Residuals:                      614
Method:                           MLE   Df Model:                            8
Date:                Sat, 26 Oct 2024   Pseudo R-squ.:                  0.3455
Time:                        08:38:32   Log-Likelihood:                -267.81
converged:                       True   LL-Null:                       -409.17
Covariance Type:            nonrobust   LLR p-value:                 1.950e-56
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
const          5.2482      0.739      7.097      0.000       3.799       6.698
Pclass        -1.1255      0.188     -5.992      0.000      -1.494      -0.757
Age           -0.0412      0.010     -4.079      0.000      -0.061      -0.021
SibSp         -0.3044      0.126     -2.416      0.016      -0.551      -0.057
Parch         -0.0978      0.142     -0.690      0.490      -0.375       0.180
Fare           0.0036      0.004      1.021      0.307      -0.003       0.010
Sex_male      -2.7491      0.239    -11.513      0.000      -3.217      -2.281
Embarked_Q    -0.0840      0.460     -0.183      0.855      -0.986       0.818
Embarked_S    -0.4395      0.286     -1.535      0.125      -1.001       0.122
==============================================================================


* Pseudo R-squared: 0.3455, indicating a moderate fit of the model. 
* Log-Likelihood: -267.81, which is useful for model comparison.
* LLR p-value: 1.950×10−56, suggesting that the model significantly predicts survival.

* **Sex_male:** -2.7491:

The strong association of being male with decreased survival odds aligns with the historical narrative that women and children were prioritized in lifeboat assignments, reinforcing gender dynamics during the disaster. 

* **Age:** -0.0412: 

The trend showing that older passengers had lower survival odds is consistent with accounts of the disaster, where younger individuals, especially women and children, were often prioritized for lifeboat access. 

* **Pclass: **-1.1255: 

Passengers in first class had significantly higher survival rates compared to those in second and third classes. This reflects socioeconomic factors that influenced access to lifeboats and safety. 

* **SibSp: **-0.3044 

The negative correlation with the number of siblings/spouses aboard suggests that larger family groups may have had more difficulty in the chaotic evacuation, leading to lower survival rates.

# **Key Insights**

Our study confirms that survival was significantly influenced by gender, class, and family presence. Among the machine learning models, Random Forest and Gradient Boosting demonstrated the best performance. Future work may involve striving to enhance model accuracy and investigating additional features.
