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

