import pandas as pd

def load_titanic_dataset():
    df = pd.read_csv("data/titanic.csv")


    df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], inplace=True)


    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])


    df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

    return df
