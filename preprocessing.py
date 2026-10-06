import pandas as pd
def get_communities_data(as_df=False):

    df = pd.read_csv('/Users/deborah/PycharmProjects/FairModelSelection/data/communities.csv')
    df = df.fillna(0)

    sens_attrs = ['racepctblack', 'racePctWhite', 'racePctAsian', 'racePctHisp']
    df['race'] = df[sens_attrs].idxmax(axis=1) # creating a new column based on ethnicity
    df = df.drop(columns=sens_attrs)

    df = df.drop(df[df['ViolentCrimesPerPop']==0].index)
    y = df['ViolentCrimesPerPop'] #target
    df = df.drop('ViolentCrimesPerPop', axis=1)

    mapping = {'racePctWhite':1, 'racepctblack':0, 'racePctAsian':0, 'racePctHisp':0}

    S = df['race'].map(mapping) #sensitive attribute: S=1 for white, S=0 for non-white
    df = df.drop('race', axis=1)

    X = df.to_numpy() #features

    if as_df: #for comparing with agarwal
        return df, S, y
    else:
        return X, S.to_numpy(), y.to_numpy()


def get_lawschool_data(as_df=False):
    df = pd.read_csv('/Users/deborah/PycharmProjects/FairModelSelection/data/lawschool.csv')
    df = df.dropna()
    y = df['ugpa']  # target: gpa in [0,4]
    df = df.drop('ugpa', axis=1)
    df['gender'] = df['gender'].map({'male': 1, 'female': 0})
    df_bar = df['bar1']
    df = df.drop('bar1', axis=1)
    df['bar1'] = [int(grade == 'P') for grade in df_bar]
    df['race'] = [int(race == 7.0) for race in df['race']]  # setting S=1 for white, S=0 for non-white
    S = df['race'].to_numpy()  # sensitive attribute
    df = df.drop('race', axis=1)
    X = df.to_numpy()  # features

    if as_df:  # for comparing with agarwal
        return df, S, y
    else:
        return X, S, y.to_numpy()

