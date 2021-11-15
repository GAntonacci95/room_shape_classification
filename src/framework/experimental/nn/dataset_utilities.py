import numpy as np
import pandas as p


def retain_class_field_labels(df: p.DataFrame, field: str, labels: [str]) -> p.DataFrame:
    return df[df[field].isin(labels)]


def to_uniform_class_field_distro(df: p.DataFrame, field: str) -> p.DataFrame:
    # let's go for a more uniform classes distribution
    a = df[field].value_counts()
    head_to = int(np.round(a.min() / 100) * 100)
    np.random.seed(11)
    for c in a[a > head_to].index.values:
        df = df.drop(np.random.choice(df[df[field] == c].index.values,
                                        a[c] - head_to,
                                        replace=False))
    # now the set is almost uniform
    return df


def retain_field_as_y(df: p.DataFrame, field: str) -> p.DataFrame:
    if field not in df.columns:
        raise Exception("Field not found!")
    ret = df[['X', field]]
    ret.columns = ['X', 'y']
    return ret


def split_tvt(df: p.DataFrame):
    train, test, val = np.array_split(df, [int(df.shape[0] * 0.7), int(df.shape[0] * 0.8)])
    return train, val, test


def id_recode_class_field(df: p.DataFrame, infield: str, classes: [str], outfield: str):
    for i in range(len(classes)):
        df.loc[df[infield] == classes[i], infield] = i
    df.columns = [colname if colname != infield else outfield for colname in df.columns]
    return df


def curr_cls_vs_oth(df: p.DataFrame, infield: str, classes: [str]):
    for i in range(len(classes)):
        sec_field = "class_label_{}".format(i)
        df[sec_field] = df[infield]
        df.loc[df[sec_field] != classes[i], sec_field] = "other"
    return df


def transform_scalar_field(df: p.DataFrame, infield: str, application, outfield: str):
    df.loc[:, infield] = application(df[infield].astype(np.float))
    df.columns = [colname if colname != infield else outfield for colname in df.columns]
    return df


def select_by_scalar_field_range(df: p.DataFrame, field: str, begin: float, end: float) -> p.DataFrame:
    return df[df[field].between(begin, end)]


def select_by_field_name(df: p.DataFrame, infield: str) -> p.DataFrame:
    return df[["X", infield]]


def get_field_frequencies(df: p.DataFrame, field: str):
    distro = df[field].value_counts().sort_index()
    return np.array(distro.index), np.array(distro.values)


def get_field_distro(df: p.DataFrame, field: str):
    uq, cnts = get_field_frequencies(df, field)
    return uq, cnts / len(df)


def show_field_distro(df: p.DataFrame, field: str):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    df[field].plot.kde()
    plt.show()
    plt.close(fig)
    return


def subsets_handler(handler, dfs: [p.DataFrame], further_params: []) -> []:
    return [handler(df, *further_params) for df in dfs]
