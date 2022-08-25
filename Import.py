import pandas as pd
import numpy as np

dfGenomeScores = pd.read_csv('genome-scores.csv', delimiter=',')
dfGenomeTags = pd.read_csv('genome-tags.csv', delimiter=',')
dfMovies = pd.read_csv('movies.csv', delimiter=',')
dfRatings = pd.read_csv('ratings.csv', delimiter=',', parse_dates=['timestamp'])

dfMoviesWithTag = dfGenomeScores.join(dfGenomeTags.set_index('tagId'), on='tagId')
dfMoviesWithTag.head(5)

def reduce_memory_usage(df, verbose=True):
    numerics = ["int8", "int16", "int32", "int64", "float16", "float32", "float64"]
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if (
                        c_min > np.finfo(np.float16).min
                        and c_max < np.finfo(np.float16).max
                ):
                    df[col] = df[col].astype(np.float16)
                elif (
                        c_min > np.finfo(np.float32).min
                        and c_max < np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose:
        print(
            "Mem. usage decreased to {:.2f} Mb ({:.1f}% reduction)".format(
                end_mem, 100 * (start_mem - end_mem) / start_mem
            )
        )
    return df

val_tags = dfMoviesWithTag.tag
tags_list = []
for t in val_tags:
    if t not in tags_list:
        tags_list.append(t)


reduce_memory_usage(dfMoviesWithTag)
reduce_memory_usage(dfMovies)

for index, row in dfMovies.iterrows():
    movieId = row.movieId
    title = row.title
    try:
        #TAGS
        tags_df_select = dfMoviesWithTag[dfMoviesWithTag.movieId == movieId]  #Otteniamo i vari tag per un singolo film
        for i, r in tags_df_select.iterrows():
            tag = r.tag
            score = r.relevance
            dfMovies.loc[index, tag] = score


        #VALUTAZIONI
        ratings_df_select = dfRatings[dfRatings.movieId == movieId]
        average_rating = np.mean(ratings_df_select.rating)
        dfMovies.loc[index, 'AVG_RATING'] = average_rating

    except Exception:
         pass
dfMoviesWithTag = dfMoviesWithTag.iloc[: , 1:]
dfMoviesWithTag = dfMoviesWithTag.iloc[: , 1:]
dfMoviesWithTag = dfMoviesWithTag.iloc[: , 1:]
dfMoviesWithTag.to_csv('ImportDataset1.csv')