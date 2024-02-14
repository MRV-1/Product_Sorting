###################################################
# Sorting Products
###################################################

###################################################
# Application : Course Sorting
###################################################
import pandas as pd
import math
import scipy.stats as st
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

df = pd.read_csv(r"datasets\product_sorting.csv")
print(df.shape)
df.head(10)

# The comment count here is the sum of the 5,4,3,2,1 points of the dataset. But the data has been tampered with a bit and the fractions have been removed, so it may not be the sum when you drill down
# The aim here is to examine what problems are encountered when trying to rank these products.
# As a result of these examinations, we will develop a special ranking approach for ourselves and we will handle the scientific form of this with the statistical method.

####################
# Sorting by Rating
####################


df.sort_values("rating", ascending=False).head(20)

####################
# Sorting by Comment Count or Purchase Count
####################

# When ranked by number of purchases, I may not want to put bad courses up even if the number of purchases is high, I may want the user to somehow go for courses that they will be satisfied with.
# Getting the rating and reviews right is an important topic, but it can be a business decision to highlight products that might be good in some way when ranked with user satisfaction in mind.

df.sort_values("purchase_count", ascending=False).head(20)
df.sort_values("commment_count", ascending=False).head(20)

####################
# Sorting by Rating, Comment and Purchase
####################
# We aim to standardize all three metrics and be able to sort at the same time.

df["purchase_count_scaled"] = MinMaxScaler(feature_range=(1, 5)). \
    fit(df[["purchase_count"]]). \
    transform(df[["purchase_count"]])

df.describe().T

df["comment_count_scaled"] = MinMaxScaler(feature_range=(1, 5)). \
    fit(df[["commment_count"]]). \
    transform(df[["commment_count"]])


# The weights of each variable were created differently

(df["comment_count_scaled"] * 32 / 100 +
 df["purchase_count_scaled"] * 26 / 100 +
 df["rating"] * 42 / 100)


def weighted_sorting_score(dataframe, w1=32, w2=26, w3=42):
    return (dataframe["comment_count_scaled"] * w1 / 100 +
            dataframe["purchase_count_scaled"] * w2 / 100 +
            dataframe["rating"] * w3 / 100)


df["weighted_sorting_score"] = weighted_sorting_score(df)
df.sort_values("weighted_sorting_score", ascending=False).head(20)

# the values obtained in this table are reliable, this result was obtained by weighting 3 different constructs

df[df["course_name"].str.contains("Veri Bilimi")].sort_values("weighted_sorting_score", ascending=False).head(20)
# when we have more than one factor that can affect the sorting, the variables should be brought into the same range, then sorting is done by weighting if desired, or by equal weighting if desired


####################
# Bayesian Average Rating Score
####################

# Sorting Products with 5 Star Rated
# Sorting Products According to Distribution of 5 Star Rating

# Can a ranking be made by considering ratings from different perspectives or by focusing only on ratings?

# bayesian_average_rating calculates a weighted probabilistic average over the distribution of scores.
# this method calculates an average rating using the distribution of the individual scores for each course
# can then be sorted accordingly (in which case there are likely to be some problems) or a hybrid approach can be developed
# bayesian_average_rating is a probabilistic rating calculation

def bayesian_average_rating(n, confidence=0.95):
    if sum(n) == 0:
        return 0
    K = len(n)
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    N = sum(n)
    first_part = 0.0
    second_part = 0.0
    for k, n_k in enumerate(n):
        first_part += (k + 1) * (n[k] + 1) / (N + K)
        second_part += (k + 1) * (k + 1) * (n[k] + 1) / (N + K)
    score = first_part - z * math.sqrt((second_part - first_part * first_part) / (N + K + 1))
    return score


df.head()

# The star expressions seen in df will be entered in place of the expression n


df["bar_score"] = df.apply(lambda x: bayesian_average_rating(x[["1_point", "2_point", "3_point","4_point", "5_point"]]), axis=1)


df.sort_values("weighted_sorting_score", ascending=False).head(20)
df.sort_values("bar_score", ascending=False).head(20)

# According to the initial ranking according to the rating, in this ranking, it seems as if some courses have risen very high, this is because this method focuses on the distribution of points.


df[df["course_name"].index.isin([5, 1])].sort_values("bar_score", ascending=False)



####################
# Hybrid Sorting: BAR Score + Other Factors
####################

# Rating Products
# - Average
# - Time-Based Weighted Average
# - User-Based Weighted Average
# - Weighted Rating
# - Bayesian Average Rating Score

# Sorting Products
# - Sorting by Rating
# - Sorting by Comment Count or Purchase Count
# - Sorting by Rating, Comment and Purchase
# - Sorting by Bayesian Average Rating Score (Sorting Products with 5 Star Rated)
# - Hybrid Sorting: BAR Score + Other Factors



def hybrid_sorting_score(dataframe, bar_w=60, wss_w=40):
    bar_score = dataframe.apply(lambda x: bayesian_average_rating(x[["1_point", "2_point", "3_point", "4_point", "5_point"]]), axis=1)
    wss_score = weighted_sorting_score(dataframe)
    return bar_score*bar_w/100 + wss_score*wss_w/100



df["hybrid_sorting_score"] = hybrid_sorting_score(df)
df.sort_values("hybrid_sorting_score", ascending=False).head(20)


