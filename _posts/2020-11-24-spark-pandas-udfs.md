---
layout: post
title: Pandas UDFs in PySpark 3
excerpt: "Implementing and understanding Pandas UDFs in PySpark 3"
categories: [Spark, Guides]
mathjax: true
comments: true
---

Pandas UDFs are a unique feature of PySpark that provide significant improvements in performance over regular UDFs. They have been present since Spark 2.3. However, with PySpark 3.0.1, significant changes have been made to how Pandas UDFs are defined and work.

Let's run through these changes and look at some of the UDFs you can use and when to use them. In particular, we will focus on the series-to-series Pandas UDF, as well as the grouped map operation, given these are probably the ones you will likely use most regularly.

## Pandas UDFs vs Regular UDFs:

In case you are not familiar/forgotten the benefits provided by a Pandas UDF over a regular UDF, I'll recap the key points here:

- **Apache Arrow:** Regular Python UDFs incur significant performance costs from data serialization between the JVM and the Python process executing the function. Pandas UDFs leverage [Apache Arrow](https://arrow.apache.org/), which is a data serialization framework. The key takeaway here is that Arrow reduces the data serialization cost dramatically.

- **Vectorized operations:** Regular UDFs carry our element by element (row by row) operations across a column. However, a Pandas UDF benefits from the speed-up offered by vectorized operations when using Pandas data structures. A Pandas UDF will serialized and operate on an entire partition of a column at once before stitching the transformed partitions back together

According to Databricks, a Pandas UDFs can be up to 100x(!!!!) faster than an equivalent standard Python UDF.


<img src="/assets/images/pandas_vs_udf.png" alt="pandas-pyudfs.png" title="Python UDFs Vs Pandas UDFs" height=600 width=600>



The key takeaway from the diagram above is how the Pandas UDF splits the data into partitions and will apply the transformation on the entire partition in parallel (vectorization). A standard Python UDF is applied on a *row-by-row* basis.

## Types of Pandas UDFs

There are two main ways of implementing Pandas UDFs:

1. Pandas UDFs via the `pyspark.sql` module.
2. The Pandas Functions API

The former is the traditional location for Pandas UDFs. However, several Pandas UDFs, such as the map and grouped map operations, have been shifted into the new Pandas Functions API.

### Pandas UDFs via pyspark.sql:

There are currently 4 supported types of Pandas UDF:

- ***Series to series UDFs -*** These are UDFs that take in a series and return a series of equal length.

- ***Iterator of series to iterator of series -*** The same concept as above just takes in an iterator of a series and returns an iterator of a series. Primarily used to get benefit from fetching.

- ***Iterator of multiple series to iterator of series -***  The same concept as above just takes in an iterator of numerous series and returns an iterator of series. Primarily used to get benefit from fetching.

- ***Series to scalar (a single value) -*** Takes in a Series and returns a single value. Very much like a standard aggregation function. The intention is for this to be used after a .groupBy() operation.

With the advent of PySpark 3, we now specify which type of Pandas UDF we are implementing via Python type hints. If you are unfamiliar with type hints, take a look at this [excellent guide](https://realpython.com/python-type-checking/). The kind of hint provided for the parameters and output of the function defines which type of pandas UDF is being used.

Let's take a look at the series to series UDF. We won't go into the detail for all the UDF types as they are mostly just variations of the series to series UDF.

#### Series to Series UDF:

This is probably the most common type of UDF you will apply. In this scenario, our function will take in a column of type `pd.Series` and return a column of type pd.Series of the same length as the input column. Internally, PySpark will take the column, split it into batches, call the function on each batch, and concatenate the results. Let's look at a straightforward example where we create a new series based on some conditional logic.

```python
import pandas as pd
import numpy as np 
from pyspark.sql.functions import pandas_udf, col

# We still need to apply the decorator and specify the datatype of the input col

@pandas_udf('long')
def high_low(col: pd.Series) -> pd.Series:
		
	# Notice how we wrap the np.where() statement in a pd.Series().
	# The UDF MUST return a Pandas Series.
    new_col = pd.Series(np.where(col <= 70, 'Low','High')
		
		return new_col 

# We can apply this function in a variety of ways. 
# In this scenario df is a spark df with student grades

df = df.withColumn('high_low', high_low(col('grades'))

```

 A key point to note here is that the UDF ***must*** return a `Series` object. This means we cannot return a `ndarray` despite their similarities, which is why we wrapped our `np.where()` with the `pd.Series()` constructor.  

### Pandas Function API

For Spark 3.0, the developers decided to move 3  Pandas UDF types into their own section known as the Pandas Function API. The 3 types are:

- Grouped map operations
- Map operations
- Co-grouped map operations

The difference between these operations and what we saw earlier is that we can directly apply a Python function that takes and outputs pd.DataFrames onto a PySpark DataFrame. This may not make much sense right now, so let's go ahead and dive into an example to drive home the differences!

Of these three, I'd say the grouped map operation is the most useful and easy to grasp, so let's look at this first.

#### Grouped Map

Consider the following scenario:

- You've collected blood pressure data from multiple groups of people and entered it all into a dataframe.
- You want a new column, which is the median from each group subtracted from each row's blood pressure value.
- The final table should have the same number of rows as the initial dataframe.

This is a grouped map operation. You may have also heard this kind of process being called a Split-Apply-Combine operation. The idea here is that we split the data into distinct groups, apply some function to some elements of the rows of that group, and finally combine the groups back together. What is really important to note is that we are not **aggregating** as part of the grouped map operation. The returned table will have the **same** number of rows as the initial table. The diagram below visualizes the process.

Compare the results table to a standard aggregation with grouping. In this kind of process, the number of rows in the final table would reduce to the number of unique groups.

Let's take a look at the code to implement this process.

<img src="/assets/images/split_apply_combine.png" alt="split-apply-combine.png" title="Split-Apply-Combine Operations" height=800 width=600>

Lets have a look at some code:

```python
import pandas as pd
import numpy as np 
from pyspark.sql.functions import pandas_udf, col

# Lets create an example dataframe for the scenario above
bp_df = spark.createDataFrame(
    [(1, 1, 105.0), (2, 1, 100.0), (3, 2, 128.0), (4, 2, 119.0), (5, 1, 108.0),
	(6, 2, 122.0)],
    ("id", "group", "bp"))

# +---+------+-------+
# | id| group|     bp|
# +---+------+-------+
# |  1|     1|  105.0|
# |  2|     1|  100.0|
# |  3|     2|  128.0|
# |  4|     2|  119.0|
# |  5|     1|  108.0|
# |  6|     2|  122.0|
# +---+------+-------+

# Lets define the function to calculate the mean of our group
def median_subtract(df: pd.DataFrame) -> pd.DataFrame:
		df['new_col'] = df['bp'] - df['bp'].median()
		return df

# Apply to our dataframe
bp_df.groupBy('group').applyInPandas(median_subtract, \
									schema="id long, group int, bp float, new_col float")

### RESULTS:
# +---+------+-------+--------+
# | id| group|     bp| new_col|
# +---+------+-------+--------+
# |  1|     1|  105.0|       0|
# |  2|     1|  100.0|      -5|
# |  3|     2|  128.0|       6|
# |  4|     2|  119.0|      -3|
# |  5|     1|  108.0|       3|  
# |  6|     2|  122.0|       0|
# +---+------+-------+--------+
```

In this example, note how we don't have to decorate our function. Technically we don't even need type hints here. Instead, we use a specific PySpark method, `.applyInPandas()`, to apply our function. As the name would suggest, grouped map operations are chained onto the end of a `groupBy()` statement.

Each operation from the Pandas Functions API works similarly. We don't need type hints or decorators for our Python Function. We just need to use the correct method for the operation we want to do.

- Grouped Map = .applyInPandas()
- Map = .mapInPandas()
- Co-grouped Map = .cogroup().applyInPandas()

### Factors to consider

One thing to always keep in mind is that you should **always** use native PySpark functions wherever possible. The speed-ups offered by Pandas UDFs over regular UDFs pale in comparison to the speed of native functions, which do not have to go through the process of data serialization and transfer.

In addition to this, Catalyst, the Spark optimizer, treats all UDFs as black-boxes. Therefore it can only optimize the script before and after a UDF. It cannot optimize your script as a whole. So it's beneficial to leave your UDF until the end whenever possible.

### Wrap up

It's not always easy to work out *when* you need a UDF or Pandas UDF. Much of this will come down to your knowledge and comfort with existing PySpark functionality. It may be tempting to use these UDFs frequently, but this will come at a performance cost due to the reasons mentioned earlier. 

If you want some inspiration, take a look at this [cool example](https://towardsdatascience.com/scalable-python-code-with-pandas-udfs-a-data-science-application-dd515a628896) of implementing Scikit-learn models on different groups of large datasets via the grouped map operation. The code uses the old method of defining PySpark Pandas UDFs, but you should be able to follow along.