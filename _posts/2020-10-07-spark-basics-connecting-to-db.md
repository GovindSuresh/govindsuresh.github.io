---
layout: post
title: Spark - Connecting to Databases
excerpt: "A straightforward guide to connecting to databases via PySpark."
categories: [Spark, Guides]
mathjax: true
comments: true
---

Ingesting and writing to databases is one of the most common workflows when working with big data tools such as Spark. In this post, we will go through the practical steps to connecting to databases using PySpark via the Spark SQL module. We will be focussing on the basics here of connecting in a local Spark environment.

## How does Spark connect to databases?

Spark connects to databases via JDBC (Java Database Connectivity). This is a Java API for connecting to various databases. Importantly for us PySpark users, we don't actually need to write any Java code. All we need to do is download the relevant JDBC drivers to connect to the database we are using. These drivers are analogous to  OBDC (Open Database Connectivity) drivers you may have used previously to connect applications to an RDBMS.

These drivers are just `.jar` files that contain the Java code to manage the connection between Spark and the database.   

The process to connecting to various databases is pretty much the same across the board, you just need to have the relevant JDBC driver for the RDBMS you are trying to work with. 

## Connecting to MySQL

Let's dive right in and go through how to connect to the MySQL database. I assume you already have a working installation of MySQL. I will be using the [Sakila]([https://dev.mysql.com/doc/sakila/en/](https://dev.mysql.com/doc/sakila/en/)) database in my examples. Though you can use any database you have access to. 

In the script, we will read in from the 'actor' and 'film_actor' table within the Sakila database, perform some aggregation and join operations between the two and finally write the output to a new table in the Sakila database. We will breakdown the code that is relevant to reading and writing to the database. 

### Step 1: Download the MySQL JDBC connector

The first thing we need to do is to download the JDBC connector for MySQL. You can do this (here)[[https://dev.mysql.com/downloads/connector/j/](https://dev.mysql.com/downloads/connector/j/)]. Make sure the connector version you download is the one for your operating system. The particular file we are interested in from the downloaded archive is the actual `.jar` file. It will look something like this `mysql-connector-java-8.0.21.jar`.

Technically you can leave the .jar file anywhere. However, I recommend storing all these files in the same location as we will need to specify the path to the driver file whenever we submit a Spark job that needs to connect to the database. You can store them with the other jar files in your Spark installation.

### Step 2: Reading in from the database

It's time to begin writing our python script. 

```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, desc

# Set up SparkSession
spark = SparkSession.builder.appName('read_from_mysql').master('local[*]').getOrCreate()

user = '<your MySQL username>'
pw = '<your MySQL pw>'
use_ssl = 'false' # Needed due to MySql default security
mysql_url = 'jdbc:mysql://localhost:3306/sakila?serverTimezone=EST' #This is the JDBC URI

dbschema = 'sakila'

# read in the actor table
actors = spark.read.format('jdbc') \
            .options(url = mysql_url,
                database= dbschema,
                dbtable = 'actor',
                user=user,
                password = pw).load()

film_actors = spark.read.format('jdbc') \
            .options(url = mysql_url,
                database= dbschema,
                dbtable = 'film_actor',
                user= user,
                password = pw).load()
```

Let's break down the base options that we need to set. Some of these are pretty self-explanatory. You will need to pass in your MySQL username and password. It's important that this account has the relevant permissions to read/write/update the database in question. Remember not to hardcode your login details into the file, especially if storing this on a version control system. 

The next key argument is the URI to connect to the database. The first part URI follows a pretty standard format, for example, the URI for Postgres would start `jdbc:postgresql://`. The next part of the URI is the address of the database. In this case, I'm connecting to a local server. The next part is the database/schema name. 

For MySQL, you may also need to pass in an additional part at the end specifying the server timezone. This takes us to an additional point, you can actually pass most of the options we are setting via the URI itself rather than specifying them as separate options. For example:

```python
"jdbc:mysql://localhost:3306/sakila?user=root&password=pw&useSSL=false&serverTimezone=EST";
```

Either way is acceptable, the first way is perhaps easier when reading in the details from a config file. You could still build the full URI string using Python formatted strings in this scenario. 

The final stage here is to read in from the database, we will be reading the table into a Spark dataframe object. We pass in the options as arguments into the `.options()` method. As with anything in Spark, there are multiple ways to do this, we could have  passed these in separate `.options()` methods:

```python
actors = spark.read.format("jdbc").option("url", "jdbc:mysql://localhost:3306/sakila") \
            .option("dbtable", "actor").option("user", "root") \
            .option("password", "pw") \
            .option("useSSL", "false") \
            .option("serverTimezone", "EST").load()
```

### Step 3: Carry out your operations

We are going to perform some transformations to the data. The actual operations are not important here, we are more interested in reading and writing. The code for the transformations I have applied is below:

```python
# Aggregate film_actors table first to get counts then join actors names
film_actors = film_actors.groupBy('actor_id').count().withColumnRenamed('count','number_of_films')

# Joining the two tables on actor_id, and ordering by the number of films descending. 
new_table = film_actors.join(actors, on='actor_id', how='inner').select(film_actors['actor_id'],
                        actors['first_name'], actors['last_name'], film_actors['number_of_films']) \
                        .sort(desc('number_of_films'))
```

### Step 4: Write back to the database

To write back to the database we will take advantage of the Spark JDBC writer.  In our example, we will create a new table in the Sakila database named 'film_counts'. 

```python
# Properties to connect to the database, the JDBC driver is part of our pom.xml
prop = {"user": user, "password": pw}

# Write in a table called ch02
new_table.write.jdbc(mode='overwrite', url=mysql_url, table="film_counts", properties=prop)
```

In the code snippet above, you can see we pass in some properties related to the user account as a dictionary. We then specify the URL to the MySQL server and also specify the table to write to in the `.write.jdbc()` method itself. 

The most important option here is the `mode`. This governs how the JDBC writer object behaves when modifying the database. Currently, these 4 modes are supported:

- `append`: Append contents of this  DataFrame to an existing table.
- `overwrite`: Overwrite existing data. For this scenario, new tables will be created unless truncate option is used.
- `ignore`: Silently ignore this operation if data already exists.
- `error` (default case): Throw an exception if data already exists.

You may be wondering why we've used `new_table.write.jdbc()` and passed in a dictionary rather than `new_table.write.format('jdbc')` and used the `.options()` method like we did when we read in the tables. The reason for this is that in my experience when I use the latter method when writing tables, I seem to frequently run into errors relating to not being able to run `CREATE TABLE AS SELECT` commands. I'm not sure why this is the case, Spark can be temperamental!

### Step 5: Submit the job

Our code is ready to go. We are going to submit the job via `spark-submit` on the command line. So go ahead and open up your terminal and enter the below:

```bash
spark-submit --driver-class-path /path/to/mysqljdbcdriver /path/to/pythoncode.py 
```

The important thing to note here is that we need to pass in the location of our MySQL JDBC driver that we downloaded as the `--driver-class-path` argument. So it's a good idea to save these drivers somewhere secure and also easy to access. You could also assign the driver path to some environment variable, it's up to you. Please also note that the MySQL server needs to be running and reachable.

If the script has appeared to execute fully, go ahead and check the new table has been added to the Sakila database.

### A quick note on dialects

Spark uses what's known as a dialect to communicate with the RDBMS. I like to think of this as a translation table that allows the Spark JDBC writer to translate how the database handles things like data types to Spark data types. These dialects are therefore unique to each RDBMS. As of Spark 3.0.0, the following dialects are supported:

- MySQL
- PostgreSQL
- SQL Server
- IBM Db2
- Apache Derby
- Oracle
- Teradata Database

If you want to be connect with other databases, you will need to define your own custom dialect which is a topic we may cover in a later post. 

## Next Steps

Congrats! You have successfully read and written to an RDBMS. The process for other popular database software such as Postgres is very similar. You just need to make sure you have the Postgres JDBC driver downloaded. A few options here and there will be different but the base principles are the same. 

If you have any questions, leave a comment below. There are some further considerations we have to make when doing this via an actual cluster deployment. These will be covered in a later post, so stay tuned! 🙏🏽