import configparser
import os
from pyspark.sql import SparkSession
from datetime import datetime

from pyspark.sql.functions import udf, col, to_timestamp
from pyspark.sql.functions import year, month, dayofmonth, hour, weekofyear, date_format
from pyspark.sql.window import Window


config = configparser.ConfigParser()
config.read('dl.cfg')

os.environ['AWS_ACCESS_KEY_ID']=config['Udacity_cc']['AWS_ACCESS_KEY_ID']
os.environ['AWS_SECRET_ACCESS_KEY']=config['Udacity_cc']['AWS_SECRET_ACCESS_KEY']


def create_spark_session():
    spark = SparkSession \
        .builder \
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:2.7.0") \
        .getOrCreate()
    return spark


def process_song_data(spark, input_data, output_data):
    """
    Description: This function loads song_data from S3 and processes it by extracting the songs and artist tables
                and then again loaded back to S3
        
    Parameters:
            spark       : Spark Session
            input_data  : location of song_data json files with the songs metadata
            output_data : S3 bucket were dimensional tables in parquet format will be stored
    """
    # get filepath to song data file
    song_data = input_data + "song-data/A/A/A/*.json"
    
    # read song data file
    df = spark.read.json(song_data)

    # extract columns to create songs table
    '''remove the duplicates'''
    songs_table = df.select(['song_id', 'title', 'artist_id', 'year', 'duration']).dropDuplicates(['song_id'])
   
    # write songs table to parquet files partitioned by year and artist
    songs_table.write.parquet(output_data + 'songs.parquet', partitionBy=['year','artist_id'])
    
    # extract columns to create artists table
    artists_table = df.select(['artist_id', 'artist_name', 'artist_location', 'artist_latitude', 'artist_longitude']).dropDuplicates(["artist_id"])
    
    # write artists table to parquet files
    artists_table.write.parquet(output_data + 'artists.parquet')


def process_log_data(spark, input_data, output_data):
    """
        Description: This function loads log_data from S3 and processes it by extracting the songs and artist tables
                    and then again loaded back to S3. Also output from previous function is used in by spark.read.json command
        
        Parameters:
            spark       : Spark Session
            input_data  : location of log_data json files with the events data
            output_data : S3 bucket were dimensional tables in parquet format will be stored
            
    """
    # get filepath to log data file
    log_data = input_data + "log-data/*/*/*.json"

    # read log data file
    log_df = spark.read.json(log_data)
    
    # filter by actions for song plays
    log_df = log_df.where('page = "NextSong"')
    
    # extract columns for users table    
    users_table = log_df.select(['userId', 'firstName', 'lastName', 'gender', 'level']).dropDuplicates(['userId'])
    
    # write users table to parquet files
    users_table.write.parquet(output_data + 'users.parquet')


    # create timestamp column from original timestamp column
    get_timestamp = udf(lambda x: str(int(int(x)/1000)))
    log_df = log_df.withColumn('timestamp', get_timestamp(log_df.ts))
    
    # create datetime column from original timestamp column
    get_datetime = udf(lambda x: str(datetime.fromtimestamp(int(x) / 1000.0)))
    log_df = log_df.withColumn("datetime", get_datetime(log_df.ts))
    
    # extract columns to create time table
    time_table = log_df.select(
                      col('datetime').alias('start_time'),
                      hour('datetime').alias('hour'),
                      dayofmonth('datetime').alias('day'),
                      weekofyear('datetime').alias('week'),
                      month('datetime').alias('month'),
                      year('datetime').alias('year') ).dropDuplicates(['start_time'])
    
    
    # write time table to parquet files partitioned by year and month
    time_table.write.parquet(output_data + 'time.parquet', partitionBy=['year','month'])

    # read in song data to use for songplays table
    song_data = input_data + "song-data/A/A/A/*.json"
    
    song_df = spark.read.json(song_data)

    song_df.createOrReplaceTempView("song")
    log_df.createOrReplaceTempView("log")
    
   
    # extract columns from joined song and log datasets to create songplays table 
    songplays_table = spark.sql("""
                                select 
                                l.timestamp as start_time,
                                l.userId,
                                l.level,
                                s.song_id,
                                s.artist_id,
                                l.sessionId,
                                l.location, 
                                l.userAgent, 
                                year(l.timestamp) as year,
                                month(l.timestamp) as month 
                                from log as l 
                                inner join song as s on l.song = s.title
                                """)
    
    # write songplays table to parquet files partitioned by year and month
    songplays_table.write.parquet(output_data + 'songplays.parquet',partitionBy=['year', 'month'])


def main():
    """
    Run the ETL to process the song_data and the log_data files
    """
    spark = create_spark_session()
    input_data = "s3a://udacity-dend/"
    output_data = "s3a://udacity-cc/"
    
    process_song_data(spark, input_data, output_data)    
    process_log_data(spark, input_data, output_data)


if __name__ == "__main__":
    main()