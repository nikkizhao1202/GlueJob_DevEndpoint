import sys
from awsglue.transforms import *
from datetime import datetime 
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from pre_processing.pre_processing import PreProcessor


## @params: [JOB_NAME]
args = getResolvedOptions(sys.argv, ['JOB_NAME',"PADDING_SIZE","MAX_DICTIONARY_SIZE"])


sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)


my_pre_processor = PreProcessor(padding_size=int(args["PADDING_SIZE"]), max_dictionary_size=int(args["MAX_DICTIONARY_SIZE"]))

time_stamp = datetime.now().strftime("%Y-%b-%d-%H-%M-%S")

## @type: DataSource
## @args: [database = "tweets", table_name = "train", transformation_ctx = "datasource0"]
## @return: datasource0
## @inputs: []
datasource0 = glueContext.create_dynamic_frame.from_catalog(database = "tweets", table_name = "train", transformation_ctx = "datasource0")
## @type: ApplyMapping
## @args: [mapping = [("tweet", "string", "tweet", "string"), ("sentiment", "string", "sentiment", "string")], transformation_ctx = "applymapping1"]
## @return: applymapping1
## @inputs: [frame = datasource0]
applymapping1 = ApplyMapping.apply(frame = datasource0, mappings = [("tweet", "string", "tweet", "string"), ("sentiment", "string", "sentiment", "string")], transformation_ctx = "applymapping1")

## @type: Map
## @args: [f = map_function, transformation_ctx = "mapping1"]
## @return: mapping1
## @inputs: [frame = applymapping1]
def map_function(dynamicRecord):
    tweet = dynamicRecord["tweet"]
    features = my_pre_processor.pre_process_text(tweet)
    dynamicRecord["features"] = features
    return dynamicRecord

mapping1 = Map.apply(frame = applymapping1, f = map_function, transformation_ctx = "mapping1")



## @type: DataSink
## @args: [connection_type = "s3", connection_options = {"path": "s3://twitter-text/result_job"}, format = "json", transformation_ctx = "datasink2"]
## @return: datasink2
## @inputs: [frame = mapping1]
datasink2 = glueContext.write_dynamic_frame.from_options(frame = mapping1, connection_type = "s3", connection_options = {"path": "s3://twitter-text/result_job/time={}".format(time_stamp)}, format = "json", transformation_ctx = "datasink2")
job.commit()