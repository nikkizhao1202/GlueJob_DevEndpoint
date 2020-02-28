# Glue Dev Endpoint Tutorial

## Step 1: Create a Glue Job

In Glue create a Job, name it and select the Glue IAM role.
Select the Glue catalog table your dataset is at.
Select change schema as the transform.
As a target, select create table, data store S3, format JSON, and a path.
In the mapping section, remove all columns that you won't be using.
Save the job to get access to the script.
To be on the safe side, run the job as is an watch that it runs.

## Step 2: Create a Glue Dev Endpoint

Edit the config to add the rsa key you created, enter the region you are using, add a name for your endpoint and add the Glue Role you used in the Glue Job Wizzard. The role should be something like "arn:aws:iam::<account_number>/role/<your_role>"

Create and connect to the endpoint using the `glue_dev_endpoint.py` script.

## Step 3: Exectute your job in the interactive shell

The dev endpoint shell is a python3 shell with a Spark backend.
You should be able to copy / paste the entire script that Glue generated into it.

Hitting the following error messages is not problematic:

Copy pasting `args = getResolvedOptions(sys.argv, ['JOB_NAME'])` will produce
```
awsglue.utils.GlueArgumentError: the following arguments are required: --JOB_NAME
```
as we are not using Glue directly. So no arguments are necessary.

Copy pasting `sc = SparkContext()` will produce
```
ValueError: Cannot run multiple SparkContexts at once; existing SparkContext(app=PySparkShell, master=yarn) created by <module> at /usr/lib/spark/python/pyspark/shell.py:40 
```
but Spark is already initialized.

Copy pasting `job.init(args['JOB_NAME'], args)` will produce
```
NameError: name 'args' is not defined
```
but again we are not using arguments

Reading the first dataset might produce
```
2020-02-08 23:11:19,216 ERROR [Thread-5] util.UserData (UserData.java:getUserData(70)) - Error encountered while try to get user data
java.io.IOException: File '/var/aws/emr/userData.json' cannot be read
```
This error doesn't seem to prevent a job from running

While the shell is running, some network issue may arrise:
```
2020-02-08 23:13:09,732 WARN  [SparkUI-41] server.HttpChannel (HttpChannel.java:handle(499)) - //ip-172-32-91-107.us-east-2.compute.internal:4040/api/v1/applications/application_1581203309398_0001?proxyapproved=true
java.lang.NoSuchMethodError: javax.servlet.http.HttpServletRequest.isAsyncStarted()Z
```
```
2020-02-08 23:13:09,732 WARN  [SparkUI-41] servlet.ServletHandler (ServletHandler.java:doHandle(632)) - /api/v1/applications/application_1581203309398_0001
java.lang.NullPointerException
```
These errors don't seem to prevent a job from running

A job running sucessfully will produce such kind of log:
```
[Stage 0:========>                                             (425 + 8) / 2822]