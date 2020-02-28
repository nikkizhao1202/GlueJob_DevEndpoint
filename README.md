# AWS Glue Create DevEndpoint

## OverView
This repository holds tools to help interacting with AWS Glue Dev Endpoint

## Generating an RSA key

On Mac or Linux

```
ssh-keygen -t rsa
```

## Config

To use the helper, you should write a `config.json`

```
    {
        "region_name": "region where to deploy the endpoint us-east-1 for example",
        "rsa_public": "absolute path to the public ssh key used to connect to Glue",
        "rsa_private": "absolute path to the private ssh key used to connect to Glue",
        "dev_endpoint_name": "enpoint name",
        "dev_endpoint_role": "endpoint role. should be a glue role. starts with arn:aws:iam::..."
    }
```

## Functionalities

### Create
```
python3 glue_dev_endpoint.py --config <path_to_config.json> --create
```
### Connect
```
python3 glue_dev_endpoint.py --config <path_to_config.json> --connect
```
### Delete
```
python3 glue_dev_endpoint.py --config <path_to_config.json> --delete
```
