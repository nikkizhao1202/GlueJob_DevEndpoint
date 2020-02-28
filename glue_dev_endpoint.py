"""
Glue Dev End point helper

Author: Pierre-Hadrien Arnoux

"""

import os
import time
import argparse
import json
import boto3


class GlueDevEndpoint():
    """
    Class implementing a Glue Dev Endpoint helper
    Main capacities:
      - Create
      - Delete
      - Connect

    """

    def __init__(self, config):

        config = json.load(open(config, "r"))

        self.region_name = config["region_name"]
        self.glue_endpoint = "https://glue.{}.amazonaws.com".format(self.region_name)

        self.python_library = ""

        if "python_library" in config:
            self.python_library = config["python_library"]

        self.glue_engine = boto3.client(
            service_name="glue", region_name=self.region_name, endpoint_url=self.glue_endpoint)

        self.dev_endpoint_pub_rsa = open(config["rsa_public"], "r").read().replace("\n", "")

        self.dev_endpoint_private_rsa = config["rsa_private"]

        self.dev_endpoint_name = config["dev_endpoint_name"]
        self.dev_endpoint_role = config["dev_endpoint_role"]

        self.dev_endpoint = None

    def create_dev_endpoint(self):
        """
        Create Glue Dev Endpoint

        """

        self.dev_endpoint = self.glue_engine.create_dev_endpoint(
            EndpointName=self.dev_endpoint_name,
            RoleArn=self.dev_endpoint_role,
            PublicKey=self.dev_endpoint_pub_rsa,
            NumberOfNodes=3,
            ExtraPythonLibsS3Path=self.python_library,
            GlueVersion="1.0",
            Arguments={"GLUE_PYTHON_VERSION": "3"})

    def delete_dev_endpoint(self):
        """
        Delete Glue Dev Endpoint

        """
        self.glue_engine.delete_dev_endpoint(EndpointName=self.dev_endpoint_name)

    def connect_dev_endpoint(self):
        """
        Connect to Glue Dev Endpoint

        """

        done = False

        while not done:

            endpoint = self.glue_engine.get_dev_endpoint(EndpointName=self.dev_endpoint_name)

            status = endpoint["DevEndpoint"]["Status"]

            done = status == "READY"

            if status == "PROVISIONING":
                print("Still provisionning...")
                time.sleep(30)
            elif status == "READY":
                print("Done")
                done = True
            else:
                print("There was an error")
                print(status)

        public_ip = endpoint["DevEndpoint"]["PublicAddress"]

        os.system(
            "ssh -i {} glue@{} -t gluepyspark".format(self.dev_endpoint_private_rsa, public_ip))

if __name__ == "__main__":

    PARSER = argparse.ArgumentParser(description="Glue ETL Dev")
    PARSER.add_argument("--create", action="store_true")
    PARSER.add_argument("--connect", action="store_true")
    PARSER.add_argument("--delete", action="store_true")
    PARSER.add_argument("--config", required=True)

    ARGS = PARSER.parse_args()

    if ARGS.create:
        GlueDevEndpoint(ARGS.config).create_dev_endpoint()
    if ARGS.connect:
        GlueDevEndpoint(ARGS.config).connect_dev_endpoint()
    if ARGS.delete:
        GlueDevEndpoint(ARGS.config).delete_dev_endpoint()
