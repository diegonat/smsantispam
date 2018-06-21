import boto3
import os

FILE_MODEL_NAME = os.environ['FILE_MODEL_NAME']
SOURCE_BUCKET = os.environ['SOURCE_BUCKET']
DESTINATION_BUCKET = os.environ['DESTINATION_BUCKET']



client = boto3.client('s3')

def lambda_handler(event, context):
    # TODO implement
    print event['detail']['requestParameters']['key']
    name = event['detail']['requestParameters']['key'].split("/")
    print name[-1]
    
    if name[-1] == FILE_MODEL_NAME:
            response = client.copy_object(
                Bucket=DESTINATION_BUCKET,
                CopySource={'Bucket': SOURCE_BUCKET, 'Key': event['detail']['requestParameters']['key']},
                Key=FILE_MODEL_NAME
            )
            print response

    else:
        print "Ignore the file"

