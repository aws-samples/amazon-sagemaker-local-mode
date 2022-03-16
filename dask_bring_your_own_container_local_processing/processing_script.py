import os
import sys
import logging
from dask.distributed import Client
import dask.bag as db
import json


processed_data_path = '/opt/ml/processing/processed_data'


def main():
    print("Processing Started")

    # Convert command line args into a map of args
    args_iter = iter(sys.argv[1:])
    args = dict(zip(args_iter, args_iter))
    scheduler_ip = sys.argv[-1]
    print(f"scheduler_ip: {scheduler_ip}")

    # Start the Dask cluster client
    try:
        print("initiating client")
        client = Client("tcp://{ip}:8786".format(ip=scheduler_ip))
        print("Cluster information: {}".format(client))
    except Exception as err:
        logging.exception(err)

    print(f"Received arguments {args}")

    if "site_uri" in args:
        print(f"Processing web site JSON: {args['site_uri']}")
        filenames = (db.read_text(args['site_uri'])
                     .map(json.loads)
                     .pluck('name')
                     .compute())

        filenames = ['https://archive.analytics.mybinder.org/' + fn for fn in filenames]
        print(f"Total filenames: {len(filenames)}")
        print(f"Sample filenames found: {filenames[:5]}")

        output_file = os.path.join(processed_data_path, "filenames_in_json.txt")
        print(f'Writing output file: {output_file}')
        with open(output_file, 'w') as outfile:
            outfile.write(json.dumps(filenames))
    else:
        print("No `site_uri` parameter - doing nothing")

    print("Processing Complete")

    print(client)
    sys.exit(os.EX_OK)

if __name__ == "__main__":
    main()