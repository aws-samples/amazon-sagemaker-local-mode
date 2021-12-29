import os
import sys
import delta_sharing


profile_path = '/opt/ml/processing/profile/'
processed_data_path = '/opt/ml/processing/processed_data'


def main():
    print("Processing Started")

    # Convert command line args into a map of args
    args_iter = iter(sys.argv[1:])
    args = dict(zip(args_iter, args_iter))

    print('Received arguments {}'.format(args))

    profile_files = [os.path.join(profile_path, file) for file in os.listdir(profile_path)]
    if len(profile_files) == 0:
        raise ValueError(
            (
                    "There are no files in {}.\n"
                    + "This usually indicates that the channel ({}) was incorrectly specified,\n"
                    + "the data specification in S3 was incorrectly specified or the role specified\n"
                    + "does not have permission to access the data."
            ).format(profile_path)
        )

    profile_file = profile_files[0]
    print(f'Found profile file: {profile_file}')

    # Create a SharingClient
    client = delta_sharing.SharingClient(profile_file)
    table_url = profile_file + "#delta_sharing.default.owid-covid-data"

    # Load the table as a Pandas DataFrame
    print('Loading owid-covid-data table from Delta Lake')
    data = delta_sharing.load_as_pandas(table_url)
    print(f'Data shape: {data.shape}')

    # Aggregate total_cases per location
    cases_per_location = data.groupby(['location'])['total_cases'].sum()
    print(f'cases_per_location\n{cases_per_location}\n')

    output_file = os.path.join(processed_data_path,'total_cases_per_location.csv')
    print(f'Writing output file: {output_file}')
    cases_per_location.to_csv(output_file)

    print("Processing Complete")

if __name__ == "__main__":
    main()