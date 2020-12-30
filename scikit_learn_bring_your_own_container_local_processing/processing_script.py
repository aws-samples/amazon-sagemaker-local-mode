import os
import sys
from datetime import datetime

input_data_path = '/opt/ml/processing/input_data/'
processed_data_path = '/opt/ml/processing/processed_data'


def main():
    print("Processing Started")

    # Convert command line args into a map of args
    args_iter = iter(sys.argv[1:])
    args = dict(zip(args_iter, args_iter))

    print('Received arguments {}'.format(args))
    print('Reading input data from {}'.format(input_data_path))

    print("Got Args: {}".format(args))

    input_files = [file for file in os.listdir(input_data_path) if file.endswith('.' + 'txt')]
    print('Available input text files: {}'.format(input_files))

    if args['job-type'] == 'word-count':
        print('Word Count Job Type Started')
        total_words = 0
        for input_file in input_files:
            file = open(os.path.join(input_data_path, input_file), 'r')
            data = file.read()
            words = len(data.split())
            print('Detected {} words in {} file'.format(words, input_file))
            total_words = total_words + words

        print('Total words in {} files detected: {}'.format(len(input_files), total_words))
    else:
        print('{} job-type not supported! Doing Nothing'.format(args['job-type']))

    output_file = os.path.join(processed_data_path, 'total_words_'+datetime.now().strftime("%d%m%Y_%H_%M_%S")+'.txt')
    print('Writing output file: {}'.format(output_file))
    f = open(output_file, "a")
    f.write('Total Words: {}'.format(total_words))
    f.close()

    output_files = [file for file in os.listdir(processed_data_path) if file.endswith('.' + 'txt')]
    print('Available output text files: {}'.format(output_files))

    print("Processing Complete")

if __name__ == "__main__":
    main()