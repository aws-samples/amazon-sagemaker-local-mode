import os
import sys
from datetime import datetime
import subprocess
subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "/opt/ml/processing/dependencies/requirements.txt"])

import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize

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

    if args['job-type'] == 'word-tokenize':
        print('Word Tokenize Job Type Started')
        all_tokenized_words = []
        for input_file in input_files:
            file = open(os.path.join(input_data_path, input_file), 'r')
            data = file.read()
            tokenized_words = word_tokenize(data)
            print('Detected {} words in {} file'.format(tokenized_words, input_file))
            all_tokenized_words.append(tokenized_words)
    else:
        print('{} job-type not supported! Doing Nothing'.format(args['job-type']))

    output_file = os.path.join(processed_data_path, 'all_tokenized_words_'+datetime.now().strftime("%d%m%Y_%H_%M_%S")+'.txt')
    print('Writing output file: {}'.format(output_file))
    f = open(output_file, "a")
    f.write('Tokenized Words: {}'.format(all_tokenized_words))
    f.close()

    output_files = [file for file in os.listdir(processed_data_path) if file.endswith('.' + 'txt')]
    print('Available output text files: {}'.format(output_files))

    print("Processing Complete")

if __name__ == "__main__":
    main()