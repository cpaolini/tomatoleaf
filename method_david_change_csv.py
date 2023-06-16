import argparse
import csv
import os

# create argument parser
parser = argparse.ArgumentParser(description='Process CSV file and filter data by folder location')
parser.add_argument('input_file', type=str, help='path and name of input CSV file')
parser.add_argument('output_file', type=str, help='path and name of output CSV file')
parser.add_argument('string_to_replace', type=str, help='string to be replaced in the filenames')
parser.add_argument('folder_location', type=str, help='folder location to filter by')
parser.add_argument('file_extension', type=str, help='file extension of files to filter by')

# parse command-line arguments
args = parser.parse_args()

# list to store processed data
processed_data = []
# iterate through files in the specified folder
for filename in os.listdir(args.folder_location):
    if filename.endswith(args.file_extension):
        filepath = os.path.join(args.folder_location, filename)
        # read CSV data corresponding to the current file
        with open(args.input_file, 'r') as file:
            reader = csv.reader(file, delimiter=';')
            for row in reader:
                if row[0].startswith(filepath.replace(args.folder_location, args.string_to_replace)):
                    # replace folder location
                    filename = row[0].replace(args.string_to_replace, args.folder_location)
                    # create new row with required headers
                    new_row = [filename, row[1], row[2], row[3], row[4], row[5]]
                    # add new row to processed data list
                    processed_data.append(new_row)

# write processed data to output CSV file
with open(args.output_file, 'w', newline='') as file:
    writer = csv.writer(file)
    # write headers
    writer.writerow(['filename', 'class', 'xmin', 'ymin', 'xmax', 'ymax'])
    # write processed data
    writer.writerows(processed_data)
