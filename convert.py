import csv

def csv_to_txt(csv_file, txt_file):
    with open(csv_file, 'r') as csv_file:
        reader = csv.DictReader(csv_file)

        with open(txt_file, 'w') as txt_file:
            for i, row in enumerate(reader):
                value = row['Low']
                txt_file.write(f"{i+1} {value}\n")

# Replace 'input.csv' and 'output.txt' with your file names
csv_to_txt('apple_stock.csv', 'apple_ds.txt')
