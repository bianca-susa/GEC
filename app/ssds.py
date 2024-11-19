import csv
import os


def txt_to_csv_two_columns(txt_filename, csv_filename, directory="data"):
    txt_path = os.path.join(directory, txt_filename)
    csv_path = os.path.join(directory, csv_filename)

    with open(txt_path, 'r', encoding='utf-8') as txt_file, open(csv_path, 'w', encoding='utf-8',
                                                                 newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["input", "output"])

        lines = txt_file.readlines()

        i = 1
        for line in lines:
            # Elimină spațiile inutile și liniile goale
            line = line.strip()
            if not line:
                continue

            if i == 1:
                output = line  # Prima propoziție (output)
                i = 2
            else:
                input = line  # A doua propoziție (input)
                csv_writer.writerow([input, output])  # Scrie în CSV
                i = 1  # Resetează pentru următorul set


# Exemplu de utilizare
txt_to_csv_two_columns("train.txt", "train.csv")
txt_to_csv_two_columns("test.txt", "test.csv")
txt_to_csv_two_columns("dev.txt", "dev.csv")
