from openpyxl import load_workbook


class DataSet:
    @staticmethod
    def get_data(filepath):
        wb = load_workbook(filepath)
        ws = wb.active

        input_data, output_data = [], []

        for row in ws.iter_rows(min_row=2, values_only=True):
            input_data.append(row[:(len(row) - 1)])
            output_data.append(row[(len(row) - 1)])

        return ((input_data[:(int(len(input_data) * 70 / 100))], output_data[:(int(len(output_data) * 70 / 100))]),
                (input_data[-(int(len(input_data) * 30 / 100)):], output_data[-(int(len(output_data) * 30 / 100)):]))



