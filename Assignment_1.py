import fitz
import xlsxwriter

file = fitz.open('keppel-corporation-limited-annual-report-2018.pdf')
text = file[11].get_text('blocks')
workbook = xlsxwriter.Workbook('Assignment_1.xlsx')
worksheet = workbook.add_worksheet()
row = 0
col = 0
for each in text:
    sen = each[4].split()
    for word in sen:
        worksheet.write(row, col, word)
        row += 1
    col += 1
    row = 0
workbook.close()

with open('Draft.txt', 'w') as f:
    f.write(str(text))
print(text)
