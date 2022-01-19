import camelot
import pandas as pd


tables = camelot.read_pdf('keppel-corporation-limited-annual-report-2018.pdf', pages='69', flavor='stream', table_areas=['50,325,555,100'])
print(tables)
y = camelot.plot(tables[0], kind='grid').show()

xx = tables[0].df
#xx.to_excel('Assignment_2.xlsx')
print(tables[0].df)

import xlsxwriter
workbook = xlsxwriter.Workbook('Assignment_2.xlsx')
worksheet = workbook.add_worksheet()
cell_format = workbook.add_format()
cell_format.set_align('center')

worksheet.merge_range('A1:G1', str(xx.iloc[0,0]), cell_format)
worksheet.merge_range('B2:G2', str(xx.iloc[1,4]), cell_format)


for i in range(xx.shape[0]-2): #iterate over rows
    for j in range(xx.shape[1]): #iterate over columns
        try:
            value = xx.iloc[i+2, j] #get cell value
            worksheet.write(i+2, j, value)
        except KeyError:
            continue
workbook.close()


