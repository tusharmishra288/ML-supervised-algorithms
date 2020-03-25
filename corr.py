
import pandas as pd
#handling csv files
print("Reading realEstate_trans.csv")
Read_csv_filename = 'realEstate_trans.csv'
Read_csv_data = pd.read_csv(Read_csv_filename)
Read_csv_data = Read_csv_data[['beds','baths','sq__ft','price','zip']]

# calculate the correlations
coefficients_type = ['pearson', 'kendall', 'spearman']

csv_corr = {}


for coefficient in coefficients_type:
    csv_corr[coefficient] = Read_csv_data.corr(method=coefficient)

print(csv_corr)

# output to a file
print("writing to realEstate_trans_correlation.csv")
Write_csv_filename = 'output/realEstate_trans_correlation.csv'
with open(Write_csv_filename,'w+') as write_csv:
   for corr in csv_corr:
       write_csv.write(corr + '\n')
       write_csv.write(csv_corr[corr].to_csv(sep=','))
       write_csv.write('\n')
