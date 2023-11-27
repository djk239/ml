import pandas as pd

# Variables for dataset paths, to be changed accordingly
attributesDataFile = "./List of attributes and their values.xlsx - Sheet.csv"
gramPositiveDataFile = "./g_data.csv"

# Read in data
dataset1 = pd.read_csv(attributesDataFile)
dataset2 = pd.read_csv(gramPositiveDataFile, header=None, names=['Fold','Label','Protien','Sequence'])

# Take column information from datasets needed in the one we are creating
resultDF = dataset2[['Label', 'Protien']]
new_df = pd.DataFrame(columns=dataset1.iloc[:, 1])
resultDF = pd.concat([resultDF, new_df], axis=1)

# Begin to extract information from rows in provided datafiles
# Sequence ex -> MGGYKGIKADGGKVDQAKQLAAKTAKDIEACQKQTQQLAEYIEGSDWEGQFANKVKDVLLIMAKFQEELVQPMADHQKAIDNLSQNLAKYDTLSIKQGLDRVNP
# Attribute ex -> Molecular weight
for k, sequence in dataset2['Sequence'].items():
    for index, row in dataset1.iterrows():
        attribute_name = row['Attributes']
    
        # Extract the value from the corresponding row
        attributeValue = row.iloc[2:].astype(float)
        
        # Create a dictionary mapping letters to their corresponding values
        letter_to_value = dict(zip(attributeValue.index, attributeValue))

        # Calculate the sum of values for each letter in the sequence
        featureSum = sum(letter_to_value.get(letter, 0) for letter in sequence)

        resultDF.at[k,attribute_name] = featureSum

resultDF.to_csv('GramPositive Protien with attibute sums(according to sequence).csv', index=False)