import pandas as pd
from ml_models.classifier import Classifier

from rdkit import Chem
from rdkit.Contrib.SA_Score import sascorer

df = pd.read_csv('predictions.csv', index_col = 0)
# df.drop_duplicates(subset = 'cofactor_smiles', inplace = True)

df_ish = pd.read_csv('data/database_CCDC.csv')

drug = df['drug_smiles']
coformer = df['cofactor_smiles']
properties = ['unobstructed']

classification = Classifier()

def predict_property(row, prop):
    return classification.predict_properties(row['drug_smiles'], row['cofactor_smiles'], [prop])[prop]

def predict_property_proba(row, prop):
    return classification.predict_properties_proba(row['drug_smiles'], row['cofactor_smiles'], [prop])[prop]

# Применяем функцию к каждому ряду DataFrame и сохраняем результат в новом столбце 'properties'
df['unobstructed'] = df.apply(lambda row: predict_property_proba(row, 'unobstructed'), axis=1)
df['h_bond_bridging'] = df.apply(lambda row: predict_property_proba(row, 'h_bond_bridging'), axis=1)
df['orthogonal_planes'] = df.apply(lambda row: predict_property_proba(row, 'orthogonal_planes'), axis=1)

def median_proba():
    return {'unobstrucred': df['unobstructed'].median(),
            'h_bond_bridging': df['h_bond_bridging'].median(),
            'orthogonal_planes': df['orthogonal_planes'].median()}

print(median_proba())

def validity():
    return 1

def novelty():
    counter = 0
    for i in df['cofactor_smiles']:
        if i not in df_ish.values:
            counter += 1
    return counter / len(df)

def duplicates():
    return df['cofactor_smiles'].duplicated().sum() / len(df)

def make_coformer(row):
    mol = Chem.MolFromSmiles(row['cofactor_smiles'])
    score = sascorer.calculateScore(mol)
    return score

def target_coformers(): 
    df['sa_score'] = df.apply(make_coformer, axis = 1)
    return (df['sa_score'] <= 3).sum() / len(df)

# # Вводим SMILES-строку молекулы
# smiles = 'CCO'  # Пример: этанол
# mol = Chem.MolFromSmiles(smiles)

# # Проверяем, удалось ли создать молекулу
# if mol is not None:
#     # Рассчитываем SA Score
#     sa_score = 
#     print(f"SA Score для молекулы {smiles}: {sa_score}")
# else:
#     print("Ошибка: не удалось создать молекулу из SMILES.")

print(validity())
print(novelty())
print(duplicates())
print(target_coformers())
