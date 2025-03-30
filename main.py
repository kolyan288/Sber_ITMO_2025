import random
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.QED import qed

class PairedEvolutionaryGenerator:
    def __init__(self, df_dataset, population_size=100, generations=50, 
                 mutation_rate=0.1, crossover_rate=0.7, elitism=0.1):
        """
        Инициализация генератора пар молекула-кофактор
        """
        # Конвертируем бинарные признаки в float и проверяем SMILES
        self.df_dataset = self._validate_dataset(df_dataset)
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism = elitism
        self.current_population = None
        self.initialize_population()

    def _validate_dataset(self, df):
        """Проверка и очистка входного датасета"""
        # Копируем DataFrame чтобы не изменять оригинал
        df = df.copy()
        
        # Конвертируем бинарные признаки
        for feature in ['unobstructed', 'h_bond_bridging', 'orthogonal_planes']:
            if feature in df.columns:
                df[feature] = pd.to_numeric(df[feature], errors='coerce').fillna(0)
        
        # Проверяем валидность SMILES
        valid_mask = df.apply(lambda row: self._are_valid_smiles(row['drug_smiles'], row['cofactor_smiles']), axis=1)
        
        if not valid_mask.all():
            print(f"Удалено {len(df) - valid_mask.sum()} невалидных пар SMILES из датасета")
            df = df[valid_mask].copy()
        
        if len(df) == 0:
            raise ValueError("После очистки датасет не содержит валидных пар SMILES")
            
        return df

    def _are_valid_smiles(self, drug_smiles, cofactor_smiles):
        """Проверяет валидность пары SMILES"""
        try:
            drug_mol = Chem.MolFromSmiles(str(drug_smiles))
            cofactor_mol = Chem.MolFromSmiles(str(cofactor_smiles))
            return drug_mol is not None and cofactor_mol is not None
        except:
            return False

    def initialize_population(self):
        """Инициализация начальной популяции из DataFrame"""
        if len(self.df_dataset) < self.population_size:
            self.current_population = self.df_dataset.sample(
                n=self.population_size, 
                replace=True,
                random_state=42
            ).copy()
        else:
            self.current_population = self.df_dataset.sample(
                n=self.population_size,
                replace=False,
                random_state=42
            ).copy()

    def calculate_fitness(self, row):
        """Расчет fitness для пары молекула-кофактор"""
        try:
            drug_mol = Chem.MolFromSmiles(str(row['drug_smiles']))
            cofactor_mol = Chem.MolFromSmiles(str(row['cofactor_smiles']))
            
            if drug_mol is None or cofactor_mol is None:
                return -10.0
                
            fitness = 0.0
            # Бинарные признаки
            fitness += float(row.get('unobstructed', 0))
            fitness += float(row.get('h_bond_bridging', 0))
            fitness += float(row.get('orthogonal_planes', 0))
            
            # QED оценки
            fitness += 0.5 * qed(drug_mol)
            fitness += 0.3 * qed(cofactor_mol)
            
            # Штраф за молекулярный вес
            fitness -= 0.01 * (Descriptors.MolWt(drug_mol) + Descriptors.MolWt(cofactor_mol)) / 200
            
            return fitness
        except Exception as e:
            print(f"Error in calculate_fitness: {e}")
            return -10.0

    def select_parents(self, tournament_size=3):
        """Турнирная селекция родителей"""
        parents = []
        population = self.current_population.to_dict('records')
        
        for _ in range(2):
            tournament = random.sample(population, tournament_size)
            winner = max(tournament, key=lambda x: self.calculate_fitness(x))
            parents.append(winner)
            
        return parents[0], parents[1]

    def crossover(self, parent1, parent2):
        """Кроссовер двух родителей с проверкой валидности"""
        max_attempts = 10
        attempts = 0
        
        while attempts < max_attempts:
            child = parent1.copy()
            
            # Выбираем тип кроссовера
            crossover_type = random.choice(['swap_drugs', 'swap_cofactors', 'internal'])
            
            if crossover_type == 'swap_drugs':
                child['drug_smiles'] = parent2['drug_smiles']
            elif crossover_type == 'swap_cofactors':
                child['cofactor_smiles'] = parent2['cofactor_smiles']
            else:
                # Внутренний кроссовер SMILES с проверкой валидности
                child['drug_smiles'] = self._crossover_smiles(
                    parent1['drug_smiles'], parent2['drug_smiles'])
                child['cofactor_smiles'] = self._crossover_smiles(
                    parent1['cofactor_smiles'], parent2['cofactor_smiles'])
            
            # Усреднение признаков
            for feature in ['unobstructed', 'h_bond_bridging', 'orthogonal_planes']:
                try:
                    val1 = float(parent1[feature])
                    val2 = float(parent2[feature])
                    child[feature] = (val1 + val2) / 2
                except (ValueError, TypeError):
                    child[feature] = random.choice([parent1[feature], parent2[feature]])
            
            # Проверяем валидность SMILES
            if self._are_valid_smiles(child['drug_smiles'], child['cofactor_smiles']):
                return child
                
            attempts += 1
        
        # Если не удалось создать валидную пару, возвращаем случайного родителя
        return random.choice([parent1, parent2])

    def _crossover_smiles(self, smiles1, smiles2):
        """Вспомогательный метод для кроссовера SMILES с проверкой валидности"""
        max_attempts = 5
        attempts = 0
        
        while attempts < max_attempts:
            if len(smiles1) < 2 or len(smiles2) < 2:
                return random.choice([smiles1, smiles2])
                
            split1 = random.randint(1, len(smiles1)-1)
            split2 = random.randint(1, len(smiles2)-1)
            new_smiles = smiles1[:split1] + smiles2[split2:]
            
            # Проверяем валидность нового SMILES
            if Chem.MolFromSmiles(new_smiles) is not None:
                return new_smiles
                
            attempts += 1
        
        # Если не удалось создать валидный SMILES, возвращаем случайный из родителей
        return random.choice([smiles1, smiles2])

    def mutate(self, individual):
        """Мутация индивидуума с проверкой валидности"""
        max_attempts = 5
        attempts = 0
        
        while attempts < max_attempts:
            mutated = individual.copy()
            
            # Выбираем цель мутации
            target = random.choice(['drug', 'cofactor', 'both'])
            
            if target in ['drug', 'both']:
                mutated['drug_smiles'] = self._mutate_smiles(individual['drug_smiles'])
            
            if target in ['cofactor', 'both']:
                mutated['cofactor_smiles'] = self._mutate_smiles(individual['cofactor_smiles'])
            
            # Мутация признаков
            for feature in ['unobstructed', 'h_bond_bridging', 'orthogonal_planes']:
                try:
                    current_val = float(individual[feature])
                    mutated_val = current_val + random.uniform(-0.2, 0.2)
                    mutated[feature] = np.clip(mutated_val, 0, 1)
                except (ValueError, TypeError):
                    mutated[feature] = individual[feature]
            
            # Проверяем валидность SMILES
            if self._are_valid_smiles(mutated['drug_smiles'], mutated['cofactor_smiles']):
                return mutated
                
            attempts += 1
        
        # Если не удалось создать валидную мутацию, возвращаем оригинал
        return individual

    def _mutate_smiles(self, smiles):
        """Мутация SMILES строки с проверкой валидности"""
        max_attempts = 5
        attempts = 0
        
        original_smiles = smiles
        
        while attempts < max_attempts:
            if len(smiles) < 1:
                return smiles
                
            mutation_type = random.choice(['add', 'delete', 'replace', 'permute'])
            
            try:
                if mutation_type == 'add' and len(smiles) < 100:
                    atoms = ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', '=', '#']
                    pos = random.randint(0, len(smiles))
                    new_smiles = smiles[:pos] + random.choice(atoms) + smiles[pos:]
                    
                elif mutation_type == 'delete' and len(smiles) > 5:
                    pos = random.randint(0, len(smiles)-1)
                    new_smiles = smiles[:pos] + smiles[pos+1:]
                    
                elif mutation_type == 'replace' and len(smiles) > 1:
                    pos = random.randint(0, len(smiles)-1)
                    atoms = ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br']
                    new_smiles = smiles[:pos] + random.choice(atoms) + smiles[pos+1:]
                    
                elif mutation_type == 'permute' and len(smiles) > 2:
                    i, j = sorted(random.sample(range(len(smiles)), 2))
                    new_smiles = smiles[:i] + smiles[j] + smiles[i+1:j] + smiles[i] + smiles[j+1:]
                else:
                    new_smiles = smiles
                
                # Проверяем валидность
                if Chem.MolFromSmiles(new_smiles) is not None:
                    return new_smiles
                    
            except:
                pass
                
            attempts += 1
        
        return original_smiles

    def generate_new_generation(self):
        """Генерация нового поколения с проверкой валидности"""
        new_population = []
        
        # Элитизм
        elite_size = int(self.elitism * self.population_size)
        elite = self.current_population.copy()
        elite['fitness'] = elite.apply(self.calculate_fitness, axis=1)
        elite = elite.nlargest(elite_size, 'fitness')
        new_population.extend(elite.drop(columns='fitness').to_dict('records'))
        
        # Генерация потомков
        while len(new_population) < self.population_size:
            parent1, parent2 = self.select_parents()
            
            if random.random() < self.crossover_rate:
                child = self.crossover(parent1, parent2)
            else:
                child = random.choice([parent1, parent2])
                
            if random.random() < self.mutation_rate:
                child = self.mutate(child)
                
            # Проверяем валидность перед добавлением
            if self._are_valid_smiles(child['drug_smiles'], child['cofactor_smiles']):
                new_population.append(child)
            else:
                # Если невалидно, добавляем случайного родителя
                new_population.append(random.choice([parent1, parent2]))
            
        self.current_population = pd.DataFrame(new_population)

    def run_evolution(self, target_drug_smiles=None):
        """Запуск эволюционного алгоритма"""
        for generation in range(self.generations):
            self.generate_new_generation()
            
            if target_drug_smiles is not None:
                self.current_population['drug_smiles'] = target_drug_smiles
                
            # Логирование
            fitness = self.current_population.apply(self.calculate_fitness, axis=1)
            print(f"Generation {generation+1}: Avg Fitness = {np.mean(fitness):.2f}, Max Fitness = {np.max(fitness):.2f}")
        
        # Возвращаем лучшие результаты
        result = self.current_population.copy()
        result['fitness'] = result.apply(self.calculate_fitness, axis=1)
        return result.sort_values('fitness', ascending=False)


df = pd.read_csv('dataset2.csv', names = ['drug_smiles', 'cofactor_smiles', 'unobstructed', 'h_bond_bridging', 'orthogonal_planes'])

# Инициализация и запуск
generator = PairedEvolutionaryGenerator(df, population_size=400, generations=40)
best_pairs = generator.run_evolution()


# Для конкретного лекарства
target_drug = 'CN1C2=C(C(=O)N(C1=O)C)NC=N2'
best_cofactors = generator.run_evolution(target_drug_smiles=target_drug)

best_cofactors.to_csv('predictions.csv')

print(best_cofactors)