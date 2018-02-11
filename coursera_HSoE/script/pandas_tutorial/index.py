import pandas
import numpy as np

data = pandas.read_csv('../data/titanic.csv', index_col='PassengerId')

# Какое количество мужчин и женщин ехало на корабле?
sex = data['Sex'].value_counts()
print(sex[0], sex[1])
print()

# Какой части пассажиров удалось выжить?
surv = data['Survived'].value_counts()
surv_part = np.round(surv[1] / (surv[0] + surv[1]) * 100, decimals=2)
print(surv_part)
print()

# Какую долю пассажиры первого класса составляли среди всех пассажиров?
p_class = data['Pclass'].value_counts()
p_class_part = np.round(p_class[1] / (p_class[1] + p_class[2] + p_class[3]) * 100, decimals=2)
print(p_class_part)
print()

# Какого возраста были пассажиры? Посчитайте среднее и медиану возраста пассажиров
age = data['Age']
mean_age = np.round(age.mean(), decimals=2)
median_age = age.median()
print(mean_age, median_age)
print()

# Посчитайте корреляцию Пирсона между признаками SibSp и Parch
sib_sp = data['SibSp']
parch = data['Parch']
print(np.round(sib_sp.corr(parch, method='pearson'), decimals=2))
print()

# Какое самое популярное женское имя на корабле?
name = data['Name']
pattern_list = name.str.extractall('(Mrs.|Miss.)(.*\(?)')[1]
name_parentheses = pattern_list.str.extractall('(\(.*\))')[0].str.extract('(\w+)', expand=True)[0].value_counts()
name_without = pattern_list.str.extractall('(\(.*\))')[0]
print(name_parentheses)