import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import ssl

# Настройка страницы
st.set_page_config(page_title="Titanic Dashboard", layout="wide")

# Заголовок
st.title("Дашборд: Анализ пассажиров Титаника")
st.markdown("---")


# Загрузка данных с URL
@st.cache_data
def load_data():
    # Отключаем проверку SSL (для некоторых сетей)
    ssl._create_default_https_context = ssl._create_unverified_context

    # Рабочие ссылки на датасет Титаника
    urls = [
        "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv",
        "https://raw.githubusercontent.com/agconti/kaggle-titanic/master/data/train.csv"
    ]

    for url in urls:
        try:
            st.info(f"Пробуем загрузить данные из: {url}")
            # Пробуем разные параметры чтения CSV
            df = pd.read_csv(url, encoding='utf-8')

            # Проверяем, что загрузились нужные колонки
            if 'survived' in df.columns:
                st.success("Данные успешно загружены!")
                return df
            elif 'Survived' in df.columns:
                # Приводим имена колонок к нижнему регистру
                df.columns = df.columns.str.lower()
                st.success("Данные успешно загружены!")
                return df
        except Exception as e:
            st.warning(f"Не удалось загрузить из {url}: {str(e)}")
            continue

    # Если ничего не сработало, показываем ошибку
    st.error("Не удалось загрузить данные ни из одного источника")
    return None


df = load_data()

# Проверяем, что данные загружены
if df is None:
    st.stop()

# ======================== 1. Описательная статистика ========================
st.header("Описательная статистика")
st.markdown("**Информация о датасете:**")

col1, col2 = st.columns(2)

with col1:
    st.write("**Форма таблицы:**")
    st.write(f"- Строк: {df.shape[0]}")
    st.write(f"- Столбцов: {df.shape[1]}")

with col2:
    st.write("**Столбцы и типы данных:**")
    dtype_df = pd.DataFrame(df.dtypes.reset_index())
    dtype_df.columns = ["Столбец", "Тип данных"]
    st.dataframe(dtype_df, use_container_width=True)

st.markdown("**Первые 5 строк данных:**")
st.dataframe(df.head(), use_container_width=True)

st.markdown("---")

# ======================== 2. Вывод n строк ========================
st.header("Просмотр данных")
n_rows = st.slider("Выберите количество строк для отображения:",
                   min_value=5, max_value=len(df), value=10, step=5)
st.dataframe(df.head(n_rows), use_container_width=True)

st.markdown("---")

# ======================== 3. Графики ========================
st.header("Визуализация данных")

# График 1: Распределение выживших
st.subheader("1. Выживаемость пассажиров")
fig1, ax1 = plt.subplots(figsize=(8, 5))
survived_counts = df['survived'].value_counts().sort_index()
ax1.bar(['Не выжил (0)', 'Выжил (1)'], survived_counts.values, color=['#FF6B6B', '#4ECDC4'])
ax1.set_ylabel('Количество пассажиров')
ax1.set_title('Распределение выживших и погибших')
for i, v in enumerate(survived_counts.values):
    ax1.text(i, v + 10, str(v), ha='center', fontweight='bold')
st.pyplot(fig1)

# График 2: Распределение возраста
st.subheader("2. Распределение возраста пассажиров")
fig2, ax2 = plt.subplots(figsize=(10, 5))
age_data = df['age'].dropna()
ax2.hist(age_data, bins=30, edgecolor='black', color='lightblue', alpha=0.7)
ax2.set_xlabel('Возраст')
ax2.set_ylabel('Количество пассажиров')
ax2.set_title('Гистограмма распределения возраста')
ax2.axvline(age_data.mean(), color='red', linestyle='--', label=f'Средний возраст: {age_data.mean():.1f}')
ax2.legend()
st.pyplot(fig2)

# График 3: Выживаемость по классу (интерактивный)
st.subheader("3. Выживаемость по классу обслуживания (интерактивный)")
class_sex = st.radio("Выберите группировку:",
                     options=['По полу', 'Без группировки'],
                     horizontal=True)

if class_sex == 'По полу':
    survival_by_class_sex = pd.crosstab([df['pclass'], df['sex']], df['survived'])
    survival_by_class_sex = survival_by_class_sex.stack().reset_index()
    survival_by_class_sex.columns = ['Класс', 'Пол', 'Выжил', 'Количество']
    survival_by_class_sex['Статус'] = survival_by_class_sex['Выжил'].map({0: 'Погиб', 1: 'Выжил'})

    fig3 = px.bar(survival_by_class_sex,
                  x='Класс',
                  y='Количество',
                  color='Статус',
                  barmode='group',
                  facet_col='Пол',
                  title='Выживаемость по классу и полу',
                  color_discrete_map={'Погиб': '#FF6B6B', 'Выжил': '#4ECDC4'})
    fig3.update_layout(height=500)
else:
    survival_by_class = df.groupby(['pclass', 'survived']).size().reset_index(name='count')
    survival_by_class['survived'] = survival_by_class['survived'].map({0: 'Погиб', 1: 'Выжил'})
    fig3 = px.bar(survival_by_class,
                  x='pclass',
                  y='count',
                  color='survived',
                  title='Выживаемость по классу обслуживания',
                  labels={'pclass': 'Класс', 'count': 'Количество', 'survived': 'Статус'},
                  color_discrete_map={'Погиб': '#FF6B6B', 'Выжил': '#4ECDC4'})
    fig3.update_layout(height=500)

st.plotly_chart(fig3, use_container_width=True)

# График 4: Стоимость билетов по классам
st.subheader("4. Распределение стоимости билетов по классам")
fig4, ax4 = plt.subplots(figsize=(10, 6))
df.boxplot(column='fare', by='pclass', ax=ax4)
ax4.set_title('Распределение стоимости билетов по классам')
ax4.set_xlabel('Класс')
ax4.set_ylabel('Стоимость билета ($)')
fig4.suptitle('')
st.pyplot(fig4)

# График 5: Тепловая карта корреляций
st.subheader("5. Корреляция числовых признаков")
numeric_cols = ['survived', 'pclass', 'age', 'sibsp', 'parch', 'fare']
# Проверяем наличие всех колонок
available_numeric = [col for col in numeric_cols if col in df.columns and df[col].dtype in ['int64', 'float64']]
corr_matrix = df[available_numeric].corr()

fig5, ax5 = plt.subplots(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax5,
            fmt='.2f', linewidths=0.5, square=True)
ax5.set_title('Тепловая карта корреляций')
plt.tight_layout()
st.pyplot(fig5)

# График 6: Интерактивный scatter plot
st.subheader("6. Интерактивный график: Возраст vs Стоимость билета")
scatter_filter = st.selectbox("Выберите цветовую группировку:",
                              options=['survived', 'pclass', 'sex'],
                              format_func=lambda x: {'survived': 'Выжил (1) / Погиб (0)',
                                                     'pclass': 'Класс (1, 2, 3)',
                                                     'sex': 'Пол'}[x])

# Удаляем строки с пропущенными значениями
scatter_data = df.dropna(subset=['age', 'fare']).copy()

fig6 = px.scatter(scatter_data,
                  x='age',
                  y='fare',
                  color=scatter_filter,
                  hover_data=['sex', 'pclass'] if 'sex' in scatter_data.columns else None,
                  title=f'Возраст vs Стоимость билета (цвет: {scatter_filter})',
                  labels={'age': 'Возраст', 'fare': 'Стоимость билета ($)',
                          'survived': 'Выжил', 'pclass': 'Класс', 'sex': 'Пол'},
                  opacity=0.6)
fig6.update_layout(height=500)
fig6.update_traces(marker=dict(size=8))
st.plotly_chart(fig6, use_container_width=True)
