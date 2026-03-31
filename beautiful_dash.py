import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import ssl

# Настройка страницы
st.set_page_config(
    page_title="Titanic Dashboard",
    page_icon="🚢",
    layout="wide"
)

# Простой CSS
st.markdown("""
<style>
    /* Заголовок */
    .title {
        background: #2c3e50;
        padding: 1.2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }

    .title h1 {
        margin: 0;
        font-size: 1.8rem;
    }

    .title p {
        margin: 0.3rem 0 0 0;
        opacity: 0.8;
    }

    /* Секции */
    .section {
        background: white;
        padding: 1rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        border: 1px solid #e0e0e0;
    }

    .section h3 {
        margin: 0 0 1rem 0;
        color: #2c3e50;
        font-size: 1.3rem;
        border-left: 4px solid #3498db;
        padding-left: 0.8rem;
    }

    /* Карточка статистики */
    .stat {
        background: #f8f9fa;
        padding: 0.8rem;
        border-radius: 8px;
        text-align: center;
        border: 1px solid #e0e0e0;
    }

    .stat-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #2c3e50;
    }

    .stat-label {
        font-size: 0.8rem;
        color: #7f8c8d;
        margin-top: 0.3rem;
    }

    /* Футер */
    .footer {
        background: #2c3e50;
        padding: 1rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-top: 2rem;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)


# Загрузка данных
@st.cache_data
def load_data():
    ssl._create_default_https_context = ssl._create_unverified_context
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    try:
        df = pd.read_csv(url)
        df.columns = df.columns.str.lower()
        return df
    except:
        st.error("Ошибка загрузки данных")
        return None


df = load_data()
if df is None:
    st.stop()

# ==================== ШАПКА ====================
st.markdown("""
<div class="title">
    <h1>Titanic Dashboard</h1>
    <p>Анализ данных пассажиров</p>
</div>
""", unsafe_allow_html=True)

# ==================== КЛЮЧЕВЫЕ МЕТРИКИ ====================
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="stat">
        <div class="stat-value">{len(df)}</div>
        <div class="stat-label">Всего пассажиров</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    survived = df['survived'].sum()
    st.markdown(f"""
    <div class="stat">
        <div class="stat-value" style="color: #27ae60;">{survived}</div>
        <div class="stat-label">Выжило</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    rate = (survived / len(df)) * 100
    st.markdown(f"""
    <div class="stat">
        <div class="stat-value" style="color: #3498db;">{rate:.1f}%</div>
        <div class="stat-label">Выживаемость</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    avg_age = df['age'].mean()
    st.markdown(f"""
    <div class="stat">
        <div class="stat-value">{avg_age:.1f}</div>
        <div class="stat-label">Средний возраст</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ==================== 1. ОПИСАТЕЛЬНАЯ СТАТИСТИКА ====================
st.markdown('<div class="section"><h3>Описательная статистика</h3>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.write("**Размер данных:**")
    st.write(f"- Строк: {df.shape[0]}")
    st.write(f"- Столбцов: {df.shape[1]}")
    st.write("")
    st.write("**Типы данных:**")
    types = pd.DataFrame({
        'Столбец': df.columns,
        'Тип': df.dtypes.values
    })
    st.dataframe(types, hide_index=True, use_container_width=True)

with col2:
    st.write("**Пример данных:**")
    st.dataframe(df.head(), use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

# ==================== 2. ВЫВОД N СТРОК ====================
st.markdown('<div class="section"><h3>🔍 Просмотр данных</h3>', unsafe_allow_html=True)

n = st.slider("Количество строк:", 5, 50, 10)
st.dataframe(df.head(n), use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

# ==================== 3. ГРАФИКИ ====================
st.markdown('<div class="section"><h3>Графики</h3>', unsafe_allow_html=True)

# График 1: Выживаемость
st.subheader("Выживаемость")
fig1, ax1 = plt.subplots(figsize=(8, 4))
counts = df['survived'].value_counts().sort_index()
colors = ['#e74c3c', '#27ae60']
bars = ax1.bar(['Погибли', 'Выжили'], counts.values, color=colors, edgecolor='white')
ax1.set_ylabel('Количество')
for bar, v in zip(bars, counts.values):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5, str(v), ha='center', fontweight='bold')
st.pyplot(fig1)

# График 2: Возраст
st.subheader("Распределение возраста")
fig2, ax2 = plt.subplots(figsize=(10, 4))
age_data = df['age'].dropna()
ax2.hist(age_data, bins=30, color='#3498db', edgecolor='white', alpha=0.7)
ax2.set_xlabel('Возраст')
ax2.set_ylabel('Количество')
ax2.axvline(age_data.mean(), color='red', linestyle='--', label=f'Средний: {age_data.mean():.1f}')
ax2.legend()
st.pyplot(fig2)

# График 3: Интерактивный (по классу/полу)
st.subheader("Выживаемость по категориям")
option = st.radio("", ['По классу', 'По полу'], horizontal=True)

if option == 'По классу':
    data = df.groupby('pclass')['survived'].mean() * 100
    labels = ['1 класс', '2 класс', '3 класс']
    fig3 = px.bar(x=labels, y=data.values, text=data.values.round(1),
                  color=data.values, color_continuous_scale='RdYlGn',
                  labels={'x': 'Класс', 'y': 'Выживаемость (%)'})
    fig3.update_traces(texttemplate='%{text}%', textposition='outside')
    fig3.update_layout(showlegend=False, height=400)
else:
    data = df.groupby('sex')['survived'].mean() * 100
    labels = ['Мужчины', 'Женщины']
    fig3 = px.bar(x=labels, y=data.values, text=data.values.round(1),
                  color=data.values, color_continuous_scale='RdYlGn',
                  labels={'x': 'Пол', 'y': 'Выживаемость (%)'})
    fig3.update_traces(texttemplate='%{text}%', textposition='outside')
    fig3.update_layout(showlegend=False, height=400)

st.plotly_chart(fig3, use_container_width=True)

# График 4: Стоимость билетов
st.subheader("Стоимость билетов по классам")
fig4, ax4 = plt.subplots(figsize=(10, 4))
df.boxplot(column='fare', by='pclass', ax=ax4, patch_artist=True)
for box in ax4.findobj(plt.Rectangle):
    box.set_facecolor('#3498db')
    box.set_alpha(0.7)
ax4.set_title('')
ax4.set_xlabel('Класс')
ax4.set_ylabel('Стоимость ($)')
st.pyplot(fig4)

# График 5: Корреляция
st.subheader("Корреляция признаков")
numeric = ['survived', 'pclass', 'age', 'sibsp', 'parch', 'fare']
corr = df[numeric].dropna().corr()

fig5, ax5 = plt.subplots(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, fmt='.2f', square=True, ax=ax5)
st.pyplot(fig5)

# График 6: Интерактивный scatter
st.subheader("Возраст vs Стоимость билета")
color_option = st.selectbox("Цвет:", ['Выжил', 'Класс', 'Пол'])

color_map = {'Выжил': 'survived', 'Класс': 'pclass', 'Пол': 'sex'}
scatter_data = df.dropna(subset=['age', 'fare'])

fig6 = px.scatter(scatter_data, x='age', y='fare', color=color_map[color_option],
                  labels={'age': 'Возраст', 'fare': 'Стоимость ($)'},
                  opacity=0.6)
fig6.update_layout(height=450)
st.plotly_chart(fig6, use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

# ==================== ФУТЕР ====================
st.markdown(f"""
<div class="footer">
    Titanic Dashboard | Данные: {len(df)} пассажиров | Выжило: {survived} ({rate:.1f}%)
</div>
""", unsafe_allow_html=True)
