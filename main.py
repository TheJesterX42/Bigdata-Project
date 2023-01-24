from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
import seaborn as sns
import matplotlib.pyplot as plt
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.clustering import KMeans, GaussianMixture, BisectingKMeans
from pyspark.ml.evaluation import ClusteringEvaluator


#Cria uma sessão do Spark
spark = SparkSession.builder.appName("Teste").getOrCreate()

# Carrega um arquivo CSV
df = spark.read.format("csv").option("header", "true").load("master.csv")

###############################Tratamento de dados####################################

# Remover linhas com dados em falta
df = df.dropna()
df.columns

#Mudar as colunas para inteiro
colunas = ['year', 'suicides_no', 'population', 'suicides/100k pop', 'HDI for year', 'gdp_per_capita ($)']
for coluna in colunas:
    df = df.withColumn(coluna, col(coluna).cast(IntegerType()))

#Mudar as colunas para double
colunas1 = [ 'suicides/100k pop', 'HDI for year']
for coluna1 in colunas1:
    df = df.withColumn(coluna1, col(coluna1).cast(DoubleType()))

# seleciona as colunas que serão utilizadas como features
assembler = VectorAssembler(inputCols=['year', 'suicides_no', 'population', 'suicides/100k pop', 'HDI for year', 'gdp_per_capita ($)'], outputCol="features")

# Cria um evaluator para calcular o índice de silhouette
evaluator = ClusteringEvaluator()
# Cria o dataframe com as features
df_al = assembler.transform(df)
# Divide o dataset em treino e teste
train_data, test_data = df_al.randomSplit([0.7, 0.3])

# Converte Dataset para Pandas
df_pd = df_al.toPandas()

##################################Analise de dados##########################################

# Verifica o esquema do conjunto de dados
df.printSchema()
print("s")
# Verifica as estatísticas básicas dos dados
df.describe().show()
# Verifica o número de dados faltantes em cada coluna
df.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in df.columns]).show()

# Verifica o número de linhas e colunas
print("Numero de linhas: ", df.count())
print("Numero de colunas: ", len(df.columns))

#Calcula o coeficiente de correlação de Pearson para as variaveis inteiras e double
def CV():
    # Verifica a correlação entre as variáveis
    int_colunas =[col[0] for col in df.dtypes if col[1] in ('int', 'double')]
    for i in range(len(int_colunas)):
        for j in range(i+1, len(int_colunas)):
            corr = df.select(int_colunas).corr(int_colunas[i], int_colunas[j])
            print(f"Correlação entre {int_colunas[i]} e {int_colunas[j]}: {corr}")

#Constroi um grafico de regressão linear
def RL():
    # divide o dataset em treino e teste
    train_data, test_data = df_al.randomSplit([0.7, 0.3])

    # Regressao lienar cria o modelo
    lr = LinearRegression(featuresCol='features', labelCol='suicides/100k pop')

    # treina o modelo
    lr_model = lr.fit(train_data)

    # faz as previsões no dataset de teste
    predictions = lr_model.transform(test_data)

    # exibe as previsões
    predictions.select("prediction", "suicides/100k pop", "features").show()
    evaluator = RegressionEvaluator(labelCol="suicides/100k pop", predictionCol="prediction", metricName="rmse")
    # Calcula o erro médio quadrático
    rmse = evaluator.evaluate(predictions)
    print("Root Mean Squared Error (RMSE) = %g" % rmse)

    # Visualização da regressão linear
    sns.regplot(x=predictions.select("prediction").toPandas(), y=predictions.select("suicides/100k pop").toPandas())
    plt.xlabel("Previsão")
    plt.ylabel("Taxa de Suicídios por 100 mil de População")
    plt.title("Regressão Linear")
    plt.show()

#Constroi um grafico 3D para uma melhor visualização das variaveis  year, suicides_no, population
def G3D():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = df_pd['year'].values
    y = df_pd['suicides_no'].values
    z = df_pd['population'].values

    ax.scatter(x, y, z)
    ax.set_xlabel('Year')
    ax.set_ylabel('Suicides_no')
    ax.set_zlabel('Population')
    plt.title("Grafico 3D ")
    plt.show()

#Cronstroi um grafico de barras relacionando as variaveis sex e suicides_no
def GSS():
#Criar um grafico de barras
    barr = df.groupBy("sex").agg(avg("suicides_no"))
    # Converter o resultado do agrupamento para um DataFrame Pandas
    barr_pandas = barr.toPandas()
    colors = ['#ff69b4' if x == 'female' else '#0000ff' for x in barr_pandas['sex']]
    # Dar plot para gráfico de barras
    plt.bar(barr_pandas['sex'], barr_pandas['avg(suicides_no)'], color =colors)

    # Adicionar títulos aos eixos
    plt.xlabel("Gênero")
    plt.ylabel("Média dos suicídios")
    plt.title("Relação de suicídios entre gêneros")
    plt.legend() #Adicionar legenda
    # Mostrar o gráfico
    plt.show()

#Constroi um grafico de calor com as variaveis year, suicides/100k pop e HDI for year
def RSI():
    # Analise a relação entre as taxas de suicídio e o IDH
    df_heatmap = df.groupBy("year").agg(avg("suicides_no").alias("num de suicidios"), avg("population").alias("População")).sort("year")
    df_heatmap_pd = df_heatmap.toPandas()
    corr = df_heatmap_pd['num de suicidios'].corr(df_heatmap_pd['População'])
    print(corr)
    plt.title("Gráfico de calor ")
    sns.heatmap(df_heatmap_pd.corr(), annot=True)

    plt.legend()
    plt.show()

#Constroi um de algoritmo K-means com as variaveis year, suicides_no e mostra a sua dispresão
def KM():
    # Selecionar os dados de interesse
    df_km = df[['year', 'suicides_no', 'gdp_per_capita ($)']]
    # Criar a coluna features usando VectorAssembler
    assembler = VectorAssembler(inputCols=['year', 'suicides_no', 'gdp_per_capita ($)'], outputCol="features")
    df_km = assembler.transform(df_km)
    # Cria o modelo
    kmeans = KMeans(k=3, seed=1)
    # Divide o dataset em treino e teste
    train_data, test_data = df_km.randomSplit([0.7, 0.3])
    # Treina o modelo com o seu dataset
    model = kmeans.fit(train_data)
    # Adiciona a coluna "prediction" com a previsão do cluster
    df_kmeans = model.transform(df_km)
    # Calcula o índice de silhouette
    silhouette = evaluator.evaluate(df_kmeans)
    print("K-means Silhouette:" + str(silhouette))
    # Converte o dataframe para um pandas dataframe
    df_kmeans_pd = df_kmeans.toPandas()
    # Cria um gráfico de dispersão com as features "gdp_per_capita" e "suicides_no"
    sns.scatterplot(x='gdp_per_capita ($)', y='suicides_no', hue='prediction', data=df_kmeans_pd)
    plt.title("Gráfico de Dispersão ")
    plt.show()

    # Cria uma lista com as coordenadas x e y das features
    x = df_kmeans.select("year").toPandas()
    y = df_kmeans.select("suicides_no").toPandas()
    # Cria uma lista de cores com base no valor da coluna "prediction"
    colors = ['red' if prediction == 0 else 'green' if prediction == 1 else 'blue' for prediction in
              df_kmeans.select("prediction").toPandas()]
    # Plota o gráfico de dispersão
    plt.scatter(x, y, c=colors)
    # Adiciona títulos e rótulos aos eixos
    plt.title("Gráfico de Dispersão K-Means")
    plt.xlabel("Ano")
    plt.ylabel("Numero de Suicidios")
    # Exibe o gráfico
    plt.show()

#Constroi um de algoritmo BisectingKMeans com as variaveis year, suicides_no e gdp_per_capita ($)
def BKM():
    # Selecionar os dados de interesse
    df_bkm = df[['year', 'suicides_no', 'gdp_per_capita ($)']]
    # Criar a coluna features usando VectorAssembler
    assembler = VectorAssembler(inputCols=['year', 'suicides_no', 'gdp_per_capita ($)'], outputCol="features")
    df_bkm = assembler.transform(df_bkm)
    # Cria o objeto do modelo
    bkm = BisectingKMeans(k=3, seed=1)
    # Divide o dataset em treino e teste
    train_data, test_data = df_bkm.randomSplit([0.7, 0.3])
    # Treina o modelo
    model = bkm.fit(train_data)
    # Faz as previsões
    df_bkm1 = model.transform(train_data)
    # Calcula o índice de silhouette
    silhouette_bkm = evaluator.evaluate(df_bkm1)
    print(" BisectingKMeans silhouette:", silhouette_bkm)

    #Modelo para pandas
    df_bkm1_pd = df_bkm1.toPandas()
    # Plota o gráfico
    plt.scatter(df_bkm1_pd['year'].to_numpy(), df_bkm1_pd['suicides_no'].to_numpy(), c=df_bkm1_pd['prediction'].to_numpy())
    plt.xlabel('Ano')
    plt.ylabel('Numero de suicidios')
    plt.title('Gráfico de Dispersão')
    plt.colorbar(label='Cluster')
    plt.legend
    plt.show()

def main():
    KM()
    BKM()
    print("final")

main()