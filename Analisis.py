import pandas as pd
import numpy as np
import os
from pathlib import Path
import pyreadstat  # .SAV
import warnings
warnings.filterwarnings('ignore')

# Librerías para visualización
import matplotlib.pyplot as plt
import seaborn as sns

# Para clustering y análisis estadístico
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy import stats

class AnalizadorDatosINE:
    """
    Clase para analizar datos del INE de Guatemala
    """
    
    def __init__(self, ruta_datos='/datos_ine'):
        """
        Inicializa el analizador con la ruta de los datos
        
        Args:
            ruta_datos (str): Ruta donde están los archivos .sav
        """
        self.ruta_datos = Path(ruta_datos)
        self.datos = {}
        self.datos_combinados = None
        
    def cargar_datos_sav(self):
        """
        Carga todos los archivos .sav de la carpeta especificada
        """
        print("=" * 60)
        print("CARGANDO DATOS DEL INE DE GUATEMALA")
        print("=" * 60)
        
        # Diccionario para mapear nombres de archivos
        categorias = {
            'nacimientos': 'nacimientos',
            'matrimonios': 'matrimonios',
            'divorcios': 'divorcios',
            'defunciones': 'defunciones',
            'defunciones_fetales': 'defunciones_fetales',
            'hechos_delictivos': 'hechos_delictivos',
            'violencia_intrafamiliar': 'violencia_intrafamiliar',
            'violencia_mujer': 'violencia_mujer_delitos_sexuales'
        }
        
        # Buscar archivos .sav
        archivos_sav = list(self.ruta_datos.glob('*.sav'))
        
        if not archivos_sav:
            print(f"No se encontraron archivos .sav en {self.ruta_datos}")
            return False
        
        for archivo in archivos_sav:
            nombre_archivo = archivo.stem.lower()
            
            # Identificar categoría
            categoria = None
            for key, value in categorias.items():
                if value in nombre_archivo:
                    categoria = key
                    break
            
            if categoria:
                try:
                    # Leer archivo .sav
                    df, meta = pyreadstat.read_sav(str(archivo))
                    
                    # Convertir a DataFrame de pandas
                    df = pd.DataFrame(df)
                    
                    # Agregar información de año si está disponible
                    if 'year' not in df.columns and 'ano' not in df.columns:
                        # Intentar extraer año del nombre del archivo
                        import re
                        match = re.search(r'\d{4}', nombre_archivo)
                        if match:
                            df['year'] = int(match.group())
                    
                    self.datos[categoria] = df
                    print(f"✓ Cargado: {categoria} - {archivo.name}")
                    print(f"  Forma: {df.shape}, Columnas: {list(df.columns[:5])}...")
                    
                except Exception as e:
                    print(f"✗ Error cargando {archivo.name}: {e}")
        
        print(f"\nTotal de categorías cargadas: {len(self.datos)}")
        return True
    
    def preprocesar_datos(self):
        """
        Preprocesa y combina los datos de diferentes categorías
        """
        print("\n" + "=" * 60)
        print("PREPROCESANDO DATOS")
        print("=" * 60)
        
        datos_anuales = []
        
        for categoria, df in self.datos.items():
            print(f"\nProcesando: {categoria}")
            
            # Encontrar columna de año
            columnas_year = [col for col in df.columns if 'year' in col.lower() or 'ano' in col.lower()]
            if columnas_year:
                year_col = columnas_year[0]
            else:
                print(f"  No se encontró columna de año, usando índice")
                continue
            
            # Agrupar por año y calcular totales
            try:
                # Intentar diferentes estrategias para obtener totales
                if 'total' in df.columns.str.lower():
                    total_col = [col for col in df.columns if 'total' in col.lower()][0]
                    datos_agrupados = df.groupby(year_col)[total_col].sum().reset_index()
                else:
                    # Sumar todas las columnas numéricas
                    columnas_numericas = df.select_dtypes(include=[np.number]).columns
                    if len(columnas_numericas) > 0:
                        df['total'] = df[columnas_numericas].sum(axis=1)
                        datos_agrupados = df.groupby(year_col)['total'].sum().reset_index()
                    else:
                        # Contar registros
                        datos_agrupados = df.groupby(year_col).size().reset_index(name='total')
                
                # Renombrar columnas
                datos_agrupados = datos_agrupados.rename(columns={
                    year_col: 'year',
                    'total': f'total_{categoria}'
                })
                
                datos_anuales.append(datos_agrupados)
                print(f"  ✓ Datos procesados para {len(datos_agrupados)} años")
                
            except Exception as e:
                print(f"  ✗ Error procesando {categoria}: {e}")
        
        # Combinar todos los datos
        if datos_anuales:
            self.datos_combinados = datos_anuales[0]
            for df in datos_anuales[1:]:
                self.datos_combinados = pd.merge(self.datos_combinados, df, on='year', how='outer')
            
            # Ordenar por año
            self.datos_combinados = self.datos_combinados.sort_values('year').reset_index(drop=True)
            
            print(f"\n✓ Datos combinados creados")
            print(f"  Años cubiertos: {self.datos_combinados['year'].min()} - {self.datos_combinados['year'].max()}")
            print(f"  Variables: {list(self.datos_combinados.columns)}")
        
        return self.datos_combinados
    
    def calcular_estadisticas_descriptivas(self):
        """
        Calcula estadísticas descriptivas para cada variable
        """
        if self.datos_combinados is None:
            print("No hay datos combinados para analizar")
            return
        
        print("\n" + "=" * 60)
        print("ESTADÍSTICAS DESCRIPTIVAS")
        print("=" * 60)
        
        # Excluir columna de año
        columnas_numericas = self.datos_combinados.select_dtypes(include=[np.number]).columns
        columnas_numericas = [col for col in columnas_numericas if col != 'year']
        
        estadisticas = pd.DataFrame()
        
        for col in columnas_numericas:
            stats_dict = {
                'Variable': col.replace('total_', '').title(),
                'Media': self.datos_combinados[col].mean(),
                'Mediana': self.datos_combinados[col].median(),
                'Desviación Estándar': self.datos_combinados[col].std(),
                'Mínimo': self.datos_combinados[col].min(),
                'Máximo': self.datos_combinados[col].max(),
                'Tasa Crecimiento (%)': ((self.datos_combinados[col].iloc[-1] / self.datos_combinados[col].iloc[0]) - 1) * 100 
                if len(self.datos_combinados) > 1 else 0
            }
            estadisticas = pd.concat([estadisticas, pd.DataFrame([stats_dict])], ignore_index=True)
        
        print(estadisticas.to_string(index=False))
        
        return estadisticas
    
    def calcular_ratios_tasas(self):
        """
        Calcula ratios y tasas importantes para el análisis social
        """
        if self.datos_combinados is None:
            print("No hay datos combinados para analizar")
            return
        
        print("\n" + "=" * 60)
        print("RATIOS Y TASAS SOCIALES")
        print("=" * 60)
        
        ratios = pd.DataFrame()
        
        # Obtener nombres de columnas disponibles
        columnas = self.datos_combinados.columns
        
        # 1. Tasa de divorcios por matrimonio
        if 'total_matrimonios' in columnas and 'total_divorcios' in columnas:
            self.datos_combinados['tasa_divorcio'] = (
                self.datos_combinados['total_divorcios'] / 
                self.datos_combinados['total_matrimonios']
            ) * 100
        
        # 2. Tasa de mortalidad infantil (aproximada)
        if 'total_defunciones_fetales' in columnas and 'total_nacimientos' in columnas:
            self.datos_combinados['tasa_mortalidad_infantil'] = (
                self.datos_combinados['total_defunciones_fetales'] / 
                self.datos_combinados['total_nacimientos']
            ) * 1000  # Por cada 1000 nacimientos
        
        # 3. Ratio de violencia por población (usando nacimientos como proxy)
        if 'total_hechos_delictivos' in columnas and 'total_nacimientos' in columnas:
            self.datos_combinados['delitos_por_1000'] = (
                self.datos_combinados['total_hechos_delictivos'] / 
                self.datos_combinados['total_nacimientos']
            ) * 1000
        
        # 4. Tasa de violencia intrafamiliar
        if 'total_violencia_intrafamiliar' in columnas and 'total_hechos_delictivos' in columnas:
            self.datos_combinados['proporcion_violencia_intrafamiliar'] = (
                self.datos_combinados['total_violencia_intrafamiliar'] / 
                self.datos_combinados['total_hechos_delictivos']
            ) * 100
        
        print("Ratios calculados:")
        nuevas_columnas = [col for col in self.datos_combinados.columns if col.startswith('tasa_') or col.startswith('ratio_') or 'proporcion' in col]
        for col in nuevas_columnas:
            print(f"\n{col.replace('_', ' ').title()}:")
            print(f"  Media: {self.datos_combinados[col].mean():.2f}")
            print(f"  Último año: {self.datos_combinados[col].iloc[-1]:.2f}")
        
        return self.datos_combinados
    
    def realizar_clustering_anual(self):
        """
        Realiza clustering de años según patrones de comportamiento
        """
        if self.datos_combinados is None:
            print("No hay datos combinados para analizar")
            return
        
        print("\n" + "=" * 60)
        print("CLUSTERING DE AÑOS POR PATRONES DE COMPORTAMIENTO")
        print("=" * 60)
        
        # Preparar datos para clustering
        columnas_clustering = [col for col in self.datos_combinados.columns 
                              if col != 'year' and not col.startswith('tasa_') 
                              and not col.startswith('ratio_')]
        
        X = self.datos_combinados[columnas_clustering].copy()
        
        # Normalizar datos
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Determinar número óptimo de clusters usando el método del codo
        inertias = []
        K_range = range(2, min(6, len(self.datos_combinados)))
        
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X_scaled)
            inertias.append(kmeans.inertia_)
        
        # Aplicar K-Means con 3 clusters (puedes ajustar)
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)
        
        self.datos_combinados['cluster'] = clusters
        
        # Análisis de clusters
        print("\nRESULTADOS DE CLUSTERING:")
        print("-" * 40)
        
        for cluster_id in range(3):
            años_cluster = self.datos_combinados[self.datos_combinados['cluster'] == cluster_id]['year'].tolist()
            datos_cluster = self.datos_combinados[self.datos_combinados['cluster'] == cluster_id][columnas_clustering].mean()
            
            print(f"\nCluster {cluster_id + 1}:")
            print(f"Años: {años_cluster}")
            print("Características promedio:")
            for var in columnas_clustering:
                valor = datos_cluster[var]
                print(f"  {var.replace('total_', '').title()}: {valor:,.0f}")
        
        return self.datos_combinados
    
    def analizar_correlaciones(self):
        """
        Analiza correlaciones entre variables
        """
        if self.datos_combinados is None:
            print("No hay datos combinados para analizar")
            return
        
        print("\n" + "=" * 60)
        print("ANÁLISIS DE CORRELACIONES")
        print("=" * 60)
        
        # Matriz de correlación
        columnas_corr = [col for col in self.datos_combinados.columns 
                        if col != 'year' and col != 'cluster']
        
        corr_matrix = self.datos_combinados[columnas_corr].corr()
        
        print("\nCorrelaciones más fuertes (abs > 0.7):")
        print("-" * 40)
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) > 0.7:
                    var1 = corr_matrix.columns[i].replace('total_', '').title()
                    var2 = corr_matrix.columns[j].replace('total_', '').title()
                    print(f"{var1} - {var2}: {corr_value:.3f}")
        
        return corr_matrix
    
    def visualizar_datos(self):
        """
        Crea visualizaciones de los datos
        """
        if self.datos_combinados is None:
            print("No hay datos combinados para visualizar")
            return
        
        print("\n" + "=" * 60)
        print("CREANDO VISUALIZACIONES")
        print("=" * 60)
        
        # Configurar estilo
        plt.style.use('seaborn-v0_8-darkgrid')
        fig = plt.figure(figsize=(15, 10))
        
        # 1. Evolución temporal
        ax1 = plt.subplot(2, 2, 1)
        columnas_plot = [col for col in self.datos_combinados.columns 
                        if col.startswith('total_') and col != 'year']
        
        for col in columnas_plot[:5]:  # Limitar a 5 variables para claridad
            ax1.plot(self.datos_combinados['year'], 
                    self.datos_combinados[col], 
                    marker='o', 
                    label=col.replace('total_', '').title())
        
        ax1.set_xlabel('Año')
        ax1.set_ylabel('Total')
        ax1.set_title('Evolución Temporal de Indicadores')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # 2. Gráfico de barras para el último año disponible
        ax2 = plt.subplot(2, 2, 2)
        ultimo_año = self.datos_combinados.iloc[-1]
        datos_ultimo_año = {col.replace('total_', '').title(): ultimo_año[col] 
                          for col in columnas_plot if col in ultimo_año}
        
        ax2.bar(datos_ultimo_año.keys(), datos_ultimo_año.values())
        ax2.set_xlabel('Indicador')
        ax2.set_ylabel('Total')
        ax2.set_title(f'Valores para el año {int(ultimo_año["year"])}')
        plt.xticks(rotation=45, ha='right')
        
        # 3. Matriz de correlación (heatmap)
        ax3 = plt.subplot(2, 2, 3)
        columnas_corr = [col for col in columnas_plot if col in self.datos_combinados.columns]
        corr_matrix = self.datos_combinados[columnas_corr].corr()
        
        im = ax3.imshow(corr_matrix, cmap='coolwarm', aspect='auto')
        ax3.set_xticks(range(len(columnas_corr)))
        ax3.set_yticks(range(len(columnas_corr)))
        ax3.set_xticklabels([col.replace('total_', '')[:10] for col in columnas_corr], rotation=45, ha='right')
        ax3.set_yticklabels([col.replace('total_', '')[:10] for col in columnas_corr])
        ax3.set_title('Matriz de Correlación')
        plt.colorbar(im, ax=ax3)
        
        # 4. Scatter plot de relaciones clave
        ax4 = plt.subplot(2, 2, 4)
        if 'total_nacimientos' in self.datos_combinados.columns and 'total_defunciones' in self.datos_combinados.columns:
            ax4.scatter(self.datos_combinados['total_nacimientos'], 
                       self.datos_combinados['total_defunciones'],
                       c=self.datos_combinados['year'],
                       cmap='viridis',
                       s=100)
            ax4.set_xlabel('Nacimientos')
            ax4.set_ylabel('Defunciones')
            ax4.set_title('Relación Nacimientos vs Defunciones')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('analisis_ine_guatemala.png', dpi=300, bbox_inches='tight')
        print("\n✓ Visualizaciones guardadas en 'analisis_ine_guatemala.png'")
        plt.show()
    

    
    def ejecutar_analisis_completo(self):
        """
        Ejecuta todo el pipeline de análisis
        """
        print("INICIANDO ANÁLISIS COMPLETO DE DATOS DEL INE")
        print("=" * 60)
        
        # Paso 1: Cargar datos
        if not self.cargar_datos_sav():
            print("No se pudieron cargar los datos")
            return
        
        # Paso 2: Preprocesar
        self.preprocesar_datos()
        
        if self.datos_combinados is not None:
            # Paso 3: Estadísticas descriptivas
            self.calcular_estadisticas_descriptivas()
            
            # Paso 4: Cálculo de ratios
            self.calcular_ratios_tasas()
            
            # Paso 5: Clustering
            if len(self.datos_combinados) >= 3:  # Necesario para clustering
                self.realizar_clustering_anual()
            
            # Paso 6: Correlaciones
            self.analizar_correlaciones()
            
            # Paso 7: Visualizaciones
            self.visualizar_datos()
            

            # Guardar datos combinados
            self.datos_combinados.to_csv('datos_ine_combinados.csv', index=False, encoding='utf-8')
            print(f"\n✓ Datos combinados guardados en 'datos_ine_combinados.csv'")
            
            print("\n" + "=" * 60)
            print("ANÁLISIS COMPLETADO EXITOSAMENTE")
            print("=" * 60)
            print("\nArchivos generados:")
            print("1. datos_ine_combinados.csv - Datos consolidados")
            print("2. analisis_ine_guatemala.png - Visualizaciones")
            print("3. reporte_analisis.txt - Reporte completo")



# Configurar directorio de datos (ajusta esta ruta)
ruta_datos = "./datos_ine"  # Cambia esta ruta según tu estructura

# Crear analizador
analizador = AnalizadorDatosINE(ruta_datos)

# Ejecutar análisis completo
analizador.ejecutar_analisis_completo()