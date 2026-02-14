import pandas as pd
import numpy as np
from pathlib import Path
import pyreadstat
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.stats import zscore
from sklearn.impute import SimpleImputer
from datetime import datetime
import os

class AnalizadorINERobusto:

    def __init__(self, ruta_datos):
        self.ruta_datos = Path(ruta_datos)
        self.datos = {}
        self.datos_combinados = None
        self.datos_normalizados = None
        # Crear carpeta para resultados si no existe
        self.carpeta_resultados = Path("./resultados_ine")
        self.carpeta_resultados.mkdir(exist_ok=True)
        self.carpeta_graficas = self.carpeta_resultados / "graficas"
        self.carpeta_graficas.mkdir(exist_ok=True)

    # =========================
    # CARGA DE DATOS
    # =========================

    def cargar_datos(self):
        archivos = list(self.ruta_datos.glob("*.sav"))

        for archivo in archivos:
            df, meta = pyreadstat.read_sav(str(archivo))
            df = pd.DataFrame(df)

            # Extraer año del nombre
            import re
            match = re.search(r'(\d{4})', archivo.stem)
            if match:
                df["year"] = int(match.group(1))

            nombre = archivo.stem.lower()
            self.datos[nombre] = df

        print(f"Se cargaron {len(self.datos)} datasets")

    # =========================
    # PREPROCESAMIENTO
    # =========================

    def combinar_datos(self):
        categorias_detectadas = {
            "nacimientos": [],
            "defunciones": [],
            "defunciones_fetales": [],
            "matrimonios": [],
            "divorcios": [],
            "violencia_intrafamiliar": []
        }

        for nombre, df in self.datos.items():
            nombre_lower = nombre.lower()
            
            # Agregar columna year si no existe
            if "year" not in df.columns:
                import re
                match = re.search(r'(\d{4})', nombre)
                if match:
                    df["year"] = int(match.group(1))

            for categoria in categorias_detectadas.keys():
                if categoria in nombre_lower:
                    categorias_detectadas[categoria].append(df)
                    print(f"  - {nombre}: asignado a {categoria}")

        base_final = None

        for categoria, lista_dfs in categorias_detectadas.items():
            if not lista_dfs:
                print(f"  No hay datos para {categoria}")
                continue

            df_categoria = pd.concat(lista_dfs, ignore_index=True)

            if "year" not in df_categoria.columns:
                print(f"  {categoria}: no tiene columna year")
                continue

            # Seleccionar columnas numéricas (excluyendo year)
            columnas_numericas = df_categoria.select_dtypes(include=np.number).columns.tolist()
            columnas_numericas = [c for c in columnas_numericas if c != "year"]
            
            print(f"  {categoria}: {len(columnas_numericas)} columnas numéricas encontradas")
            
            if not columnas_numericas:
                print(f"  {categoria}: no tiene columnas numéricas, usando conteo de filas")
                # Si no hay columnas numéricas, contar filas por año
                agrupado = df_categoria.groupby("year").size().reset_index(name=f"total_{categoria}")
            else:
                # Calcular total sumando todas las columnas numéricas
                df_categoria["total"] = df_categoria[columnas_numericas].sum(axis=1, min_count=1)
                agrupado = df_categoria.groupby("year")["total"].sum().reset_index()
                agrupado.rename(columns={"total": f"total_{categoria}"}, inplace=True)

            if base_final is None:
                base_final = agrupado
            else:
                base_final = base_final.merge(agrupado, on="year", how="outer")

        if base_final is not None:
            base_final = base_final.sort_values("year").reset_index(drop=True)
        else:
            # Crear DataFrame vacío con estructura mínima
            base_final = pd.DataFrame({"year": []})

        self.datos_combinados = base_final
        print("\nDatos combinados - columnas resultantes:")
        print(base_final.columns.tolist())
        print("\nPrimeras filas:")
        print(base_final.head())
        
        return base_final


    # =========================
    # NORMALIZACIÓN Y TASAS
    # =========================

    def calcular_metricas(self):

        df = self.datos_combinados

        # Crecimiento porcentual
        for col in df.columns:
            if col.startswith("total_"):
                df[f"crecimiento_{col}"] = df[col].pct_change() * 100

        # Ratio divorcios/matrimonios
        if "total_matrimonios" in df.columns and "total_divorcios" in df.columns:
            df["ratio_divorcios_matrimonios"] = (
                df["total_divorcios"] / df["total_matrimonios"] * 100
            )

        # Crecimiento vegetativo
        if "total_nacimientos" in df.columns and "total_defunciones" in df.columns:
            df["crecimiento_vegetativo"] = (
                df["total_nacimientos"] - df["total_defunciones"]
            )

        self.datos_combinados = df

    # =========================
    # NORMALIZACIÓN Z-SCORE
    # =========================

    def normalizar_variables(self):

        df = self.datos_combinados.copy()

        columnas = [c for c in df.columns if c.startswith("total_")]

        df_normalizado = df[["year"]].copy()
        df_normalizado[columnas] = df[columnas].apply(zscore)

        self.datos_normalizados = df_normalizado
        print("Variables normalizadas con Z-score")

        return df_normalizado

    # =========================
    # MÉTODO DEL CODO
    # =========================

    def metodo_codo(self, guardar=True):

        columnas = [c for c in self.datos_normalizados.columns if c != "year"]
        imputer = SimpleImputer(strategy="mean")
        X = self.datos_normalizados[columnas]
        X = imputer.fit_transform(X)
        inercias = []
        k_range = range(1, min(10, len(X)))

        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit_transform(X)
            inercias.append(kmeans.inertia_)

        plt.figure(figsize=(10, 6))
        plt.plot(k_range, inercias, 'bo-', linewidth=2, markersize=8)
        plt.xlabel("Numero de clusters (K)", fontsize=12)
        plt.ylabel("Inercia", fontsize=12)
        plt.title("Metodo del Codo para determinar K optimo", fontsize=14)
        plt.grid(True, alpha=0.3)
        
        if guardar:
            ruta_guardado = self.carpeta_graficas / "metodo_codo.png"
            plt.savefig(ruta_guardado, dpi=300, bbox_inches='tight')
            print(f"Grafica guardada en: {ruta_guardado}")
        
        plt.show()

    # =========================
    # CLUSTERING
    # =========================
    def clustering_kmeans(self, k):
        columnas = [c for c in self.datos_normalizados.columns if c != "year"]
        X = self.datos_normalizados[columnas]
        
        # Verificar cuantas columnas tienen al menos un valor no nulo
        columnas_validas = X.columns[~X.isna().all()].tolist()
        print(f"Columnas con al menos un valor: {columnas_validas}")
        
        if len(columnas_validas) == 0:
            print("ERROR: No hay columnas con datos validos para clustering")
            return None
        
        # Usar solo columnas con datos
        X = X[columnas_validas]
        
        # Imputar valores faltantes
        imputer = SimpleImputer(strategy='mean')
        X_imputed = imputer.fit_transform(X)
        
        if X_imputed.shape[0] < k:
            print(f"ADVERTENCIA: Solo hay {X_imputed.shape[0]} muestras para {k} clusters")
            k = min(k, X_imputed.shape[0])
        
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_imputed)
        
        # Guardar los clusters
        self.datos_normalizados['cluster'] = labels
        self.datos_combinados['cluster'] = labels
        
        if len(set(labels)) > 1:
            score = silhouette_score(X_imputed, labels)
            print(f"Silhouette Score: {score:.3f}")
            self.silhouette_score = score
        else:
            print("Solo se formo 1 cluster, no se puede calcular silhouette score")
            self.silhouette_score = None
        
        return labels

    # =========================
    # VISUALIZACIÓN PCA
    # =========================

    def grafica_pca(self, guardar=True):
        if 'cluster' not in self.datos_normalizados.columns:
            print("Primero debes ejecutar clustering_kmeans()")
            return
            
        columnas = [c for c in self.datos_normalizados.columns if c not in ["year", "cluster"]]
        X = self.datos_normalizados[columnas]
        
        # Verificar columnas con datos
        columnas_validas = X.columns[~X.isna().all()].tolist()
        if len(columnas_validas) == 0:
            print("ERROR: No hay columnas con datos para PCA")
            return
        
        X = X[columnas_validas]
        
        # Imputar valores faltantes
        imputer = SimpleImputer(strategy='mean')
        X_imputed = imputer.fit_transform(X)
        
        # Determinar numero de componentes para PCA
        n_features = X_imputed.shape[1]
        n_components = min(2, n_features)
        
        if n_components < 2:
            print(f"ADVERTENCIA: Solo hay {n_features} caracteristicas, reduciendo PCA a {n_components} componente(s)")
        
        pca = PCA(n_components=n_components)
        componentes = pca.fit_transform(X_imputed)
        
        # Guardar componentes PCA para el reporte
        self.pca_componentes = componentes
        self.pca_explicacion = pca.explained_variance_ratio_
        
        plt.figure(figsize=(12, 8))
        
        if n_components == 2:
            scatter = plt.scatter(componentes[:, 0], componentes[:, 1],
                                c=self.datos_normalizados['cluster'], 
                                cmap='viridis', alpha=0.7, s=100)
            plt.colorbar(scatter, label='Cluster')
            plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} varianza)", fontsize=12)
            plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} varianza)", fontsize=12)
            varianza_total = sum(pca.explained_variance_ratio_)
            
            # Anadir etiquetas de años a los puntos (para 2 componentes)
            for i, año in enumerate(self.datos_normalizados['year']):
                plt.annotate(str(int(año)), (componentes[i, 0], componentes[i, 1]), 
                            fontsize=8, alpha=0.7)
        else:
            # Si solo hay 1 componente, graficar contra indice
            scatter = plt.scatter(range(len(componentes)), componentes[:, 0],
                                c=self.datos_normalizados['cluster'], 
                                cmap='viridis', alpha=0.7, s=100)
            plt.colorbar(scatter, label='Cluster')
            plt.xlabel("Indice de muestra (orden cronologico)", fontsize=12)
            plt.ylabel("PC1", fontsize=12)
            varianza_total = pca.explained_variance_ratio_[0]
            
            # Anadir etiquetas de años a los puntos (para 1 componente)
            for i, año in enumerate(self.datos_normalizados['year']):
                plt.annotate(str(int(año)), (i, componentes[i, 0]), 
                            fontsize=8, alpha=0.7)
        
        plt.title(f"Clusters proyectados en PCA ({varianza_total:.2%} varianza explicada)", fontsize=14)
        plt.grid(True, alpha=0.3)
        
        if guardar:
            ruta_guardado = self.carpeta_graficas / "pca_clusters.png"
            plt.savefig(ruta_guardado, dpi=300, bbox_inches='tight')
            print(f"Grafica guardada en: {ruta_guardado}")
        
        plt.show()
        
        print(f"Varianza explicada: {varianza_total:.2%}")
    # =========================
    # CLUSTER SOCIAL
    # =========================

    def clustering_social(self, k, guardar=True):
        columnas = ["total_matrimonios", "total_divorcios", "ratio_divorcios_matrimonios"]
        columnas = [c for c in columnas if c in self.datos_combinados.columns]

        X = self.datos_combinados[columnas].dropna()
        años = self.datos_combinados.loc[X.index, 'year']
        X_scaled = StandardScaler().fit_transform(X)

        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        
        # Guardar clusters sociales
        self.datos_combinados['cluster_social'] = np.nan
        self.datos_combinados.loc[X.index, 'cluster_social'] = labels

        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(X_scaled[:, 0], X_scaled[:, 1], 
                             c=labels, cmap='viridis', s=100, alpha=0.7)
        plt.colorbar(scatter, label='Cluster Social')
        plt.xlabel(f"{columnas[0]} (estandarizado)", fontsize=12)
        plt.ylabel(f"{columnas[1]} (estandarizado)", fontsize=12)
        plt.title(f"Clustering Social: Matrimonios y Divorcios (k={k})", fontsize=14)
        plt.grid(True, alpha=0.3)
        
        # Anadir etiquetas de años
        for i, año in enumerate(años):
            plt.annotate(str(int(año)), (X_scaled[i, 0], X_scaled[i, 1]), 
                        fontsize=8, alpha=0.7)
        
        if guardar:
            ruta_guardado = self.carpeta_graficas / "clustering_social.png"
            plt.savefig(ruta_guardado, dpi=300, bbox_inches='tight')
            print(f"Grafica guardada en: {ruta_guardado}")
        
        plt.show()
        
        # Calcular silhouette score para clustering social
        if len(set(labels)) > 1:
            score_social = silhouette_score(X_scaled, labels)
            print(f"Silhouette Score (Social): {score_social:.3f}")
            self.silhouette_score_social = score_social
        else:
            self.silhouette_score_social = None

        return labels

    # =========================
    # GENERAR REPORTE
    # =========================

    def generar_reporte(self):
        """Genera un reporte completo del analisis"""
        
        print("\n" + "="*60)
        print(" GENERANDO REPORTE DEL ANALISIS ".center(60, "="))
        print("="*60 + "\n")
        
        # Crear archivo de reporte
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ruta_reporte = self.carpeta_resultados / f"reporte_ine_{timestamp}.txt"
        ruta_csv = self.carpeta_resultados / f"datos_completos_{timestamp}.csv"
        
        with open(ruta_reporte, 'w', encoding='utf-8') as f:
            # Escribir encabezado
            f.write("="*70 + "\n")
            f.write(" REPORTE DE ANALISIS DE DATOS DEL INE ".center(70, "=") + "\n")
            f.write("="*70 + "\n\n")
            f.write(f"Fecha de generacion: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Directorio de datos: {self.ruta_datos}\n\n")
            
            # 1. RESUMEN DE DATOS CARGADOS
            f.write("-"*70 + "\n")
            f.write("1. RESUMEN DE DATOS CARGADOS\n")
            f.write("-"*70 + "\n\n")
            f.write(f"Total de datasets cargados: {len(self.datos)}\n")
            f.write(f"Categorias de datasets:\n")
            
            # Contar datasets por categoria
            categorias_contadas = {cat: 0 for cat in ["nacimientos", "defunciones", 
                                                      "defunciones_fetales", "matrimonios", 
                                                      "divorcios", "violencia_intrafamiliar"]}
            for nombre in self.datos.keys():
                for cat in categorias_contadas:
                    if cat in nombre.lower():
                        categorias_contadas[cat] += 1
                        
            for cat, count in categorias_contadas.items():
                f.write(f"  - {cat}: {count} archivos\n")
            
            # 2. DATOS COMBINADOS
            f.write("\n" + "-"*70 + "\n")
            f.write("2. DATOS COMBINADOS POR AÑO\n")
            f.write("-"*70 + "\n\n")
            
            f.write(f"Rango de años: {int(self.datos_combinados['year'].min())} - {int(self.datos_combinados['year'].max())}\n")
            f.write(f"Total de años: {len(self.datos_combinados)}\n\n")
            
            f.write("Estadisticas descriptivas de variables principales:\n")
            columnas_totales = [c for c in self.datos_combinados.columns if c.startswith('total_')]
            for col in columnas_totales:
                f.write(f"\n{col}:\n")
                f.write(f"  Media: {self.datos_combinados[col].mean():.2f}\n")
                f.write(f"  Mediana: {self.datos_combinados[col].median():.2f}\n")
                f.write(f"  Desv. Estandar: {self.datos_combinados[col].std():.2f}\n")
                f.write(f"  Minimo: {self.datos_combinados[col].min():.2f}\n")
                f.write(f"  Maximo: {self.datos_combinados[col].max():.2f}\n")
            
            # 3. RESULTADOS DEL CLUSTERING
            f.write("\n" + "-"*70 + "\n")
            f.write("3. RESULTADOS DEL CLUSTERING\n")
            f.write("-"*70 + "\n\n")
            
            if hasattr(self, 'silhouette_score'):
                f.write(f"Clustering General (k=3):\n")
                f.write(f"  Silhouette Score: {self.silhouette_score:.3f}\n")
                
                # Distribucion de clusters
                f.write("\n  Distribucion de clusters por año:\n")
                for cluster in sorted(self.datos_combinados['cluster'].unique()):
                    años_cluster = self.datos_combinados[self.datos_combinados['cluster'] == cluster]['year'].tolist()
                    f.write(f"    Cluster {int(cluster)}: {len(años_cluster)} años - {años_cluster}\n")
            
            if hasattr(self, 'silhouette_score_social'):
                f.write(f"\nClustering Social (Matrimonios/Divorcios):\n")
                f.write(f"  Silhouette Score: {self.silhouette_score_social:.3f}\n")
                
                # Distribucion de clusters sociales
                f.write("\n  Distribucion de clusters sociales por año:\n")
                cluster_social_validos = self.datos_combinados.dropna(subset=['cluster_social'])
                for cluster in sorted(cluster_social_validos['cluster_social'].unique()):
                    años_cluster = cluster_social_validos[cluster_social_validos['cluster_social'] == cluster]['year'].tolist()
                    f.write(f"    Cluster Social {int(cluster)}: {len(años_cluster)} años - {años_cluster}\n")
            
            # 4. METRICAS CLAVE
            f.write("\n" + "-"*70 + "\n")
            f.write("4. METRICAS CLAVE POR AÑO\n")
            f.write("-"*70 + "\n\n")
            
            columnas_metricas = ['year'] + [c for c in self.datos_combinados.columns 
                                           if any(x in c for x in ['ratio', 'crecimiento', 'vegetativo'])]
            
            df_metricas = self.datos_combinados[columnas_metricas].round(2)
            f.write(df_metricas.to_string(index=False))
            
            # 5. CONCLUSIONES
            f.write("\n\n" + "-"*70 + "\n")
            f.write("5. CONCLUSIONES DEL ANALISIS\n")
            f.write("-"*70 + "\n\n")
            
            # Identificar años atipicos
            if hasattr(self, 'silhouette_score') and self.silhouette_score is not None:
                f.write("* Años atipicos por cluster:\n")
                for cluster in sorted(self.datos_combinados['cluster'].unique()):
                    años_cluster = self.datos_combinados[self.datos_combinados['cluster'] == cluster]['year'].tolist()
                    if len(años_cluster) <= 2:  # Considerar atipicos si hay pocos años
                        f.write(f"  - Cluster {int(cluster)} (años {años_cluster}) podria representar periodos atipicos\n")
            
            # Tendencias generales
            f.write("\n* Tendencias generales:\n")
            
            if 'ratio_divorcios_matrimonios' in self.datos_combinados.columns:
                ratio_actual = self.datos_combinados['ratio_divorcios_matrimonios'].iloc[-1]
                ratio_inicial = self.datos_combinados['ratio_divorcios_matrimonios'].iloc[0]
                cambio = ratio_actual - ratio_inicial
                if cambio > 0:
                    f.write(f"  - La proporcion de divorcios respecto a matrimonios ha aumentado en {cambio:.1f} puntos porcentuales\n")
                else:
                    f.write(f"  - La proporcion de divorcios respecto a matrimonios ha disminuido en {abs(cambio):.1f} puntos porcentuales\n")
            
            if 'crecimiento_vegetativo' in self.datos_combinados.columns:
                crecimiento_promedio = self.datos_combinados['crecimiento_vegetativo'].mean()
                if crecimiento_promedio > 0:
                    f.write(f"  - Crecimiento vegetativo promedio positivo: {crecimiento_promedio:.0f} personas/año\n")
                else:
                    f.write(f"  - Crecimiento vegetativo promedio negativo: {crecimiento_promedio:.0f} personas/año\n")
            
            # Recomendaciones
            f.write("\n* Recomendaciones:\n")
            if hasattr(self, 'silhouette_score') and self.silhouette_score > 0.5:
                f.write("  - Los clusters formados son consistentes y podrian usarse para segmentacion temporal\n")
            else:
                f.write("  - Considerar revisar las variables utilizadas para mejorar la separacion de clusters\n")
            
            if 'violencia_intrafamiliar' in str(self.datos_combinados.columns):
                f.write("  - Incluir analisis mas detallado de violencia intrafamiliar en estudios futuros\n")
            
            f.write("\n" + "="*70 + "\n")
            f.write(" FIN DEL REPORTE ".center(70, "=") + "\n")
            f.write("="*70 + "\n")
        
        # Guardar datos completos en CSV
        self.datos_combinados.to_csv(ruta_csv, index=False, encoding='utf-8')
        
        print(f"\nReporte guardado en: {ruta_reporte}")
        print(f"Datos completos guardados en: {ruta_csv}")
        print(f"Graficas guardadas en: {self.carpeta_graficas}")
        
        # Mostrar resumen en consola
        print("\n" + "="*60)
        print(" RESUMEN DEL ANALISIS ".center(60, "="))
        print("="*60)
        print(f"\nTotal de años analizados: {len(self.datos_combinados)}")
        print(f"Rango temporal: {int(self.datos_combinados['year'].min())} - {int(self.datos_combinados['year'].max())}")
        
        if hasattr(self, 'silhouette_score') and self.silhouette_score is not None:
            print(f"\nCalidad del clustering general:")
            print(f"   * Silhouette Score: {self.silhouette_score:.3f}")
            if self.silhouette_score > 0.7:
                print("   * Interpretacion: Excelente separacion de clusters")
            elif self.silhouette_score > 0.5:
                print("   * Interpretacion: Buena separacion de clusters")
            elif self.silhouette_score > 0.3:
                print("   * Interpretacion: Separacion moderada")
            else:
                print("   * Interpretacion: Separacion debil")
        
        if hasattr(self, 'silhouette_score_social') and self.silhouette_score_social is not None:
            print(f"\nCalidad del clustering social:")
            print(f"   * Silhouette Score: {self.silhouette_score_social:.3f}")
        
        print(f"\nLos resultados completos estan disponibles en:")
        print(f"   {self.carpeta_resultados}")
        print("\n" + "="*60)



        print("="*60)
        print(" TABLAS DE FRECUENCIA ".center(60, "="))
        print("="*60)

        # Asegurarnos de que las columnas existen
        if 'total_matrimonios' in analizador.datos_combinados.columns:
            print("\n--- TABLA DE FRECUENCIA: MATRIMONIOS ---")
            # Mostramos los años y el total de matrimonios de forma ordenada
            tabla_matrimonios = analizador.datos_combinados[['year', 'total_matrimonios']].copy()
            tabla_matrimonios = tabla_matrimonios.sort_values('year')
            print(tabla_matrimonios.to_string(index=False))
            
            # Podemos añadir un pequeño resumen estadístico
            print("\nEstadísticas de Matrimonios:")
            print(f"  Año con más matrimonios: {tabla_matrimonios.loc[tabla_matrimonios['total_matrimonios'].idxmax(), 'year']} ({tabla_matrimonios['total_matrimonios'].max():,.0f})")
            print(f"  Año con menos matrimonios: {tabla_matrimonios.loc[tabla_matrimonios['total_matrimonios'].idxmin(), 'year']} ({tabla_matrimonios['total_matrimonios'].min():,.0f})")
            print(f"  Promedio de matrimonios: {tabla_matrimonios['total_matrimonios'].mean():,.0f}")
        else:
            print("La columna 'total_matrimonios' no está disponible.")

        if 'total_divorcios' in analizador.datos_combinados.columns:
            print("\n--- TABLA DE FRECUENCIA: DIVORCIOS ---")
            tabla_divorcios = analizador.datos_combinados[['year', 'total_divorcios']].copy()
            tabla_divorcios = tabla_divorcios.sort_values('year')
            print(tabla_divorcios.to_string(index=False))
            
            print("\nEstadísticas de Divorcios:")
            print(f"  Año con más divorcios: {tabla_divorcios.loc[tabla_divorcios['total_divorcios'].idxmax(), 'year']} ({tabla_divorcios['total_divorcios'].max():,.0f})")
            print(f"  Año con menos divorcios: {tabla_divorcios.loc[tabla_divorcios['total_divorcios'].idxmin(), 'year']} ({tabla_divorcios['total_divorcios'].min():,.0f})")
            print(f"  Promedio de divorcios: {tabla_divorcios['total_divorcios'].mean():,.0f}")
        else:
            print("La columna 'total_divorcios' no está disponible.")


        # --- gráficos exploratorios ---

        print("\n" + "="*60)
        print(" GRÁFICOS EXPLORATORIOS ".center(60, "="))
        print("="*60)

        # Configuración general de gráficos
        plt.style.use('ggplot') 
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Análisis Exploratorio de Datos del INE', fontsize=16)

        # 1. Evolución temporal de Matrimonios y Divorcios
        ax1 = axes[0, 0]
        if 'total_matrimonios' in analizador.datos_combinados.columns:
            ax1.plot(analizador.datos_combinados['year'], analizador.datos_combinados['total_matrimonios'], 
                    marker='o', linestyle='-', label='Matrimonios', color='blue')
        if 'total_divorcios' in analizador.datos_combinados.columns:
            ax1.plot(analizador.datos_combinados['year'], analizador.datos_combinados['total_divorcios'], 
                    marker='s', linestyle='-', label='Divorcios', color='red')
        ax1.set_xlabel('Año')
        ax1.set_ylabel('Cantidad Total')
        ax1.set_title('Evolución de Matrimonios y Divorcios')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Ratio de Divorcios por Matrimonio
        ax2 = axes[0, 1]
        if 'ratio_divorcios_matrimonios' in analizador.datos_combinados.columns:
            ax2.bar(analizador.datos_combinados['year'], analizador.datos_combinados['ratio_divorcios_matrimonios'], 
                    color='purple', alpha=0.7)
            ax2.set_xlabel('Año')
            ax2.set_ylabel('Ratio (%)')
            ax2.set_title('Ratio de Divorcios por cada 100 Matrimonios')
            ax2.grid(True, alpha=0.3, axis='y')
        else:
            ax2.text(0.5, 0.5, 'Datos no disponibles', ha='center', va='center')

        # 3. Crecimiento Vegetativo (Nacimientos - Defunciones)
        ax3 = axes[1, 0]
        if 'crecimiento_vegetativo' in analizador.datos_combinados.columns:
            colores = ['green' if x > 0 else 'red' for x in analizador.datos_combinados['crecimiento_vegetativo']]
            ax3.bar(analizador.datos_combinados['year'], analizador.datos_combinados['crecimiento_vegetativo'], 
                    color=colores, alpha=0.7)
            ax3.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
            ax3.set_xlabel('Año')
            ax3.set_ylabel('Crecimiento Vegetativo')
            ax3.set_title('Crecimiento Vegetativo (Nacimientos - Defunciones)')
            ax3.grid(True, alpha=0.3, axis='y')
        else:
            ax3.text(0.5, 0.5, 'Datos no disponibles', ha='center', va='center')

        # 4. Boxplot de las variables totales para ver distribuciones y outliers
        ax4 = axes[1, 1]
        columnas_totales = [col for col in analizador.datos_combinados.columns if col.startswith('total_')]
        if columnas_totales:
            # Escalar para que sean comparables en un boxplot (opcional, pero útil)
            df_box = analizador.datos_combinados[columnas_totales].dropna()
            # Estandarización simple para visualización conjunta
            df_box_std = (df_box - df_box.mean()) / df_box.std()
            df_box_std.boxplot(ax=ax4, rot=45)
            ax4.set_title('Distribución de Variables (Estandarizadas)')
            ax4.set_ylabel('Desviaciones de la Media')
        else:
            ax4.text(0.5, 0.5, 'Datos no disponibles', ha='center', va='center')

        plt.tight_layout()
        plt.show()
        # Guardar la figura exploratoria
        ruta_guardado_exploratorio = analizador.carpeta_graficas / "graficos_exploratorios.png"
        fig.savefig(ruta_guardado_exploratorio, dpi=300, bbox_inches='tight')
        print(f"Gráficos exploratorios guardados en: {ruta_guardado_exploratorio}")



        print("\n" + "="*60)
        print(" VALIDACIÓN DE HIPÓTESIS ".center(60, "="))
        print("="*60)

        print("\n--- HIPÓTESIS 1: El ratio de divorcios ha aumentado constantemente ---")
        if 'ratio_divorcios_matrimonios' in analizador.datos_combinados.columns:
            # Crear columna de década
            analizador.datos_combinados['decada'] = (analizador.datos_combinados['year'] // 10) * 10
            
            # Calcular ratio promedio por década
            ratio_por_decada = analizador.datos_combinados.groupby('decada')['ratio_divorcios_matrimonios'].mean().reset_index()
            
            print("\nRatio promedio de Divorcios/Matrimonios por Década:")
            print(ratio_por_decada.to_string(index=False))
            
            # Gráfico de barras
            plt.figure(figsize=(10, 6))
            plt.bar(ratio_por_decada['decada'].astype(str), ratio_por_decada['ratio_divorcios_matrimonios'], 
                    color='skyblue', edgecolor='black')
            plt.xlabel('Década')
            plt.ylabel('Ratio Promedio (%)')
            plt.title('Evolución del Ratio de Divorcios por Década')
            plt.grid(True, alpha=0.3, axis='y')
            
            ruta_guardado_h1 = analizador.carpeta_graficas / "hipotesis1_ratio_por_decada.png"
            plt.savefig(ruta_guardado_h1, dpi=300, bbox_inches='tight')
            plt.show()
            
            # Conclusión automática
            if len(ratio_por_decada) > 1:
                if ratio_por_decada['ratio_divorcios_matrimonios'].is_monotonic_increasing:
                    print("\n--> Conclusión: La hipótesis se CONFIRMA. El ratio ha aumentado constantemente.")
                else:
                    print("\n--> Conclusión: La hipótesis se REFUTA. El ratio NO ha aumentado de forma constante (puede haber fluctuaciones o descensos).")
        else:
            print("Datos necesarios no disponibles.")



        print("\n--- HIPÓTESIS 2: Años con más matrimonios tienen más nacimientos ---")
        if 'total_matrimonios' in analizador.datos_combinados.columns and 'total_nacimientos' in analizador.datos_combinados.columns:
            # Correlación
            correlacion = analizador.datos_combinados['total_matrimonios'].corr(analizador.datos_combinados['total_nacimientos'])
            print(f"Correlación entre Matrimonios y Nacimientos: {correlacion:.3f}")
            
            # Gráfico de dispersión
            plt.figure(figsize=(8, 6))
            plt.scatter(analizador.datos_combinados['total_matrimonios'], 
                        analizador.datos_combinados['total_nacimientos'], 
                        c=analizador.datos_combinados['year'], cmap='viridis', alpha=0.7, s=80)
            plt.colorbar(label='Año')
            plt.xlabel('Total de Matrimonios')
            plt.ylabel('Total de Nacimientos')
            plt.title('Relación entre Matrimonios y Nacimientos')
            plt.grid(True, alpha=0.3)
            
            # Añadir línea de tendencia
            z = np.polyfit(analizador.datos_combinados['total_matrimonios'].dropna(), 
                        analizador.datos_combinados['total_nacimientos'].dropna(), 1)
            p = np.poly1d(z)
            plt.plot(analizador.datos_combinados['total_matrimonios'].sort_values(), 
                    p(analizador.datos_combinados['total_matrimonios'].sort_values()), 
                    "r--", alpha=0.8, label=f'Tendencia (r={correlacion:.2f})')
            plt.legend()
            
            ruta_guardado_h2 = analizador.carpeta_graficas / "hipotesis2_matrimonios_vs_nacimientos.png"
            plt.savefig(ruta_guardado_h2, dpi=300, bbox_inches='tight')
            plt.show()
            
            if correlacion > 0.5:
                print("--> Conclusión: La hipótesis se CONFIRMA. Existe una correlación positiva fuerte.")
            elif correlacion > 0:
                print("--> Conclusión: La hipótesis se CONFIRMA DÉBILMENTE. La correlación es positiva pero débil.")
            else:
                print("--> Conclusión: La hipótesis se REFUTA. La correlación es negativa o nula.")
        else:
            print("Datos necesarios no disponibles.")


        print("\n--- HIPÓTESIS 3: El crecimiento vegetativo varía significativamente entre los clusters ---")
        if 'crecimiento_vegetativo' in analizador.datos_combinados.columns and 'cluster' in analizador.datos_combinados.columns:
            # Calcular crecimiento promedio por cluster
            crecimiento_por_cluster = analizador.datos_combinados.groupby('cluster')['crecimiento_vegetativo'].agg(['mean', 'std', 'count']).reset_index()
            
            print("\nEstadísticas de Crecimiento Vegetativo por Cluster:")
            print(crecimiento_por_cluster.to_string(index=False))
            
            # Gráfico de barras con error
            plt.figure(figsize=(10, 6))
            clusters_ordenados = sorted(analizador.datos_combinados['cluster'].unique())
            medias = [crecimiento_por_cluster[crecimiento_por_cluster['cluster']==c]['mean'].values[0] for c in clusters_ordenados]
            errores = [crecimiento_por_cluster[crecimiento_por_cluster['cluster']==c]['std'].values[0] for c in clusters_ordenados]
            
            plt.bar(clusters_ordenados, medias, yerr=errores, capsize=5, color='teal', alpha=0.7, edgecolor='black')
            plt.axhline(y=0, color='red', linestyle='--', linewidth=1)
            plt.xlabel('Cluster')
            plt.ylabel('Crecimiento Vegetativo Promedio')
            plt.title('Crecimiento Vegetativo por Cluster')
            plt.grid(True, alpha=0.3, axis='y')
            
            ruta_guardado_h3 = analizador.carpeta_graficas / "hipotes3_crecimiento_por_cluster.png"
            plt.savefig(ruta_guardado_h3, dpi=300, bbox_inches='tight')
            plt.show()
            
            # Comprobar si las medias son significativamente diferentes (ANOVA simple)
            from scipy import stats
            grupos = [analizador.datos_combinados[analizador.datos_combinados['cluster']==c]['crecimiento_vegetativo'].dropna() for c in clusters_ordenados if len(analizador.datos_combinados[analizador.datos_combinados['cluster']==c])>0]
            if len(grupos) > 1:
                f_stat, p_value = stats.f_oneway(*grupos)
                print(f"Prueba ANOVA - F-statistic: {f_stat:.3f}, P-value: {p_value:.4f}")
                if p_value < 0.05:
                    print("--> Conclusión: La hipótesis se CONFIRMA. Hay diferencias significativas en el crecimiento vegetativo entre los clusters.")
                else:
                    print("--> Conclusión: La hipótesis se REFUTA. No hay diferencias estadísticamente significativas.")
        else:
            print("Datos necesarios no disponibles.")




        print("\n--- HIPÓTESIS 4: El cluster social y el cluster general son equivalentes ---")
        if 'cluster' in analizador.datos_combinados.columns and 'cluster_social' in analizador.datos_combinados.columns:
            # Tabla de contingencia
            tabla_contingencia = pd.crosstab(
                analizador.datos_combinados['cluster'].fillna(-1).astype(int), 
                analizador.datos_combinados['cluster_social'].fillna(-1).astype(int),
                rownames=['Cluster General'], 
                colnames=['Cluster Social']
            )
            
            print("\nTabla de Contingencia (Frecuencias):")
            print(tabla_contingencia)
            
            # Calcular el índice de acuerdo (porcentaje de años en la misma diagonal)
            # Nota: Los números de cluster pueden no coincidir, así que esto es una aproximación conceptual.
            # Una mejor medida es el Adjusted Rand Score.
            from sklearn.metrics import adjusted_rand_score
            
            # Eliminar filas con NaN para la comparación
            df_comparacion = analizador.datos_combinados.dropna(subset=['cluster', 'cluster_social']).copy()
            if len(df_comparacion) > 0:
                ari_score = adjusted_rand_score(df_comparacion['cluster'], df_comparacion['cluster_social'])
                print(f"\nAdjusted Rand Index (ARI) entre clusters: {ari_score:.3f}")
                print("(ARI = 1 es concordancia perfecta, ARI ≈ 0 es aleatorio)")
                
                if ari_score > 0.7:
                    print("--> Conclusión: La hipótesis se CONFIRMA. Los clusters son muy similares.")
                elif ari_score > 0.3:
                    print("--> Conclusión: La hipótesis se CONFIRMA PARCIALMENTE. Hay una concordancia moderada.")
                else:
                    print("--> Conclusión: La hipótesis se REFUTA. Los clusters no están alineados.")
        else:
            print("Datos necesarios no disponibles.")

        print("\n--- HIPÓTESIS 5: Existe un punto de inflexión en la tendencia de divorcios ---")
        if 'total_divorcios' in analizador.datos_combinados.columns:
            # Usaremos una media móvil para suavizar y luego ver las diferencias
            analizador.datos_combinados['divorcios_suavizado'] = analizador.datos_combinados['total_divorcios'].rolling(window=3, center=True).mean()
            
            # Calcular la diferencia (primera derivada aproximada)
            analizador.datos_combinados['tendencia_divorcios'] = analizador.datos_combinados['divorcios_suavizado'].diff()
            
            # Encontrar el año donde la tendencia cambia de positiva a negativa o viceversa (cruces por cero)
            analizador.datos_combinados['cambio_tendencia'] = np.sign(analizador.datos_combinados['tendencia_divorcios']).diff() != 0
            
            plt.figure(figsize=(12, 6))
            plt.plot(analizador.datos_combinados['year'], analizador.datos_combinados['total_divorcios'], 
                    'o-', label='Divorcios Reales', alpha=0.5)
            plt.plot(analizador.datos_combinados['year'], analizador.datos_combinados['divorcios_suavizado'], 
                    's-', label='Divorcios (Media Móvil 3 años)', linewidth=2, color='red')
            
            # Marcar posibles puntos de inflexión
            años_inflexion = analizador.datos_combinados[analizador.datos_combinados['cambio_tendencia']]['year']
            for año in años_inflexion:
                plt.axvline(x=año, color='green', linestyle='--', alpha=0.5)
            
            plt.xlabel('Año')
            plt.ylabel('Total de Divorcios')
            plt.title('Detección de Puntos de Inflexión en la Tendencia de Divorcios')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            ruta_guardado_h5 = analizador.carpeta_graficas / "hipotesis5_punto_inflexion_divorcios.png"
            plt.savefig(ruta_guardado_h5, dpi=300, bbox_inches='tight')
            plt.show()
            
            if len(años_inflexion) > 0:
                print(f"Posibles años de inflexión detectados: {años_inflexion.tolist()}")
                print("--> Conclusión: La hipótesis se CONFIRMA. Se detectan cambios en la tendencia.")
            else:
                print("--> Conclusión: La hipótesis se REFUTA. No se detectan puntos de inflexión claros.")
        else:
            print("Datos necesarios no disponibles.")
# =========================
# EJECUCION PRINCIPAL
# =========================

if __name__ == "__main__":
    
    print("\n" + "="*60)
    print(" ANALISIS DE DATOS DEL INE ".center(60, "="))
    print("="*60 + "\n")
    
    # Inicializar analizador
    analizador = AnalizadorINERobusto("./datos_ine")
    
    # Cargar datos
    print("\nCARGANDO DATOS...")
    analizador.cargar_datos()
    
    # Combinar datos
    print("\nCOMBINANDO DATOS...")
    analizador.combinar_datos()
    
    # Calcular metricas
    print("\nCALCULANDO METRICAS...")
    analizador.calcular_metricas()
    
    # Normalizar variables
    print("\nNORMALIZANDO VARIABLES...")
    analizador.normalizar_variables()
    
    # Verificar datos
    print("\nVERIFICACION DE DATOS:")
    print(analizador.datos_normalizados.count())
    
    # Metodo del codo
    print("\nMETODO DEL CODO...")
    analizador.metodo_codo(guardar=True)
    
    # Clustering general
    print("\nAPLICANDO CLUSTERING GENERAL (k=3)...")
    analizador.clustering_kmeans(k=3)
    
    # Visualizacion PCA
    print("\nVISUALIZACION PCA...")
    analizador.grafica_pca(guardar=True)
    
    # Clustering social
    print("\nAPLICANDO CLUSTERING SOCIAL (k=3)...")
    analizador.clustering_social(k=3, guardar=True)
    
    # Generar reporte
    print("\nGENERANDO REPORTE COMPLETO...")
    analizador.generar_reporte()
    
    print("\n" + "="*60)
    print(" ANALISIS COMPLETADO ".center(60, "="))
    print("="*60)