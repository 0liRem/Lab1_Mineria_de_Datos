"""
Script de Exploración Inicial de Archivos .SAV del INE Guatemala

Este script te ayuda a entender qué datos tienes antes de hacer el análisis completo.
Úsalo para conocer la estructura de tus archivos y decidir qué variables analizar.
"""

import pandas as pd
import pyreadstat
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def explorar_archivos_sav_detallado(ruta_datos):
    """
    Explora archivos .sav y genera un reporte detallado de su contenido
    
    Args:
        ruta_datos (str): Ruta donde están los archivos .sav
    """
    ruta = Path(ruta_datos)
    archivos_sav = list(ruta.glob('*.sav'))
    
    if not archivos_sav:
        print(f" No se encontraron archivos .sav en: {ruta_datos}")
        print(f"   Por favor verifica que:")
        print(f"   1. La ruta sea correcta")
        print(f"   2. Los archivos tengan extensión .sav")
        return
    
    print("=" * 100)
    print(f"EXPLORACIÓN DE {len(archivos_sav)} ARCHIVOS .SAV DEL INE GUATEMALA")
    print("=" * 100)
    
    resumen_archivos = []
    
    for i, archivo in enumerate(archivos_sav, 1):
        print(f"\n{'' * 100}")
        print(f"ARCHIVO {i}/{len(archivos_sav)}: {archivo.name}")
        print(f"{'' * 100}\n")
        
        try:
            # Leer archivo con metadatos
            df, meta = pyreadstat.read_sav(str(archivo))
            
            # ========================================
            # INFORMACIÓN GENERAL
            # ========================================
            print("INFORMACIÓN GENERAL")
            print("─" * 100)
            print(f"Nombre del archivo: {archivo.name}")
            print(f"Tamaño del archivo: {archivo.stat().st_size / 1024 / 1024:.2f} MB")
            print(f"Total de registros (filas): {len(df):,}")
            print(f"Total de variables (columnas): {len(df.columns)}")
            print(f"Memoria en uso: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
            
            # ========================================
            # TIPOS DE DATOS
            # ========================================
            print(f"\nDISTRIBUCIÓN DE TIPOS DE DATOS")
            print("─" * 100)
            tipos = df.dtypes.value_counts()
            for tipo, cantidad in tipos.items():
                porcentaje = (cantidad / len(df.columns)) * 100
                print(f"   {str(tipo):20s}: {cantidad:3d} variables ({porcentaje:5.1f}%)")
            
            # ========================================
            # VARIABLES CLAVE (TEMPORALES)
            # ========================================
            print(f"\nVARIABLES TEMPORALES DETECTADAS")
            print("─" * 100)
            vars_temporales = []
            for col in df.columns:
                col_lower = col.lower()
                if any(palabra in col_lower for palabra in ['year', 'ano', 'anio', 'fecha', 'date', 'mes', 'month']):
                    vars_temporales.append(col)
                    valores_unicos = df[col].nunique()
                    if valores_unicos <= 50:  # Mostrar valores si no son muchos
                        valores = sorted(df[col].dropna().unique())[:20]
                        print(f"   ✓ {col}: {valores}")
                        if len(df[col].dropna().unique()) > 20:
                            print(f"     ... y {len(df[col].dropna().unique()) - 20} valores más")
                    else:
                        rango = f"{df[col].min()} a {df[col].max()}"
                        print(f"   ✓ {col}: {valores_unicos} valores únicos (rango: {rango})")
            
            if not vars_temporales:
                print("    No se detectaron variables temporales automáticamente")
                print(" Revisa manualmente si hay columnas de fecha/año con otros nombres")
            
            # ========================================
            # LISTADO COMPLETO DE VARIABLES
            # ========================================
            print(f"\n LISTADO COMPLETO DE VARIABLES")
            print("─" * 100)
            print(f"{'#':>3} | {'Nombre Variable':<40} | {'Tipo':<12} | {'Únicos':>8} | {'Nulos %':>8} | {'Muestra de valores'}")
            print("─" * 100)
            
            for j, col in enumerate(df.columns, 1):
                tipo = str(df[col].dtype)
                n_unicos = df[col].nunique()
                n_nulos = df[col].isna().sum()
                pct_nulos = (n_nulos / len(df)) * 100
                
                # Muestra de valores
                if df[col].dtype in ['object', 'category']:
                    muestra = df[col].dropna().head(2).tolist()
                    muestra_str = ', '.join([str(v)[:20] for v in muestra])
                else:
                    if n_unicos <= 10:
                        muestra = sorted(df[col].dropna().unique())[:5]
                        muestra_str = ', '.join([str(v) for v in muestra])
                    else:
                        min_val = df[col].min()
                        max_val = df[col].max()
                        muestra_str = f"rango: {min_val} - {max_val}"
                
                print(f"{j:3d} | {col:<40} | {tipo:<12} | {n_unicos:8,} | {pct_nulos:7.1f}% | {muestra_str[:50]}")
            
            # ========================================
            # VARIABLES CON ETIQUETAS (si existen)
            # ========================================
            if hasattr(meta, 'column_names_to_labels') and meta.column_names_to_labels:
                print(f"\n  ETIQUETAS DE VARIABLES (METADATOS)")
                print("─" * 100)
                for col, label in list(meta.column_names_to_labels.items())[:30]:
                    print(f"   {col:30s} → {label}")
                if len(meta.column_names_to_labels) > 30:
                    print(f"   ... y {len(meta.column_names_to_labels) - 30} etiquetas más")
            
            # ========================================
            # VARIABLES CON VALORES ETIQUETADOS
            # ========================================
            if hasattr(meta, 'variable_value_labels') and meta.variable_value_labels:
                print(f"\n VARIABLES CON VALORES CODIFICADOS")
                print("─" * 100)
                for var, labels in list(meta.variable_value_labels.items())[:10]:
                    print(f"   Variable: {var}")
                    for codigo, etiqueta in list(labels.items())[:10]:
                        print(f"      {codigo} = {etiqueta}")
                    if len(labels) > 10:
                        print(f"      ... y {len(labels) - 10} códigos más")
                    print()
            
            # ========================================
            # ESTADÍSTICAS BÁSICAS (NUMÉRICAS)
            # ========================================
            columnas_numericas = df.select_dtypes(include=['int64', 'float64']).columns
            if len(columnas_numericas) > 0:
                print(f"\n ESTADÍSTICAS DESCRIPTIVAS (Top 10 variables numéricas)")
                print("─" * 100)
                stats = df[columnas_numericas].describe().T
                stats['suma'] = df[columnas_numericas].sum()
                stats = stats.sort_values('suma', ascending=False).head(10)
                print(stats[['count', 'mean', 'std', 'min', 'max', 'suma']].to_string())
            
            # ========================================
            # VARIABLES CATEGÓRICAS MÁS IMPORTANTES
            # ========================================
            columnas_categoricas = df.select_dtypes(include=['object', 'category']).columns
            if len(columnas_categoricas) > 0:
                print(f"\n DISTRIBUCIÓN DE VARIABLES CATEGÓRICAS (Top 5)")
                print("─" * 100)
                for col in columnas_categoricas[:5]:
                    print(f"\n   Variable: {col}")
                    frecuencias = df[col].value_counts().head(10)
                    total = len(df[col].dropna())
                    for valor, freq in frecuencias.items():
                        pct = (freq / total) * 100
                        print(f"      {str(valor)[:50]:50s}: {freq:7,} ({pct:5.1f}%)")
                    if df[col].nunique() > 10:
                        print(f"      ... y {df[col].nunique() - 10} categorías más")
            
            # ========================================
            # VALORES FALTANTES
            # ========================================
            print(f"\n  ANÁLISIS DE VALORES FALTANTES")
            print("─" * 100)
            total_nulls = df.isna().sum().sum()
            total_cells = len(df) * len(df.columns)
            pct_nulls = (total_nulls / total_cells) * 100
            print(f"Total de valores faltantes: {total_nulls:,} de {total_cells:,} celdas ({pct_nulls:.2f}%)")
            
            # Variables con más valores faltantes
            vars_con_nulls = df.isna().sum()
            vars_con_nulls = vars_con_nulls[vars_con_nulls > 0].sort_values(ascending=False)
            
            if len(vars_con_nulls) > 0:
                print(f"\nVariables con valores faltantes (Top 10):")
                for var, count in vars_con_nulls.head(10).items():
                    pct = (count / len(df)) * 100
                    barra = "" * int(pct / 5) + "░" * (20 - int(pct / 5))
                    print(f"   {var:40s}: {count:7,} ({pct:5.1f}%) {barra}")
            else:
                print("    ¡No hay valores faltantes en este dataset!")
            
            # ========================================
            # MUESTRA DE REGISTROS
            # ========================================
            print(f"\n  MUESTRA DE DATOS (Primeras 5 filas)")
            print("─" * 100)
            # Mostrar solo las primeras 10 columnas para que quepa en pantalla
            print(df.iloc[:5, :min(10, len(df.columns))].to_string())
            if len(df.columns) > 10:
                print(f"\n   ... y {len(df.columns) - 10} columnas más (no mostradas)")
            
            # Guardar resumen
            resumen_archivos.append({
                'Archivo': archivo.name,
                'Registros': len(df),
                'Variables': len(df.columns),
                'Tamaño_MB': archivo.stat().st_size / 1024 / 1024,
                'Variables_Temporales': ', '.join(vars_temporales) if vars_temporales else 'Ninguna detectada',
                'Pct_Nulos': pct_nulos
            })
            
        except Exception as e:
            print(f"ERROR al procesar {archivo.name}:")
            print(f"   {str(e)}")
            print(f"   Tipo de error: {type(e).__name__}")
    
    # ========================================
    # RESUMEN GENERAL
    # ========================================
    print(f"\n\n{'' * 100}")
    print(" RESUMEN GENERAL DE TODOS LOS ARCHIVOS")
    print(f"{'' * 100}\n")
    
    if resumen_archivos:
        df_resumen = pd.DataFrame(resumen_archivos)
        print(df_resumen.to_string(index=False))
        
        print(f"\nESTADÍSTICAS GENERALES:")
        print(f"   Total de archivos procesados: {len(resumen_archivos)}")
        print(f"   Total de registros: {df_resumen['Registros'].sum():,}")
        print(f"   Promedio de variables por archivo: {df_resumen['Variables'].mean():.1f}")
        print(f"   Tamaño total: {df_resumen['Tamaño_MB'].sum():.2f} MB")
    
    print(f"\n{'' * 100}")
    print(" EXPLORACIÓN COMPLETADA")
    print(f"{'' * 100}\n")


if __name__ == "__main__":
    # ========================================
    # CONFIGURACIÓN
    # ========================================
    print("\n" + "" * 50)
    print("EXPLORADOR DE ARCHIVOS .SAV DEL INE GUATEMALA")
    print("" * 50 + "\n")
    
    # IMPORTANTE: Cambia esta ruta a donde tengas tus archivos .sav
    ruta_datos = "./datos_ine"
    
    print(f"Buscando archivos en: {ruta_datos}\n")
    
    # Ejecutar exploración
    explorar_archivos_sav_detallado(ruta_datos)
