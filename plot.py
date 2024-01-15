import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_umap(checkpoint="dino_vitb16", xlim=[-7, 32], ylim=[-5,25]):
    # Load your shapefiles into GeoDataFrames
    gdf1 = gpd.read_file(f'./out/A/{checkpoint}.shp')
    gdf2 = gpd.read_file(f'./out/B/{checkpoint}.shp')

    # Concatenate GeoDataFrames vertically
    merged_gdf = pd.concat([gdf1, gdf2], ignore_index=True)
    merged_gdf['src'] = merged_gdf['src'].apply(lambda x: os.path.splitext(os.path.basename(x))[0])  # Keeps only the filename without extension


    # Create a shared figure and two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    scatter_src = sns.scatterplot(
            x='x_umap', 
            y='y_umap', 
            hue='src', 
            data=merged_gdf, 
            palette='Spectral', 
            ax=ax1, 
            s=10
            )
    scatter_src.legend(loc='lower left',ncol=1, title="src", bbox_to_anchor=(1, 0))
    # ax1.set_title('Points Colored by src')
    
    scatter_ID = sns.scatterplot(
            x='x_umap', 
            y='y_umap', 
            hue='C_ID', 
            data=merged_gdf, 
            palette='tab10', 
            ax=ax2, 
            s=10
            )
    scatter_ID.legend(loc='lower left', title="Class", bbox_to_anchor=(1, 0))
    # ax2.set_title('Points Colored by kmeans')

    if xlim and ylim:
        scatter_src.xlim(xlim)
        scatter_src.ylim(ylim)
        scatter_ID.xlim(xlim)
        scatter_ID.ylim(ylim)
    
    plt.tight_layout()

    # Save the plot to a file
    fig.savefig(f'./out/plots/{checkpoint}.png')


if __name__ == "__main__":
    for arch in ['dino_vitb16']:
        for step in ['0000','0020','0040','0060','0080','']:
            plot_umap(f'{arch}{step}')
