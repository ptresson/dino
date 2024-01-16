import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_proj(checkpoint="dino_vitb16", method='umap', xlim=[-7, 32], ylim=[-5,25]):
    # Load your shapefiles into GeoDataFrames
    gdf1 = gpd.read_file(f'./out/A/{checkpoint}.shp')
    gdf2 = gpd.read_file(f'./out/B/{checkpoint}.shp')

    # Concatenate GeoDataFrames vertically
    merged_gdf = pd.concat([gdf1, gdf2], ignore_index=True)
    merged_gdf['src'] = merged_gdf['src'].apply(lambda x: os.path.splitext(os.path.basename(x))[0])  # Keeps only the filename without extension


    # Create a shared figure and two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    scatter_src = sns.scatterplot(
            x=f'x_{method}', 
            y=f'y_{method}', 
            hue='src', 
            data=merged_gdf, 
            palette='Spectral', 
            ax=ax1, 
            s=50,
            # marker="+",
            )
    scatter_src.legend(loc='lower left',ncol=1, title="src", bbox_to_anchor=(1, 0))
    # ax1.set_title('Points Colored by src')
    
    scatter_ID = sns.scatterplot(
            x=f'x_{method}', 
            y=f'y_{method}', 
            hue='C_ID', 
            data=merged_gdf, 
            palette='tab10', 
            ax=ax2, 
            s=50,
            # marker="+",
            )
    scatter_ID.legend(loc='lower left', title="Class", bbox_to_anchor=(1, 0))
    # ax2.set_title('Points Colored by kmeans')

    scatter_src.set_xlim(xlim)
    scatter_src.set_ylim(ylim)
    scatter_ID.set_xlim(xlim)
    scatter_ID.set_ylim(ylim)
    
    plt.tight_layout()

    # Save the plot to a file
    fig.savefig(f'./out/plots/{method}/{checkpoint}.png')
    plt.close()


def get_lim(arch="dino_vitb16", method='umap', margin=0.10):
    
    total_df = None
    gdf1 = gpd.read_file(f'./out/A/{arch}0000.shp')
    gdf2 = gpd.read_file(f'./out/B/{arch}0000.shp')
    total_df = pd.concat([gdf1, gdf2], ignore_index=True)

    for step in ['0020','0040','0060','0080','']:
        gdf1 = gpd.read_file(f'./out/A/{arch}{step}.shp')
        gdf2 = gpd.read_file(f'./out/B/{arch}{step}.shp')
        merged_df = pd.concat([gdf1, gdf2], ignore_index=True)
        total_df = pd.concat([total_df, merged_df], ignore_index=True)
        
        xmargin = abs(total_df[f'x_{method}'].max() - total_df[f'x_{method}'].min()) * margin
        ymargin = abs(total_df[f'y_{method}'].max() - total_df[f'y_{method}'].min()) * margin
        x_min = total_df[f'x_{method}'].min() - xmargin
        x_max = total_df[f'x_{method}'].max() + xmargin
        y_min = total_df[f'y_{method}'].min() - ymargin
        y_max = total_df[f'y_{method}'].max() + ymargin

    return [x_min, x_max],[y_min, y_max]
    


if __name__ == "__main__":

    # for arch in ['dino_vitb16', 'dino_resnet50', 'efficientnetb0', 'efficientnetb3']:
    # for arch in ['dino_vitb16', 'dino_resnet50', 'efficientnetb0', 'efficientnetb3']:
    for arch in ['dino_vitb16', 'dino_resnet50']:

        xlim_umap, ylim_umap = get_lim(arch)
        xlim_pca, ylim_pca = get_lim(arch, method='pca')

        for step in ['0000','0020','0040','0060','0080','']:
            plot_proj(f'{arch}{step}', method='umap', xlim=xlim_umap, ylim=ylim_umap)
            plot_proj(f'{arch}{step}', method='pca', xlim=xlim_pca, ylim=ylim_pca)
