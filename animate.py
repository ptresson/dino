import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def get_lim(arch="dino_vitb16", method='umap', margin=0.10):
    
    total_df = None
    gdf1 = gpd.read_file(f'./out/A/{arch}0000.shp')
    gdf2 = gpd.read_file(f'./out/B/{arch}0000.shp')
    total_df = pd.concat([gdf1, gdf2], ignore_index=True)

    # for step in ['0020','0040','0060','0080','']:
    # for step in ['0000','0020','0040','0060','0080','0100','0120','0140','0160','0180','']:
    for step in ['0000','0010','0020','0030','0040','0050','0060','0070','0080','0090','']:
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
    
### cf. https://matplotlib.org/stable/users/explain/animations/animations.html
def update(frame):
    # for each frame, update the data stored on each artist.
    x = t[:frame]
    y = z[:frame]
    # update the scatter plot:
    data = np.stack([x, y]).T
    scat.set_offsets(data)
    # update the line plot:
    line2.set_xdata(t[:frame])
    line2.set_ydata(z2[:frame])
    return (scat, line2)


if __name__ == "__main__":

    for arch in ['dino_vitb16', 'dino_resnet50', 'efficientnet_b0', 'efficientnet_b3']:
    # for arch in ['efficientnet_b0', 'efficientnet_b3']:
    # for arch in ['dino_vitb16', 'dino_resnet50']:

        xlim_umap, ylim_umap = get_lim(arch)
        xlim_pca, ylim_pca = get_lim(arch, method='pca')

        # for step in ['0000','0020','0040','0060','0080','0100','0120','0140','0160','0180','']:
        for step in ['0000','0010','0020','0030','0040','0050','0060','0070','0080','0090','']:
            plot_proj(f'{arch}{step}', method='umap', xlim=xlim_umap, ylim=ylim_umap)
            plot_proj(f'{arch}{step}', method='pca', xlim=xlim_pca, ylim=ylim_pca)
