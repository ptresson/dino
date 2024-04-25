import os
import geopandas as gpd
import pandas as pd
from sklearn.metrics import rand_score
# from sklearn.metrics import adjusted_rand_score as rand_score


def get_rand_score(checkpoint, target='C_ID'):

    gdf1 = gpd.read_file(f'./out/A/{checkpoint}.shp')
    gdf2 = gpd.read_file(f'./out/B/{checkpoint}.shp')

    # Concatenate GeoDataFrames vertically
    merged_gdf = pd.concat([gdf1, gdf2], ignore_index=True)
    merged_gdf['src'] = merged_gdf['src'].apply(lambda x: os.path.splitext(os.path.basename(x))[0])  # Keeps only the filename without extension

    return rand_score(merged_gdf[target], merged_gdf['kmeans'])

if __name__ == "__main__":
    
    for arch in ['vit_base_patch16_224','resnet50', 'efficientnet_b0']:
        print(arch)

        scores_id = []
        scores_src = []
        for step in ['0000','0010','0020','0030','0040','0050','0060','0070','0080','0090','']:
            score_id = get_rand_score(f'{arch}{step}')
            score_src = get_rand_score(f'{arch}{step}', 'src')
            scores_id.append(score_id)
            scores_src.append(score_src)
            print(f'{score_id:.2}\t{score_src:.2}')

        # print(f'{arch}\t{min(scores_id), max(scores_id)}\t{min(scores_src), max(scores_src)}')

