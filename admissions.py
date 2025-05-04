import geopandas as gpd
import matplotlib.pyplot as plt


def admission_rate(df):
    '''
    Plots the average admission rate for the universities/institutions per state.
    '''
    avg_state_admission = df.groupby('ST_FIPS',
                                     as_index=False)['ADM_RATE'].mean()
    usa = gpd.read_file('data/gz_2010_us_040_00_5m.json')
    usa['STATE'] = usa['STATE'].astype(int)
    # merges with JSON state geospatial file
    merged = usa.merge(avg_state_admission, left_on='STATE', right_on='ST_FIPS',
                       how='inner')
    fig, ax = plt.subplots(1)
    ax.set_xlim([-200, -50])
    ax.set_title("Average Admissions Rate Per State", fontsize=20)
    merged.plot(ax=ax, column='ADM_RATE', legend='true')
    fig.tight_layout()
    plt.savefig('graphs/admissions_rate.png')
    # returns merged
    return merged
