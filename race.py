import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
'''
Code for the first question about race across states and race changes over time.

3 important columns in the college_racial_rep csv dataset
    col_RACE: Percentage of total number of enrolled students who are that Race
    mkt_RACE: Percentage of total population (college aged students) who are that Race
    dif_RACE: Difference in percent of enrolled and marker (col_RACE - mkt_RACE)
'''


def race_percent_geoplot(df, test=False):
    '''
    This method finds the average percentage of the representation of every race 
    in every college within a state. The geospatial plot represents this data 
    though a spectrum. This takes in the college racial representatives dataset for
    the year 2017.
    '''
    average_races = pd.DataFrame()
    df_copy = df.copy()
    # amind = American Indians; twora = Two or More Races; hispa = Hispanic
    races = ['white', 'black', 'asian', 'hispanic', 'amind', 'pacis', 'twora']
    df_copy.columns = ['col_hispanic' if x == 'col_hispa' else x for x in df.columns]
    for race in races:
        average_races[race] = df_copy.groupby('fips_ipeds')['col_' + race].mean()
    # finds average minority percentage by adding minority races together
    average_races['minority'] = (average_races['asian'] +
                                 average_races['black'] +
                                 average_races['hispanic'] +
                                 average_races['pacis'] +
                                 average_races['amind'] +
                                 average_races['twora'])
    # merges with the JSON file which contains shape of states
    usa = gpd.read_file('data/gz_2010_us_040_00_5m.json')
    merged = usa.merge(average_races,
                       left_on='NAME',
                       right_index=True,
                       how='right')

    # only plots if not test function
    if test == False:
        plot_race_percent(merged, 'minority')
        plot_race_percent(merged, 'white')
        plot_race_percent(merged, 'black')
        plot_race_percent(merged, 'asian')
        plot_race_percent(merged, 'hispanic')
    return merged


def plot_race_percent(merged, race):
    '''
    Helper method for the race_percent_geoplot method.
    Plots the graph for each race, by taking in the merged dataset
    with groupby and specific race.
    '''
    fig, ax = plt.subplots(1)
    fig.tight_layout()
    ax.set_title('Avg Enrollment % of ' + race.capitalize() +
                 ' Populations Per State')
    merged.plot(ax=ax, column=race, legend='true')
    plt.xlim(-200, -35)
    plt.savefig('graphs/state_race_' + race + '.png')
    plt.clf()


def plot_market_share(df):
    """
    Plots the market share of the specific racial populations
    within each state using the college_racial_rep dataset 
    Filtered to the data in the year 2017
    """
    races = ['white', 'black', 'asian', 'hispanic', 'amind', 'pacis', 'twora']
    # makes a copy of the college_racial_rep dataset
    df = df.copy()
    mkt_share = pd.DataFrame()
    df.columns = [
        'mkt_hispanic' if x == 'mkt_hispa' else x for x in df.columns]
    for race in races:
        mkt_share[race] = df.groupby('fips_ipeds')['mkt_' + race].mean()
    # groups toegther minority populations for minority group metrics
    mkt_share['minority'] = (mkt_share['asian'] + mkt_share['black'] +
                             mkt_share['hispanic'] + mkt_share['pacis'] +
                             mkt_share['amind'] + mkt_share['twora'])
    usa = gpd.read_file('data/gz_2010_us_040_00_5m.json')
    merged = usa.merge(mkt_share,
                       left_on='NAME',
                       right_index=True,
                       how='right')

    # for loop plots the market share graphs for each of the racial categories
    for race in ['minority', 'white', 'black', 'hispanic', 'asian']:
        fig, ax = plt.subplots(1)
        fig.tight_layout()
        ax.set_title('Average Market Share of ' + race.capitalize() +' People Per State')
        merged.plot(ax=ax, column=race, legend='true')
        plt.xlim(-200, -35)
        plt.savefig('graphs/market_race_' + race + '.png')
        plt.clf()


def race_enrollment_diff(df, test=False):
    '''
    The market difference represents the difference in percentage of the university's 
    enrolled population of a race and the state's overall market 
    of the race (col_RACE - mkt_RACE).
    Takes in the college racial representative dataset for the year 2017. 
    '''
    races_dif = pd.DataFrame()
    df_copy = df.copy()
    races = ['white', 'black', 'asian', 'hispanic', 'amind', 'pacis', 'twora']
    df_copy.columns = [
        'dif_hispanic' if x == 'dif_hispa' else x for x in df.columns]
    for race in races:
        races_dif[race] = df_copy.groupby('fips_ipeds')['dif_' + race].mean()

    # merges with the JSON file which contains shape of states
    usa = gpd.read_file('data/gz_2010_us_040_00_5m.json')
    merged = usa.merge(races_dif, left_on='NAME', right_index=True, how='inner')

    # only plots if not testing call
    if test == False:
        plot_market_difference(merged, 'white')
        plot_market_difference(merged, 'black')
        plot_market_difference(merged, 'asian')
        plot_market_difference(merged, 'hispanic')

    # returns the dataset with groupby for testing
    return merged


def plot_market_difference(merged, race):
    '''
    Helper method for race_enrollment_diff method. Plots the market difference of 
    the race passed in and the merged/groupby dataset. 
    '''
    fig, ax = plt.subplots(1)
    fig.tight_layout()
    ax.set_title('Avg Market Difference of ' + race.capitalize() +' People Per State')
    merged.plot(ax=ax, column=race, legend='true')
    plt.xlim(-200, -35)
    plt.savefig('graphs/market_diff_' + race + '.png')
    plt.clf()


def calculate_racial_diversity_index(df):
    '''
    Calculate the racial diversity scores by using the Shannon-Wiener Diversity Index.
    Excluded amind, pacis, and twora races as they caused the diversity index to be 
    negative. Uses a subset of the data (year == 2017) out of the college racial rep dataset.
    https://archives.huduser.gov/healthycommunities/sites/default/files/public/Racial%20Diversity%20using%20Shannon-Wiener%20Index.pdf
    '''
    df_copy = df.copy()
    races = ['white', 'black', 'asian', 'hispa', 'amind', 'pacis', 'twora']

    for race in races:
        # divided by 100 to get in decimal format
        df_copy['col_' + race] = df_copy['col_' + race].div(100)
        # finds population number fo race
        df_copy[race +'_pop'] = df_copy['total_enrollment'] * df_copy['col_' + race]
        # Shannon-Wiener Diversity Index method from pdf
        df_copy[race +'_diverse'] = df_copy['col_' +
                                      race].apply(log_apply) * df_copy['col_' + race]

    # performs the -sum of all the races
    df_copy['index'] = df_copy.loc[:, ['white_diverse', 'hispa_diverse']].sum(axis=1) * -1
    return df_copy


def log_apply(series):
    '''
    Helper method for calculating racial diversity index
    '''
    # returns log if it doesn't equal 0
    if series == 0:
        return 0
    else:
        return np.log(series)


def race_top_bottom_5(df):
    '''
    Uses the dataset returned by the calculate_racial_diversity_index method to
    find the top 5 and bottom 5 universities in racial diversity and plots a bar
    graph of their racial percentages.
    '''
    fig, ax = plt.subplots(1)
    top_5 = df.nlargest(5, 'index')
    lowest_5 = df.nsmallest(5, 'index')
    # concatenates the two dataframes
    top_and_worst = pd.concat([top_5, lowest_5])

    # plots bar plot and customizes it
    top_and_worst[[
        'inst_name', 'col_white', 'col_black', 'col_asian', 'col_hispa',
        'col_amind', 'col_pacis', 'col_twora'
    ]].plot.bar(x='inst_name', ax=ax, fontsize=8)
    plt.xlabel('Institution Name', fontsize=15)
    plt.ylabel('Racial Percentages', fontsize=15)
    plt.title('Top/Bottom 5 Universities in Racial Diversity', fontsize=20)

    # used ChatGPT for legend change code
    handles, labels = ax.get_legend_handles_labels()
    new_labels = [
        'White', 'Black', 'Asian', 'Hispanic', 'American Indian',
        'Pacific Islander', '2+ Races'
    ]
    ax.legend(handles, new_labels, fontsize=12)
    plt.savefig('graphs/top_and_worst.png', bbox_inches='tight')
    plt.clf()


def race_percent_time_bar(df, test=False):
    '''
    Takes in the college racial rep dataset and plots a stacked bar
    plot of the change in racial percentages and calls the method for
    the line plot for enrollment over time.
    '''
    average_races = pd.DataFrame()
    average_races['total_enrollment'] = df.groupby(
        'year')['total_enrollment'].sum()
    races = ['white', 'asian', 'black', 'hispa', 'pacis', 'amind', 'twora']

    # performs average of each race by year with groupby
    for race in races:
        average_races[race] = df.groupby('year')['col_' + race].mean()

    # only plots if it is not a test call
    if test == False:
        fig, ax = plt.subplots(1)
        # filters to not include year with NaN value
        average_races['year'] = [year for year in range(2009, 2018)]
        average_races = average_races[average_races['year'] >= 2010]
        average_races1 = average_races.drop(columns='total_enrollment')

        # plots bar plot and customizes graph
        average_races1.plot.bar(x='year', stacked=True, ax=ax)
        handles, labels = ax.get_legend_handles_labels()
        new_labels = [
            'White', 'Black', 'Asian', 'Hispanic', 'American Indian',
            'Pacific Islander', '2+ Races'
        ]
        ax.legend(handles, new_labels)
        plt.title('Enrollment Percentages By Race Across the Nation')
        plt.xlabel('Year')
        plt.ylabel('Percentage')
        plt.savefig('graphs/race_percent_over_time.png', bbox_inches='tight')
        race_enrollment_time_line(average_races)
    return average_races


def race_enrollment_time_line(average_races, test=False):
    '''
    Takes in the dataset which is already grouped by year and plots
    a line plot of the enrollment numbers of the different races over
    the years.
    '''
    races = ['white', 'asian', 'black', 'hispa', 'pacis', 'amind', 'twora']

    # finds the total enrollment from the average percent
    for race in races:
        average_races[race] = average_races[race] / 100 * average_races['total_enrollment']
    fig, ax = plt.subplots(1)
    average_races = average_races.drop(columns='total_enrollment')
    average_races.plot(x='year', figsize=(12, 10), ax=ax, fontsize=12)
    handles, labels = ax.get_legend_handles_labels()
    
    # new labels adds names for the legend so they are not the columns names from the dataset
    new_labels = [
        'White', 'Black', 'Asian', 'Hispanic', 'American Indian',
        'Pacific Islander', '2+ Races'
    ]
    ax.legend(handles, new_labels, fontsize=15)
    plt.title('Enrollment By Race Across the Nation', fontsize=20)
    plt.xlabel('Year', fontsize=15)
    plt.ylabel('Enrollment Count', fontsize=15)
    plt.ticklabel_format(style='plain')
    plt.savefig('graphs/race_count_over_time.png', bbox_inches='tight')
