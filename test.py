import demographics
import machine_learning
import race
import admissions
import pandas as pd
from cse163_utils import assert_equals

'''
File to hold all the methods related to testing the code.
Many tests use True in the paramaters, so that the method
can differentiate test methods and not plot the fake data.
'''

def run_testing_methods(merged_df, race_df):
    '''
    Instantiates the test csv files
    Calls the tests methods for the different research questions/code groups
    '''
    test_race_df = pd.read_csv('test_data/racial_test.csv')
    # also used for tests which assumed that it was only one year
    test_race_df2 = pd.read_csv('test_data/racial_test_diff_years.csv')
    test_ipeds_df = pd.read_csv('test_data/ipeds_test.csv')
    test_recent_df = pd.read_csv('test_data/recent_cohorts_test1.csv')
    test_df_gender = pd.read_csv('test_data/gender_test.csv')
    # calls on all of the testing methods
    call_race_test(test_race_df,test_race_df2, test_ipeds_df)
    test_race_tuition_correlation(test_ipeds_df)
    test_filter_by_gender(test_ipeds_df)
    test_in_out_state(test_ipeds_df)
    test_plot_gender_barplot(test_df_gender)
    test_admissions_plot(test_recent_df)
    test_ml_data_prep(merged_df, race_df)


def test_filter_by_gender(df_ipeds_test):
    '''
    Tests the accuracy of the geospatial plot made by the filter_by_gender method
    in the demographics.py file
    '''
    df_gender = demographics.filter_by_gender(df_ipeds_test, True)
    assert_equals(list(df_gender['NAME']), ['Alabama', 'California', 'Washington'])
    assert_equals(list(df_gender['women']), [52.5,90,32.5])
    assert_equals(list(df_gender['men']), [47.5,10,67.5])
    assert_equals(list(df_gender['GEO_ID']), ['0400000US01', '0400000US06', '0400000US53' ])


def test_plot_gender_barplot(df_gender):
    '''
    Tests the stacked gender bar plot's accuracy in finding
    difference between men and women.
    '''
    df_gender_state = demographics.plot_gender_barplot(df_gender)
    assert_equals(list(df_gender_state['difference']),[0.2, 0.4, 1.0, 0.8, -1.0, -0.8, -1.0, -0.4, -0.4])


def test_race_tuition_correlation(df):
    '''
    Tests data organization for the race tuition correlation graphs
    '''
    white_test = demographics.plot_race_tuition(df, 'White', 'purple', True)
    assert_equals(list(white_test), [46.0, 60.0, 76.0, 130.0, 180.0])
    black_test = demographics.plot_race_tuition(df, 'Black', 'red', True)
    assert_equals(list(black_test), [9.0, 7.5, 4.0, 22.5, 30.0])


def test_in_out_state(df):
    '''
    Tests the organizations for the residency vs admissions rate correlation regplots.
    '''
    df_admissions_residency = demographics.in_state_out_state(df, True)
    assert_equals(list(df_admissions_residency['admission_rate']), [20.0, 75.0, 10.0, 7.143, 15.0])
    assert_equals(list(df_admissions_residency['Number of first-time undergraduates - in-state']), [400, 500, 200, 400, 700])


def test_admissions_plot(df):
    '''
    Tests the admission plot's average groupby part.
    '''
    admission_df = admissions.admission_rate(df)
    assert_equals(list(admission_df['STATE']), [1, 6])
    assert_equals(list(admission_df['ADM_RATE']), [0.85, 0.605])


def call_race_test(test_race_df,test_race_df2, test_ipeds_df):
    '''
    Calls all the methods for testing the different race research
    question methods.
    '''
    test_race_percent_geoplots(test_race_df, test_race_df2)
    test_race_market_diff_geoplots(test_race_df)
    test_racial_index(test_race_df,test_race_df2)
    test_year_averages(test_race_df,test_race_df2)


def test_race_percent_geoplots(race_df, race_df2):
    '''
    Tests the average calculation for the race percent
    geospatial plots and the merging with geospatial data.
    '''
    avg_race_df = race.race_percent_geoplot(race_df, True)
    avg_race_df2 = race.race_percent_geoplot(race_df2, True)
    assert_equals(list(avg_race_df['white']), [140/3, 130/3])
    assert_equals(list(avg_race_df['black']), [70/3, 20])
    assert_equals(list(avg_race_df['hispanic']), [20, 80/3])
    # only one column with number/ all columns 0
    assert_equals(list(avg_race_df['twora']), [1/3,0])
    assert_equals(list(avg_race_df['minority']), [161/3,170/3])
    # one university in each state
    assert_equals(list(avg_race_df2['white']),[50.0, 40.0, 30.0, 50.0, 20.0, 50.0])
    assert_equals(list(avg_race_df['GEO_ID']), ['0400000US06', '0400000US53'])


def test_race_market_diff_geoplots(race_df):
    '''
    Tests the plots for the market different per state.
    '''
    avg_diff_df = race.race_enrollment_diff(race_df, True)
    # positive only
    assert_equals(list(avg_diff_df['white']), [1, 2])
    # negative only
    assert_equals(list(avg_diff_df['hispanic']), [-14/3, -8/3])
    # negative and positive
    assert_equals(list(avg_diff_df['black']), [1/3, -5/3])
    # makes sure the states are merged correctly
    assert_equals(list(avg_diff_df['GEO_ID']), ['0400000US06', '0400000US53'])


def test_racial_index(race_df1, race_df2):
    '''
    Uses fake data to calculate racial indexes.
    '''
    race_div_df1 = race.calculate_racial_diversity_index(race_df1)
    race_div_df2 = race.calculate_racial_diversity_index(race_df2)
    assert_equals(list(race_div_df1['index']), [0.668461, 0.668461, 0.727708, 0.576832, 0.668461, 0.727708])
    assert_equals(list(race_div_df2['index']), [0.6437752, 0.576832, 0.6684612, 0.6684612, 0.727708, 0.727708])


def test_ml_data_prep(merged_df, race_df):
    '''
    Makes sure that the filtering and splitting
    for prepping the dataframes for machine learning
    is performed  correctly.
    '''
    X1_train, X1_test, y1_train, y1_test = machine_learning.race_selectivity_data_prep(race_df)
    X2_train, X2_test, y2_train, y2_test = machine_learning.socio_and_SAT_data_prep(merged_df)
    print(y1_train)
    print(y2_train)
    # checks length
    assert_equals(list(X1_test.columns), list(X1_test))
    assert_equals(list(X2_test.columns), list(X2_test))
    
   
def test_year_averages(race_df1, race_df2):
    '''
    Tests the averaging per year organization
    of the race enrollment over time plots.
    '''
    avg_race_df = race.race_percent_time_bar(race_df1, True)
    avg_race_df2 = race.race_percent_time_bar(race_df2, True)

    # all years in df the same
    assert_equals(list(avg_race_df['white']), [45])
    assert_equals(list(avg_race_df['black']), [65/3])
    # different years
    assert_equals(list(avg_race_df2['white']), [40])
    assert_equals(list(avg_race_df2['black']), [65/3])
    # different_years
    assert_equals(list(avg_race_df2['white']), [40])
               
