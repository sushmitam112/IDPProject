# Lalitha Chandolu & Sushmita Musunuri
# The Final Project Replit ran out of storage
# All of our code for this project is present here

import pandas as pd
import demographics
import race
import test
import admissions
import machine_learning


def main():
    '''
    Main method to call all other methods
    '''

    # reads alls datasets into dataframes
    df_ipeds = pd.read_csv("data/IPEDS_data.csv")
    df_race = pd.read_csv('data/college_racial_rep.csv')
    df_race_2017 = df_race[df_race['year'] == 2017]
    df_recent = pd.read_csv('data/Most_recent_cohorts_institution_filtered.csv')
    df_merged = df_race_2017.merge(df_recent, left_on='inst_name', right_on='INSTNM', how = 'left')

    # runs all the test methods
    test.run_testing_methods(df_merged, df_race_2017)

    # RACE PLOTS
    race.race_percent_geoplot(df_race_2017)
    race.race_enrollment_diff(df_race_2017)
    race.plot_market_share(df_race_2017)
    df_racial_index = race.calculate_racial_diversity_index(df_race_2017)
    race.race_top_bottom_5(df_racial_index)
    race.race_percent_time_bar(df_race)

    # DEMOGRAPHIC PLOTS
    demographics.race_tuition_enrollment_correlation(df_ipeds)
    demographics.filter_by_gender(df_ipeds)
    demographics.first_gen_selectivity(df_recent)
    demographics.in_state_out_state(df_ipeds)

    # ADMISSION PLOTS
    admissions.admission_rate(df_merged)

    # MACHINE LEARNING MODELS
    X1_train, X1_test, y1_train, y1_test = machine_learning.race_selectivity_data_prep(df_race_2017)
    machine_learning.run_race_selectivity_ml_models(X1_train, X1_test, y1_train, y1_test)
    X2_train, X2_test, y2_train, y2_test = machine_learning.socio_and_SAT_data_prep(df_merged)
    machine_learning.run_socio_and_SAT_data_ml_models(X2_train, X2_test, y2_train, y2_test)
     

if __name__ == '__main__':
    main()