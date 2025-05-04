import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import geopandas as gpd

"""
Includes code for the 2nd research question
Using various demographic characteristics to explore correlations between
enrollment/admissions and other factors
"""


def race_tuition_enrollment_correlation(df):
    '''
    Calls the method to plot the correlation between
    tutition for the 4 highest racial populations. 
    Takes in the IPEDS Dataset.
    '''
    plot_race_tuition(df, 'Black', 'red')
    plot_race_tuition(df, 'Asian', 'blue')
    plot_race_tuition(df, 'Hispanic/Latino', 'green')
    plot_race_tuition(df, 'White', 'purple')


def plot_race_tuition(df, race, color, testing=False):
    '''
    Plots a scatterplot which signifies the correlation between the number
    of students of a particular race. Takes in the color of the graph, and
    the race being graphed.
    '''
    fig, ax = plt.subplots(1)
    if race == 'Black':
        percent_col = 'Black or African American'
    else:
        percent_col = race
    name = race + '_enrollment_count'
    df_copy = df.copy()
    # calculates enrollment from total enrollment and percentage columns
    total_enroll = df_copy['Total  enrollment']
    df_copy[name] = df_copy['Percent of total enrollment that are ' +
                            percent_col] / 100 * total_enroll

    if testing == False:
        # color is changed for each race; alpha modifies the transparency of the dots
        plt.scatter(x='Tuition and fees, 2013-14',
                    y=name, data=df_copy,
                    s=df_copy['Percent of total enrollment that are ' +
                              percent_col],
                    color=color, alpha=0.4)
        plt.xlabel('Tuition ($)'),
        plt.ylabel('Number of ' + race + ' People In Insititution')
        plt.title('Correlation Between ' + race + ' Population and Tuition')
    # exception for Hispanic/Latino column
    if race == 'Hispanic/Latino':
        race = race.split('/')[0]
    plt.savefig('graphs/' + race + '_enrollment_tuition.png')
    plt.close()
    return df_copy[name]


def filter_by_gender(df, testing=False):
    '''
    Takes in the IPEDS Dataframe and filters the data
    for men and women statistics and calls the two gender
    demographics plots.
    '''
    df_gender = pd.DataFrame()
    # average enrollment of Women per each state
    df_gender['women'] = df.groupby('FIPS state code')[
        'Percent of total enrollment that are women'].mean()
    # no column for men: manually calculated as the remainder for whatever is not men
    df_gender['men'] = 100 - df_gender['women']
    # merges with JSON file for state geometry
    usa = gpd.read_file('data/gz_2010_us_040_00_5m.json')
    merged = usa.merge(df_gender,
                       left_on='NAME',
                       right_index=True,
                       how='inner')
    if testing == False:
        plot_gender_geospatial(merged, 'women', 'men')
        plot_gender_barplot(df_gender)
    return merged


def plot_gender_geospatial(merged, gender1, gender2):
    '''
    Helper method for filter_by_gender and takes in the merged dataset
    Plots the geospatial plots on one figure
    Top plot represents the enrollment of women
    Bottom plot represents the enrollment of men
    '''
    fig, axs = plt.subplots(2, figsize=(30, 25))
    # set up the axes for the figure
    for ax in axs:
        ax.set_xlim([-200, -50])
        ax.xaxis.set_tick_params(labelsize=20)
        ax.yaxis.set_tick_params(labelsize=20)
        ax.set_ylabel('Enrollment %', fontsize=30)
    # plots the geospatial graphs
    axs[0].set_title('Avg Enrollment % of ' + gender1.capitalize() +' Per State',
                     fontsize=40)
    merged.plot(ax=axs[0], column=gender1, legend='true')
    axs[1].set_title('Avg Enrollment % of ' + gender2.capitalize() +' Per State',
                     fontsize=40)
    merged.plot(ax=axs[1], column=gender2, legend='true')
    # sets a supertitle for the figure
    fig.suptitle('Average Enrollment By Gender Across the Nation', fontsize=60)
    # padding spaces out text from graphs, enhancing visual appearence
    fig.tight_layout(pad=3)
    plt.savefig('graphs/geospatial_gender.png')
    plt.close()


def plot_gender_barplot(df_gender):
    """
    Helper method for the filter_by_gender method'''
    Uses the IPEDS dataset
    Plots a stacked bar plot of the
    """
    # measures the difference in enrollment percentage between Men and Women
    df_gender['difference'] = df_gender['men'] - df_gender['women']
    # find the 5 universities with the smallest percentage difference
    smallest = df_gender.nsmallest(5, 'difference')
    # find the 5 universities with the largest percentage difference
    largest = df_gender.nlargest(5, 'difference')
    df_gender_state = pd.concat([smallest, largest])
    fig, ax = plt.subplots(1)
    df_gender_state = df_gender_state.drop(columns='difference')
    # plots the stacked bar plot, consecutive order top 5 -- bottom 5
    df_gender_state.plot.bar(use_index=True, stacked=True, ax=ax, fontsize=10)

    plt.title('Gender Enrollment Percentages By State: Top/Bottom 5')
    plt.xlabel('State')
    plt.ylabel('Percentage')
    plt.savefig('graphs/gender_barplot.png', bbox_inches='tight')
    plt.close()
    return df_gender


def first_gen_selectivity(df):
    """
    Takes the Recent Cohort Scorecard dataset
    The method plots the correlations between an institution's admission rate
    and the number of their first-generation enrolled students
    Each dot represents an individual institution
    """
    fig, ax = plt.subplots(1)
    df_copy = df.copy()
    df_copy['ADM_RATE'] = df_copy['ADM_RATE'] * 100
    df_copy['FIRST_GEN'] = df_copy['FIRST_GEN'] * 100
    ax = sns.regplot(x='ADM_RATE', y='FIRST_GEN', data=df_copy)
    ax.set(title='Correlation Between First Gen Percent and Admissions',
           xlabel='Admissions Rate %',
           ylabel='% of First Gen Students')
    plt.savefig('graphs/first_gen_admission.png')
    plt.close()


def in_state_out_state(df, testing=False):
    """
    Takes the IPEDS dataset
    Plots three graphs on one figure representing number of students admitted from
    In-State, Out-Of-State, and Foreign Countries
    Finds the correlation between enrollment metrics and acceptance/admissions rate
    """
    # calculates the admission rate
    df['admission_rate'] = df['Admissions total'] / df['Applicants total'] * 100
    if testing == False:
        # sets up the figure, titles, and labels
        fig, [ax1, ax2, ax3] = plt.subplots(3, figsize=(20, 13))
        axs = [ax1, ax2, ax3]
        ax1.set_title('Number of In-State versus Admission Rate', fontsize=15)
        ax2.set_title('Number of Out-of-State versus Admission Rate',
                      fontsize=15)
        ax3.set_title('Number of International Students versus Admission Rate',
                      fontsize=15)
        fig.suptitle(
            'Correlation Between In State, Out of State, and International Student Enrollment versus Admission Rate', fontsize=25)
        # plots three regplots: in-state, out-of-state, foreign
        sns.regplot(x='admission_rate',
                    y='Number of first-time undergraduates - in-state',
                    data=df, ax=ax1)
        sns.regplot(x='admission_rate',
                    y='Number of first-time undergraduates - out-of-state',
                    data=df, ax=ax2, color='r')
        sns.regplot(
            x='admission_rate',
            y='Number of first-time undergraduates - foreign countries',
            data=df, ax=ax3, color='g')
        # sets the x and y labels for each axis
        for ax in axs:
            ax.set_xlabel('Admissions Rate (%)', fontsize=15)
            ax.set_ylabel('Number of Students', fontsize=15)
        fig.tight_layout(pad=3.0)
        plt.savefig('graphs/residency_and_admissions.png')
        plt.close()
    return df
