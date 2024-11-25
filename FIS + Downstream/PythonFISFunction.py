"""
This program is a Python implementation of a Fuzzy Inference System (FIS) 
that was first designed within MATLAB, using the Fuzzy Logic Designer. 

This file serves to host the functions that are used for the FIS used in 
TME 6015.

This particular FIS utilizes the load history, distance from the task, and the 
distance history to make decisions about the suitability of a given robot
for use in multi-robot task allocation (MRTA).

"""
######################## Import Packages ########################

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

####################### Define Functions ########################

def fis_create():
    ################ FIS Step 1: Define Fuzzy Sets ##################

    """
    Must firstly define the input linguistic variables. This involves:
        - Defining the linguistic variable itself
        - Defining the crisp universe of discourse for these variables
        - Defining the linguistic values and their membership functions
    """
    # universes of discourse:
    lh_range = [0, (5/6), 4, 5, 6, (55/6), 10]
    dtt_range = [0, (25/12), 10, 12.5, 15, (275/12), 25]
    tdt_range = [0, (25/6), 15, 25, 30, (275/6), 50]
    cap_range = [0, 1]

    # define lingusitic input variables:
    lh = ctrl.Antecedent(lh_range, 'Load History')
    dtt = ctrl.Antecedent(dtt_range, 'Distance to Task')
    tdt = ctrl.Antecedent(tdt_range, 'Total Distance Travelled')
    cap = ctrl.Antecedent(cap_range, 'Capability')

    # define membership functions:
    lh['Low'] = fuzz.trimf(lh.universe, [0, 0, 6])
    lh['Medium'] = fuzz.trimf(lh.universe, [5/6, 5, 55/6])
    lh['High'] = fuzz.trimf(lh.universe, [4, 10, 10])

    dtt['Low'] = fuzz.trimf(dtt.universe, [0, 0, 15])
    dtt['Medium'] = fuzz.trimf(dtt.universe, [25/12, 12.5, 275/12])
    dtt['High'] = fuzz.trimf(dtt.universe, [10, 25, 25])

    tdt['Low'] = fuzz.trimf(tdt.universe, [0, 0, 30])
    tdt['Medium'] = fuzz.trimf(tdt.universe, [25/6, 25, 275/6])
    tdt['High'] = fuzz.trimf(tdt.universe, [15, 50, 50])

    cap['No Match'] = fuzz.trimf(cap.universe, [0, 0, 0])
    cap['Matched'] = fuzz.trimf(cap.universe, [1, 1, 1])

    """
    Now we can define the output linguistic variable. This involves:
        - Defining the linguistic variable itself
        - Defining the crisp universe of discourse for this variable
        - Defining the linguistic value and its membership function
    """

    # universe of discourse:
    suit_range = [0, 5/12, 25/12, 2.5, 35/12, 55/12, 5, 65/12, 85/12, 7.5, 95/12, 115/12, 10]

    # define linguistic output variable:
    suit = ctrl.Consequent(suit_range, 'Suitability')

    # membership functions for linguistic values:
    suit['Unacceptable'] = fuzz.trimf(suit.universe, [0, 0, 0])
    suit['Very Low'] = fuzz.trimf(suit.universe, [0, 0, 25/12])
    suit['Low'] = fuzz.trimf(suit.universe, [5/12, 2.5, 55/12])
    suit['Medium'] = fuzz.trimf(suit.universe, [35/12, 5, 85/12])
    suit['High'] = fuzz.trimf(suit.universe, [65/12, 7.5, 115/12])
    suit['Very High'] = fuzz.trimf(suit.universe, [95/12, 10, 10])

    ################ FIS Step 2: Define Rule-Base ###################
    """
    Now we can define the fuzzy rule base. This rulebase consists of 28 rules, and 
    these rules were selected based on their provided surface of control, which was
    iteratively sculpted using the rules.

    """

    # initialize rulebase
    rulebase =[]

    # commence defining the rules for the matched case:
    rulebase.append(ctrl.Rule(lh['Low'] & dtt['Low'] & tdt['Low'] & cap['Matched'], suit['Very High']))      # Rule 01
    rulebase.append(ctrl.Rule(lh['Medium'] & dtt['Low'] & tdt['Low'] & cap['Matched'], suit['High']))        # Rule 02
    rulebase.append(ctrl.Rule(lh['High'] & dtt['Low'] & tdt['Low'] & cap['Matched'], suit['Medium']))        # Rule 03
    rulebase.append(ctrl.Rule(lh['Low'] & dtt['Medium'] & tdt['Low'] & cap['Matched'], suit['High']))        # Rule 04
    rulebase.append(ctrl.Rule(lh['Medium'] & dtt['Medium'] & tdt['Low'] & cap['Matched'], suit['Medium']))   # Rule 05
    rulebase.append(ctrl.Rule(lh['High'] & dtt['Medium'] & tdt['Low'] & cap['Matched'], suit['Medium']))     # Rule 06
    rulebase.append(ctrl.Rule(lh['Low'] & dtt['High'] & tdt['Low'] & cap['Matched'], suit['Medium']))        # Rule 07
    rulebase.append(ctrl.Rule(lh['Medium'] & dtt['High'] & tdt['Low'] & cap['Matched'], suit['Medium']))     # Rule 08
    rulebase.append(ctrl.Rule(lh['High'] & dtt['High'] & tdt['Low'] & cap['Matched'], suit['Low']))          # Rule 09
    rulebase.append(ctrl.Rule(lh['Low'] & dtt['Low'] & tdt['Medium'] & cap['Matched'], suit['High']))        # Rule 10
    rulebase.append(ctrl.Rule(lh['Medium'] & dtt['Low'] & tdt['Medium'] & cap['Matched'], suit['Medium']))   # Rule 11
    rulebase.append(ctrl.Rule(lh['High'] & dtt['Low'] & tdt['Medium'] & cap['Matched'], suit['Medium']))     # Rule 12
    rulebase.append(ctrl.Rule(lh['Low'] & dtt['Medium'] & tdt['Medium'] & cap['Matched'], suit['Medium']))   # Rule 13
    rulebase.append(ctrl.Rule(lh['Medium'] & dtt['Medium'] & tdt['Medium'] & cap['Matched'], suit['Low']))   # Rule 14
    rulebase.append(ctrl.Rule(lh['High'] & dtt['Medium'] & tdt['Medium'] & cap['Matched'], suit['Low']))     # Rule 15
    rulebase.append(ctrl.Rule(lh['Low'] & dtt['High'] & tdt['Medium'] & cap['Matched'], suit['Medium']))     # Rule 16
    rulebase.append(ctrl.Rule(lh['Medium'] & dtt['High'] & tdt['Medium'] & cap['Matched'], suit['Low']))     # Rule 17
    rulebase.append(ctrl.Rule(lh['High'] & dtt['High'] & tdt['Medium'] & cap['Matched'], suit['Very Low']))  # Rule 18
    rulebase.append(ctrl.Rule(lh['Low'] & dtt['Low'] & tdt['High'] & cap['Matched'], suit['Medium']))        # Rule 19
    rulebase.append(ctrl.Rule(lh['Medium'] & dtt['Low'] & tdt['High'] & cap['Matched'], suit['Medium']))     # Rule 20
    rulebase.append(ctrl.Rule(lh['High'] & dtt['Low'] & tdt['High'] & cap['Matched'], suit['Low']))          # Rule 21  
    rulebase.append(ctrl.Rule(lh['Low'] & dtt['Medium'] & tdt['High'] & cap['Matched'], suit['Medium']))     # Rule 22
    rulebase.append(ctrl.Rule(lh['Medium'] & dtt['Medium'] & tdt['High'] & cap['Matched'], suit['Low']))     # Rule 23
    rulebase.append(ctrl.Rule(lh['High'] & dtt['Medium'] & tdt['High'] & cap['Matched'], suit['Very Low']))  # Rule 24
    rulebase.append(ctrl.Rule(lh['Low'] & dtt['High'] & tdt['High'] & cap['Matched'], suit['Low']))          # Rule 25
    rulebase.append(ctrl.Rule(lh['Medium'] & dtt['High'] & tdt['High'] & cap['Matched'], suit['Very Low']))  # Rule 26
    rulebase.append(ctrl.Rule(lh['High'] & dtt['High'] & tdt['High'] & cap['Matched'], suit['Very Low']))    # Rule 27

    # for the unmatched case:
    rulebase.append(ctrl.Rule(cap['No Match'], suit['Unacceptable']))   # Rule 28

    return rulebase

def fis_solve(rulebase, load, distance, travelled, cap):
    """
    First must create the control system, and then we can run simulations on
    this control system by further passing inputs into it, and then getting it
    to compute the output.

    """

    # create control system:
    fis_ctrl = ctrl.ControlSystem(rulebase)

    # create an instance of the control system for simulation:
    sim = ctrl.ControlSystemSimulation(fis_ctrl)

    # solve:
    sim.input['Load History'] = load
    sim.input['Distance to Task'] = distance
    sim.input['Total Distance Travelled'] = travelled
    sim.input['Capability'] = cap

    sim.compute()

    result = sim.output['Suitability']
    return result
