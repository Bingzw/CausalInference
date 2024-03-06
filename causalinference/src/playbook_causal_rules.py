import os
import pandas as pd
import pykrylov.ems as experiment
from typing import Dict
from core.root_cause_detector import RootCauseDetector
from utils.exp_utils import get_config_from_ems, create_logger


def playbook_causal_rules(run_params: Dict = None) -> Dict:
    """
    check root cause for low impression and low roas issue
    ------------------------------------------------------
    :param run_params: experiment parameters
    :return: run_params
    """
    logger = create_logger()
    logger.info("==== playbook_causal_rules starts ====")
    # load the processed training data and config setting
    run_params = experiment.get_parameters()
    data_config = get_config_from_ems(run_params, "data_config")
    processed_training_data = data_config["processed_training_data"]
    output_path = run_params["output_path"]
    k_cond_search = data_config["k_cond_search_para"]
    k_treat_search = data_config["k_treat_search_para"]
    defect_low_imp = data_config["defect_low_imp"]
    defect_low_roas = data_config["defect_low_roas"]

    feature_config = get_config_from_ems(run_params, "feature_config")
    outcome_column = feature_config["outcome_column"]
    treatment_columns = feature_config["treatment_columns"]
    inverse_treatment_columns = feature_config["inverse_treatment_columns"]
    low_imp_treatment_columns = treatment_columns["low_imp_treatment_columns"]
    low_roas_treatment_columns = treatment_columns["low_roas_treatment_columns"]
    low_imp_inverse_treatment_columns = inverse_treatment_columns["low_imp_inverse_treatment_columns"]
    low_roas_inverse_treatment_columns = inverse_treatment_columns["low_roas_inverse_treatment_columns"]
    covariate_columns = feature_config["covariate_columns"]

    # load the processed training data
    processed_training_data_path = os.path.join(output_path, processed_training_data)
    df_cleaned = pd.read_csv(processed_training_data_path, index_col=False)

    # check root cause for low impression issue
    logger.info("==== check root cause for low impression issue ====")
    df_imp = df_cleaned.loc[df_cleaned["defect_type"] == defect_low_imp].copy()

    # check the hypothesis for low impression issue
    for cur_treatment_column in low_imp_treatment_columns:
        logger.info("Testing the hypothesis that low impression is caused by low %s" % cur_treatment_column)
        low_imp_treatment_column = [cur_treatment_column]
        df_low_imp = df_imp[outcome_column + low_imp_treatment_column + covariate_columns].copy()
        # remove the null rows since current causal model does not support censored data
        df_low_imp = df_low_imp.loc[df_low_imp[low_imp_treatment_column[0]].notnull()]
        if cur_treatment_column in low_imp_inverse_treatment_columns:
            df_low_imp[cur_treatment_column] = df_low_imp[cur_treatment_column] * (-1)
        rcd_low_imp = RootCauseDetector(data=df_low_imp, outcome=outcome_column[0],
                                        treatment_variable=low_imp_treatment_column[0])
        rcd_optimal_low_imp_dict, df_rcd_low_imp = rcd_low_imp.grid_search_optimization(k_treat=k_treat_search)
        rcd_optimal_low_imp_dict['diagnostic_type'] = defect_low_imp
        logger.info("Optimal root cause detector parameters: %s" % rcd_optimal_low_imp_dict)
        save_file_name = "df_rcd_imp_kw_%s.csv" % cur_treatment_column
        rcd_low_imp_path = os.path.join(output_path, save_file_name)
        df_rcd_low_imp.to_csv(rcd_low_imp_path, index=False)
        experiment.record_asset(save_file_name, rcd_low_imp_path)
        logger.info("saving the root cause detector result to: %s" % rcd_low_imp_path)

    # check root cause for low roas issue
    logger.info("==== check root cause for low roas issue ====")
    df_roas = df_cleaned.loc[df_cleaned["defect_type"] == defect_low_roas].copy()
    for cur_treatment_column in low_roas_treatment_columns:
        logger.info("Testing the hypothesis that low roas is caused by low %s" % cur_treatment_column)
        low_roas_treatment_column = [cur_treatment_column]
        low_roas_condition_column = ["ctr", "cvr"]
        df_low_roas = df_roas[outcome_column + low_roas_condition_column + low_roas_treatment_column +
                              covariate_columns].copy()
        df_low_roas = df_low_roas.loc[df_low_roas[low_roas_treatment_column[0]].notnull()]
        if cur_treatment_column in low_roas_inverse_treatment_columns:
            df_low_roas[cur_treatment_column] = df_low_roas[cur_treatment_column] * (-1)
        df_low_roas["ctr"] = df_low_roas["ctr"] * (-1)
        rcd_low_roas = RootCauseDetector(data=df_low_roas, outcome=outcome_column[0],
                                         treatment_variable=low_roas_treatment_column[0],
                                         condition_expression='(ctr and cvr)')
        rcd_optimal_low_roas_dict, df_rcd_low_roas = rcd_low_roas.grid_search_optimization(k_cond=k_cond_search,
                                                                                           k_treat=k_treat_search)
        rcd_optimal_low_roas_dict['diagnostic_type'] = defect_low_roas
        logger.info("Optimal root cause detector parameters: %s" % rcd_optimal_low_roas_dict)
        save_file_name = "df_rcd_roas_kw_%s.csv" % cur_treatment_column
        rcd_low_roas_path = os.path.join(output_path, save_file_name)
        df_rcd_low_roas.to_csv(rcd_low_roas_path, index=False)
        experiment.record_asset(save_file_name, rcd_low_roas_path)
        logger.info("saving the root cause detector result to: %s" % rcd_low_roas_path)

    logger.info("==== done with root cause detection ====")

    return run_params