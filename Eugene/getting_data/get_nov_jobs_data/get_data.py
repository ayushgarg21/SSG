
import sys
import logging
sys.path.insert(0, "/Users/eugene/Documents/py_utils")
import re


# Import your custom module
from jt_golden_utility import execute_read_query, create_db_engine, execute_insert_update, setup_logger

logger = setup_logger(log_dir='logs', log_file='log', retention_days=30)
logger.setLevel(logging.INFO)

RRdb=create_db_engine("RR",logger,"jobtech_data_singapore2")

# jobtech_job_test_ssg_2025
# UATdb=create_db_engine("REBOOTPROACCCONFIG",logger,"account")

def get_data(RRdb):
    query="""SELECT
    jj.job_id,
    jj.job_url,
    jj.job_title,
    jj.job_description,
    jj.date_posted,
    jj.date_expiring,
    c.company_id,
    c.company_id_normalised,
    c.company_name,
    c.industry_id,
    jj.wage_min,
    jj.wage_max,
    jj.source_id
    FROM
    jobtech_data_singapore2 .jobtech_job jj
    INNER JOIN (
    SELECT
    cnim.company_id AS company_id_normalised,
    cnim.company_id_deprecated AS company_id,
    cnim.company_name_deprecated AS company_name,
    industry_id
    FROM jobtech_data_singapore2.company_norm_industry_map cnim
    JOIN jobtech_industry.industry_specialisation ji
    ON cnim.industry_spec_id_norm = ji.industry_specialisation_id
    ) c
    ON jj.company_id = c.company_id
    WHERE
    jj.date_posted BETWEEN '2025-10-01' AND '2025-10-31';"""
    data=execute_read_query(RRdb,query,logger)
    print(data)
    print(f"Total rows fetched: {len(data)}")
    return data

data=get_data(RRdb)

data.to_parquet("nov_25_jobs_final.parquet",index=False)