# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())

def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """

    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    file = pd.read_csv(input_filepath,sep=';')


    file['Date_Time'] = file['Date'] + ' ' + file['Time']
    file.drop(columns=['Date', 'Time'], inplace=True)  # %%
    name = output_filepath + '_procesado.csv'
    file["Date_Time"] = pd.to_datetime(file["Date_Time"])
    file.set_index("Date_Time", inplace=True)
    file.replace(',', '.', regex=True, inplace=True)
    file.to_csv(name, sep=';')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
