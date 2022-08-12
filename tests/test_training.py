# from streamlit_app import do_something
from pycaret.classification import *
import pandas as pd
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

from src.data.load_dataset import load_data
from
@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))

def entrenar_modelos(df, etiqueta, metrica, ensamble=True):
    '''
    ARGS: dataframe (pd.DataFrame),
    etiqueta con nombre de dataframe.column (str),
    metrica puede ser ['f1', 'accuracy', 'recall'] (str) y
    ensamble[default=True, False] (boolean)
    '''

    # setup
    pycaret_s = setup(df, target=etiqueta, session_id=123, silent=True, use_gpu=True, profile=False,
                                 log_experiment=False)
    # model training and selection
    top10 = compare_models(n_select=10)
    st.write(top10)
    top5 = top10[0:4]
    # tune top 5 base models
    grid_a = pull()
    tuned_top5 = [tune_model(i, fold=5, optimize='F1', search_library='scikit-optimize') for i in top5]
    grid_b = pull()
    stacker = stack_models(estimator_list=top5[1:], meta_model=top5[0])

        #
        return (stacker, grid_a, grid_b)
    else:
        best = supervised.compare_models(sort=metrica, n_select=3)
        grid = supervised.pull()
        return (best, grid, grid)
class TestTraining:
    """Test pycaret run on venv without errors"""

    def test_Pycaret(self)
        logger = logging.getLogger(__name__)
        logger.info('testing pycaret training')
        df = load_data(input_filepath)

        s