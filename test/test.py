# %%
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sys
from pathlib import Path
import unittest

# Asumiendo que tus funciones específicas están en src, pero necesitarías importarlas específicamente.

root_path = Path(__file__).resolve().parent.parent
sys.path.append(str(root_path))

import src 

# Importando los datos. Asegúrate de que la ruta relativa funcione desde la ubicación de este script.
data_path = root_path.joinpath('data/test_data.csv')
data = pd.read_csv(data_path, index_col=0, parse_dates=True)

lags = {
    'pi': {
        'lag': 1,
        'nowcast': src.pi_nowcast_
    },
    'pi_star': {
        'lag': 1,
        'nowcast': src.pi_nowcast_
    },
    'Y': {
        'lag': 1,
        'nowcast': src.gap_nowcast_
    },
    'Y_star': {
        'lag': 2,
        'nowcast': src.gap_nowcast_
    }
}

# %%
data
# %%
data_preprocessed = src.preprocess_data_(
    data.drop(columns=['S']),
    data.S,
    lags=lags
)
len(data_preprocessed)
# %%
data_nowcasted = src.nowcasting_(
    data.drop(columns=['S']),
    data.S,
    lags=lags
)
len(data_nowcasted)
# %%

class TestSrcPackage(unittest.TestCase):
    
    
    def test_y_transform(self):
        y = data.S 
        tmp = src.y_transform_(y)
        self.assertIsInstance(tmp, pd.Series)
        self.assertEqual(len(tmp), len(y))
        self.assertEqual(tmp.index[0], y.index[0])
        self.assertEqual(tmp.index[-1], y.index[-1])
        
    
    def test_y_gap(self):
        y = data.Y
        tmp = src.y_gap_(y)
        self.assertIsInstance(tmp, pd.Series)
        self.assertEqual(len(tmp), len(y))
        self.assertEqual(tmp.index[0], y.index[0])
        self.assertEqual(tmp.index[-1], y.index[-1])
        
        
    def test_pi_nowcast(self):
        pi = data.pi
        tmp = src.pi_nowcast_(pi, steps=1)
        self.assertIsInstance(tmp, pd.Series)
        self.assertEqual(len(tmp), len(pi))
        self.assertEqual(tmp.index[0], pi.index[0])
        self.assertEqual(tmp.index[-1], pi.index[-1])

        
    def test_gap_nowcast(self):
        y = data.Y
        tmp = src.gap_nowcast_(y)
        self.assertIsInstance(tmp, pd.Series)
        self.assertEqual(len(tmp), len(y))
        self.assertEqual(tmp.index[0], y.index[0])
        self.assertEqual(tmp.index[-1], y.index[-1])
     
        
    def test_preprocess_data(self):
        tmp = src.preprocess_data_(
            data.drop(columns=['S']),
            data.S,
            lags=lags
        )
        self.assertIsInstance(tmp, pd.DataFrame)
        self.assertEqual(len(tmp), len(data) - 3)
        self.assertLess(tmp.index.max(), data.index.max())
        self.assertGreater(tmp.index.min(), data.index.min())
        
    
    def test_nowcasting(self):
        transformed = src.preprocess_data_(
            data.drop(columns=['S']),
            data.S,
            lags=lags
        )
        
        nowcasted = src.nowcasting_(
            data.drop(columns=['S']),
            data.S,
            lags=lags
        )
        self.assertIsInstance(nowcasted, pd.DataFrame)
        self.assertEqual(len(nowcasted), len(data) - 4)
        self.assertEqual(len(nowcasted), len(transformed) - 1)
        self.assertLess(nowcasted.index.max(), data.index.max())
        self.assertEqual(nowcasted.index.max(), transformed.index.max())
        self.assertGreater(nowcasted.index.min(), data.index.min())
        self.assertGreater(nowcasted.index.min(), transformed.index.min())
        

# %%
if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)  # Modificado para permitir la ejecución en notebooks.
